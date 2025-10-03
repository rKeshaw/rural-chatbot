# llm_handler.py
import os
from groq import Groq
from dotenv import load_dotenv
from tavily import TavilyClient
from datetime import datetime
import pytz
from config import GROQ_MODEL_ID, LLAMA_GUARD_MODEL_ID, GROQ_WHISPER_MODEL_ID

load_dotenv()

class LLMHandler:
    def __init__(self):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not groq_api_key or not tavily_api_key:
            raise ValueError("API keys for Groq or Tavily not found in .env file.")
        
        self.client = Groq(api_key=groq_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        print("✅ Groq and Tavily clients initialized.")
    
    def _get_current_time(self):
        """Gets the current time for the IST timezone."""
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.now(ist).strftime('%A, %B %d, %Y, %I:%M %p IST')
    
    # In llm_handler.py, inside the LLMHandler class

    def get_weather(self, city: str) -> str:
        """Gets the current weather for a specified city using OpenWeatherMap API."""
        print(f"---TOOL: Getting weather for {city}---")
        api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        if not api_key:
            return "Weather API key not configured."
        
        # --- THIS IS THE FIX ---
        # Removed the '&lang=hi' parameter which was likely causing the error.
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
        try:
            response = requests.get(url)
            response.raise_for_status() # This will raise an error for bad responses like 401 or 404
            data = response.json()
            
            # Extract and format the key information in English
            description = data['weather'][0]['description']
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            
            # We now create a factual string in English. The main LLM will translate this to Hindi.
            return f"Weather data for {city}: Condition is {description}, Temperature is {temp}°C, Feels like {feels_like}°C."
        except Exception as e:
            print(f"Error getting weather: {e}")
            return f"An error occurred while fetching weather for {city}."

    def transcribe_audio(self, audio_filepath: str) -> str:
        """Transcribes audio using the Groq Whisper API."""
        print(f"Transcribing audio from: {audio_filepath}")
        try:
            with open(audio_filepath, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_filepath, file.read()),
                    model=GROQ_WHISPER_MODEL_ID,
                    language="hi"
                )
            print(f"Transcription successful: {transcription.text}")
            return transcription.text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return "Maaf kijiye, main aapki baat sun nahi paaya."

    def search_the_web(self, query: str) -> str:
        """Searches the web using Tavily for up-to-date information."""
        print(f"Searching the web for: '{query}'")
        try:
            results = self.tavily_client.search(query, search_depth="basic", max_results=3)
            context = "\n\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results['results']])
            return context
        except Exception as e:
            print(f"Error during web search: {e}")
            return "Maaf kijiye, web search karte samay ek samasya aa gayi."

    def get_streaming_response(self, messages: list, context: str = "", user_profile: dict = {}, custom_system_prompt: str = None):
        """Gets a streaming response from the main LLM based on a full conversation history."""
        current_time = self._get_current_time()
        
        if custom_system_prompt:
            system_prompt = custom_system_prompt
        else:
            # This is the main prompt for when the agent is using tools like web search
            system_prompt = f"""# YOUR PERSONA
You are 'Gram Sahayak', a helpful, patient, and knowledgeable AI assistant for Rural India. Your goal is to provide clear, direct, and useful answers in simple Hindi or Hinglish. Always be respectful and encouraging.

# YOUR INSTRUCTIONS
1. Read the entire conversation history to understand the user's need, especially for follow-up questions to resolve context (like 'waha' or 'uska').
2. If [Information from Web Search] is provided, you MUST use it to form your answer.
3. **Summarize the information**. Do not just repeat what you found. Extract the key facts and present them in a natural, conversational sentence.
4. If the web search information is not relevant or not enough to answer, politely say "Is vishay par mujhe sahi jaankari nahi mili."
5. Always use simple language. Avoid difficult or very formal words.
6. BE DIRECT AND CONFIDENT. Do not talk about your own process, limitations, or the quality of the information found. Just provide the best possible answer based on the information.

# EXAMPLE OF A GOOD RESPONSE
[CONVERSATION HISTORY]
- User: "New Delhi mein aaj ka mausam kaisa hai?"
[Information from Web Search]
"Weather in New Delhi, India: Min Temp: 28°C, Max Temp: 39°C, Conditions: Mainly Sunny, Wind: 10 km/h W."
[CORRECT ASSISTANT ANSWER]
New Delhi mein aaj mausam saaf aur dhoop wala rahega. Zyada se zyada तापमान 39°C aur kam se kam 28°C ke aas paas hoga.

# CURRENT CONTEXT
[CURRENT TIME]
{current_time}
[Information from Web Search]
"{context}"
"""
        
        # --- THIS IS THE FIX ---
        # We now combine the system prompt with the FULL message history, instead of just the last message.
        system_message = {"role": "system", "content": system_prompt}
        full_messages = [system_message] + messages
        # ---------------------

        try:
            streamer = self.client.chat.completions.create(
                messages=full_messages, # The full history is now passed to the model
                model=GROQ_MODEL_ID, stream=True
            )
            for chunk in streamer:
                token = chunk.choices[0].delta.content
                if token: yield token
        except Exception as e:
            print(f"An error occurred with the Groq API: {e}")
            yield "Maaf kijiye, abhi ek takneeki samasya aa gayi hai."
            
    def is_response_safe(self, user_query: str, assistant_response: str) -> bool:
        """Checks the assistant's final response for safety."""
        try:
            response = self.client.chat.completions.create(
                model=LLAMA_GUARD_MODEL_ID,
                messages=[{"role": "user", "content": f"[CONVERSATION]\nUser: {user_query}\nAssistant: {assistant_response}\n[/CONVERSATION]\n\n[TASK]Is the assistant's response safe? Answer with only 'safe' or 'unsafe'.[/TASK]"}],
                max_tokens=10, stream=False
            )
            moderation_result = response.choices[0].message.content.lower()
            print(f"Llama Guard check -> Result: '{moderation_result}'")
            return "safe" in moderation_result
        except Exception as e:
            print(f"An error occurred with Llama Guard: {e}")
            return False

llm_handler = LLMHandler()
