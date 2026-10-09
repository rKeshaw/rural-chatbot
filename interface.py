# interface.py - The Final, Correct, and Simplified Version
import gradio as gr
import uuid
from threading import Thread
from llm_handler import llm_handler
from langchain_core.messages import HumanMessage, AIMessage
from agent import agent_app
from tts_handler import tts_handler
from user_profile_manager import user_profile_manager

class AssistantInterface:
    def __init__(self):
        pass

    def text_to_speech(self, text: str):
        """Initiates speech synthesis in a separate thread using the unified handler."""
        if text and text.strip():
            Thread(target=tts_handler.speak, args=(text,)).start()

    def predict(self, audio_input, text_input, chat_history, session_id):
        """Main prediction function that handles conversation history and user profiles."""
        if audio_input is None and (not text_input or not text_input.strip()):
            return chat_history, text_input, "", session_id

        query = ""
        if audio_input is not None:
            query = llm_handler.transcribe_audio(audio_input)
        elif text_input and text_input.strip():
            query = text_input.strip()

        if not query:
            return chat_history, "", "", session_id

        # Ensure a session_id exists
        if not session_id:
            session_id = str(uuid.uuid4())

        # Load or create user profile
        try:
            user_profile_manager.get_or_create_profile(session_id)
        except Exception as e:
            print(f"Profile init warning: {e}")

        # Convert Gradio chat history to LangChain message format
        conversation_history = []
        for message in chat_history:
            if message["role"] == "user":
                conversation_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                conversation_history.append(AIMessage(content=message["content"]))

        conversation_history.append(HumanMessage(content=query))

        final_state = agent_app.invoke({"messages": conversation_history, "session_id": session_id})
        full_response = final_state['messages'][-1].content

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": ""})

        for char in full_response:
            chat_history[-1]["content"] += char
            yield chat_history, "", full_response, session_id

        # Safety check
        if not llm_handler.is_response_safe(user_query=query, assistant_response=full_response):
            safe_response = "Maaf kijiye, main is vishay par charcha nahi kar sakta."
            chat_history[-1]["content"] = safe_response
            full_response = safe_response
            yield chat_history, "", full_response, session_id

    def build_ui(self):
        """Builds the Gradio Blocks UI with the 'Read Aloud' button."""
        with gr.Blocks(title="Gram Sahayak") as chat_ui:
            gr.Markdown("# 🌾 Gram Sahayak")
            last_response_state = gr.State("")
            session_id_state = gr.State("")
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                textbox = gr.Textbox(label="Type your question here:", placeholder="PM Kisan yojana kya hai?", scale=3)
                audiobox = gr.Audio(sources=["microphone"], type="filepath", label="Or, speak your question here:", scale=1)
            with gr.Row():
                read_aloud_button = gr.Button("🔊 Read Aloud")

            textbox.submit(self.predict, [audiobox, textbox, chatbot, session_id_state], [chatbot, textbox, last_response_state, session_id_state])
            audiobox.stop_recording(self.predict, [audiobox, textbox, chatbot, session_id_state], [chatbot, textbox, last_response_state, session_id_state])
            read_aloud_button.click(self.text_to_speech, [last_response_state], None)

        return chat_ui


