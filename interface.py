# interface.py - The Final, Correct, and Simplified Version
import gradio as gr
import os
import torch
from threading import Thread
import pysbd
from llm_handler import llm_handler
from knowledge_base_manager import kb_manager
from langchain_core.messages import HumanMessage, AIMessage
from TTS.api import TTS
from config import TTS_MODEL_NAME, SPEAKER_VOICE_DIR
from agent import agent_app
from tts_handler import tts_handler

class AssistantInterface:
    def __init__(self):
        """
        print("Loading Coqui XTTS text-to-speech model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # With our stable, pinned library versions, we don't need the security workarounds.
        # We can now load the model directly using the older, simpler syntax.
        self.tts_model = TTS(TTS_MODEL_NAME, gpu=(device=="cuda"))
        self.sent_segmenter = pysbd.Segmenter(language="hi", clean=False)
        print("✅ Coqui TTS model loaded.")
        """
        pass

    def text_to_speech(self, text: str):
        """Initiates speech synthesis in a separate thread using the unified handler."""
        if text and text.strip():
            # We use a thread to prevent the UI from freezing
            Thread(target=tts_handler.speak, args=(text,)).start()

    def predict(self, audio_input, text_input, chat_history):
        """Main prediction function that now handles conversation history."""
        if audio_input is None and (not text_input or not text_input.strip()):
            return chat_history, text_input, ""

        query = ""
        if audio_input is not None:
            query = llm_handler.transcribe_audio(audio_input)
        elif text_input and text_input.strip():
            query = text_input.strip()

        if not query:
            return chat_history, "", ""

        # --- Convert Gradio chat history to LangChain message format ---
        conversation_history = []
        for message in chat_history:
            if message["role"] == "user":
                conversation_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                conversation_history.append(AIMessage(content=message["content"]))

        # Add the current user query to the history
        conversation_history.append(HumanMessage(content=query))
        # -----------------------------------------------------------------

        final_state = agent_app.invoke({"messages": conversation_history})
        full_response = final_state['messages'][-1].content

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": ""})

        for char in full_response:
            chat_history[-1]["content"] += char
            yield chat_history, "", full_response

        # Note: The safety check can be enhanced later to include conversation context
        if not llm_handler.is_response_safe(user_query=query, assistant_response=full_response):
            safe_response = "Maaf kijiye, main is vishay par charcha nahi kar sakta."
            chat_history[-1]["content"] = safe_response
            full_response = safe_response
            yield chat_history, "", full_response
        
    def build_ui(self):
        """Builds the Gradio Blocks UI with the 'Read Aloud' button."""
        with gr.Blocks(theme="soft", title="Gram Sahayak") as chat_ui:
            gr.Markdown("# 🌾 Gram Sahayak")
            last_response_state = gr.State("")
            chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")
            with gr.Row():
                textbox = gr.Textbox(label="Type your question here:", placeholder="PM Kisan yojana kya hai?", scale=3)
                audiobox = gr.Audio(sources=["microphone"], type="filepath", label="Or, speak your question here:", scale=1)
            with gr.Row():
                read_aloud_button = gr.Button("🔊 Read Aloud")

            textbox.submit(self.predict, [audiobox, textbox, chatbot], [chatbot, textbox, last_response_state])
            audiobox.stop_recording(self.predict, [audiobox, textbox, chatbot], [chatbot, textbox, last_response_state])
            read_aloud_button.click(self.text_to_speech, [last_response_state], None)
            
        return chat_ui

assistant_interface = AssistantInterface()
chat_ui = assistant_interface.build_ui()
