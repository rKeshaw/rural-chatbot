# main.py
import uvicorn
from fastapi import FastAPI
import gradio as gr
from interface import AssistantInterface

app = FastAPI(title="Gram Sahayak Main App")

assistant = AssistantInterface()
chat_ui = assistant.build_ui()

app = gr.mount_gradio_app(app, chat_ui, path="/")

if __name__ == "__main__":
    print("🚀 Gram Sahayak is LIVE!")
    print("   Go to http://127.0.0.1:8000 to start talking.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
