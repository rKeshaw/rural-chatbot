# Gram Sahayak

A voice-enabled AI assistant for Rural India, communicating in Hindi and Hinglish.

## Features

- **Text and voice input** — Users can type or speak their questions in Hindi/Hinglish
- **Voice output** — "Read Aloud" button speaks responses using available TTS providers (Piper, ElevenLabs, or OpenAI)
- **General conversation** — Handles greetings, conversational questions, and general knowledge
- **Government scheme knowledge base** — Answers questions about PM Kisan, Ayushman Bharat, MGNREGA, crop insurance, Kisan Credit Card, and more using a Supabase-backed knowledge base
- **Live weather** — Fetches real-time weather for any city using OpenWeatherMap
- **Web search** — Searches the internet for up-to-date information using Tavily
- **Conversation memory** — Remembers context across the conversation for follow-up questions
- **Safety filtering** — Every response is checked by Llama Guard before being shown
- **User profiles** — Per-session user profiles stored in Supabase (location, language, interests)
- **Smart routing** — An LLM-based router decides which tool to use for each query

## Architecture

- **Frontend**: Gradio web UI served via FastAPI
- **LLM**: Groq (Llama 3.1 8B) for responses, Llama Guard for safety
- **Speech-to-Text**: Groq Whisper API
- **Text-to-Speech**: Piper (local), ElevenLabs, or OpenAI (whichever is available)
- **Knowledge Base**: Supabase (PostgreSQL) with keyword-based search
- **User Profiles**: Supabase (PostgreSQL)
- **Agent Orchestration**: LangGraph with conditional routing
- **Web Search**: Tavily API
- **Weather**: OpenWeatherMap API

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env`:
   - `GROQ_API_KEY` — Required for LLM and speech-to-text
   - `TAVILY_API_KEY` — Required for web search
   - `OPENWEATHERMAP_API_KEY` — Required for weather queries
   - `ELEVENLABS_API_KEY` — Optional, for cloud TTS
   - `OPENAI_API_KEY` — Optional, for OpenAI TTS
   - `VITE_SUPABASE_URL` — Supabase project URL (auto-configured)
   - `VITE_SUPABASE_ANON_KEY` — Supabase anon key (auto-configured)
3. Run: `python main.py`
4. Open http://127.0.0.1:8000

## Testing

Run the test suite: `python -m pytest tests/ -v`

Integration tests (requiring API keys) are automatically skipped if keys are not set.
