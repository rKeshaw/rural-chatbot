# tests/test_config.py
"""Tests for config.py — verifies all expected config values are present and valid."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    KNOWLEDGE_BASE_DIR,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_ID,
    WHISPER_MODEL_NAME,
    TTS_VOICE_DIR,
    SPEAKER_VOICE_DIR,
    GROQ_WHISPER_MODEL_ID,
    ELEVENLABS_VOICE_ID,
    LLAMA_GUARD_MODEL_ID,
    TTS_MODEL_NAME,
)


def test_groq_model_id_is_valid():
    assert GROQ_MODEL_ID == "openai/gpt-oss-20b"

def test_whisper_model_id_is_correct():
    """The Whisper model should be a speech transcription model, not a chat model."""
    assert "whisper" in GROQ_WHISPER_MODEL_ID.lower(), \
        f"Expected GROQ_WHISPER_MODEL_ID to contain 'whisper', got '{GROQ_WHISPER_MODEL_ID}'"
    assert "mixtral" not in GROQ_WHISPER_MODEL_ID.lower(), \
        f"GROQ_WHISPER_MODEL_ID should not be 'mixtral' (a chat model), got '{GROQ_WHISPER_MODEL_ID}'"

def test_llama_guard_model_exists():
    assert LLAMA_GUARD_MODEL_ID is not None
    assert len(LLAMA_GUARD_MODEL_ID) > 0

def test_tts_model_name_exists():
    assert TTS_MODEL_NAME is not None
    assert "xtts" in TTS_MODEL_NAME.lower()

def test_directories_are_strings():
    assert isinstance(KNOWLEDGE_BASE_DIR, str)
    assert isinstance(TTS_VOICE_DIR, str)
    assert isinstance(SPEAKER_VOICE_DIR, str)

def test_embedding_model_name():
    assert EMBEDDING_MODEL_NAME is not None
    assert len(EMBEDDING_MODEL_NAME) > 0
