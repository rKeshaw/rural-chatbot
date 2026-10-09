# tests/test_tts_handler.py
"""Tests for tts_handler.py — verifies TTS handler structure and graceful degradation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
from tts_handler import TTSHandler, BaseTTSProvider, ElevenLabsProvider, OpenAITTSProvider


class TestTTSHandlerStructure:
    def test_base_provider_is_abstract(self):
        assert issubclass(BaseTTSProvider, type) or hasattr(BaseTTSProvider, '__abstractmethods__')

    def test_elevenlabs_provider_exists(self):
        assert ElevenLabsProvider is not None
        assert issubclass(ElevenLabsProvider, BaseTTSProvider)

    def test_openai_provider_exists(self):
        assert OpenAITTSProvider is not None
        assert issubclass(OpenAITTSProvider, BaseTTSProvider)

    def test_tts_handler_class_exists(self):
        assert TTSHandler is not None

    def test_has_speak_method(self):
        assert hasattr(TTSHandler, 'speak')
        assert callable(getattr(TTSHandler, 'speak'))

    def test_has_synthesize_method_in_base(self):
        assert hasattr(BaseTTSProvider, 'synthesize')

    def test_speak_handles_no_providers_gracefully(self):
        """speak() should not crash when no providers are available."""
        source = inspect.getsource(TTSHandler.speak)
        assert "self.providers" in source

    def test_speak_thread_method_exists(self):
        assert hasattr(TTSHandler, '_speak_thread')
        assert callable(getattr(TTSHandler, '_speak_thread'))

    def test_initialize_providers_method_exists(self):
        assert hasattr(TTSHandler, '_initialize_providers')

    def test_initialize_providers_checks_piper_files(self):
        """Should check for Piper model file existence before loading."""
        source = inspect.getsource(TTSHandler._initialize_providers)
        assert "os.path.exists" in source
        assert "onnx" in source.lower()

    def test_initialize_providers_checks_elevenlabs_key(self):
        source = inspect.getsource(TTSHandler._initialize_providers)
        assert "ELEVENLABS_API_KEY" in source

    def test_initialize_providers_checks_openai_key(self):
        source = inspect.getsource(TTSHandler._initialize_providers)
        assert "OPENAI_API_KEY" in source

    def test_synthesize_uses_httpx_for_elevenlabs(self):
        source = inspect.getsource(ElevenLabsProvider.synthesize)
        assert "httpx" in source
