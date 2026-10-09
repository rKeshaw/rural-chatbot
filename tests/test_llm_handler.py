# tests/test_llm_handler.py
"""Tests for llm_handler.py — verifies structure, imports, and method signatures."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
from llm_handler import LLMHandler


class TestLLMHandlerStructure:
    def test_requests_imported(self):
        """Verify that the 'requests' module is imported (was previously missing)."""
        import llm_handler
        assert hasattr(llm_handler, 'requests'), "llm_handler should import 'requests' module"

    def test_llm_handler_class_exists(self):
        assert LLMHandler is not None

    def test_has_get_weather_method(self):
        assert hasattr(LLMHandler, 'get_weather')
        assert callable(getattr(LLMHandler, 'get_weather'))
    def test_has_transcribe_audio_method(self):
        assert hasattr(LLMHandler, 'transcribe_audio')
        assert callable(getattr(LLMHandler, 'transcribe_audio'))

    def test_has_search_the_web_method(self):
        assert hasattr(LLMHandler, 'search_the_web')
        assert callable(getattr(LLMHandler, 'search_the_web'))

    def test_has_get_streaming_response_method(self):
        assert hasattr(LLMHandler, 'get_streaming_response')
        assert callable(getattr(LLMHandler, 'get_streaming_response'))

    def test_has_is_response_safe_method(self):
        assert hasattr(LLMHandler, 'is_response_safe')
        assert callable(getattr(LLMHandler, 'is_response_safe'))

    def test_get_weather_uses_requests(self):
        """Verify get_weather source code uses the requests library."""
        source = inspect.getsource(LLMHandler.get_weather)
        assert "requests.get" in source, "get_weather should use requests.get()"

    def test_transcribe_audio_uses_whisper_model(self):
        """Verify transcribe_audio doesn't use a chat model like mixtral."""
        source = inspect.getsource(LLMHandler.transcribe_audio)
        assert "mixtral" not in source.lower(), "transcribe_audio should not reference mixtral (a chat model)"

    def test_get_streaming_response_is_generator(self):
        """Verify get_streaming_response yields (is a generator)."""
        assert inspect.isgeneratorfunction(LLMHandler.get_streaming_response)

    def test_get_weather_returns_string(self):
        """Check the return type annotation."""
        sig = inspect.signature(LLMHandler.get_weather)
        assert sig.return_annotation == str

    def test_get_current_time_returns_ist(self):
        """Verify the time helper uses IST timezone."""
        source = inspect.getsource(LLMHandler._get_current_time)
        assert "Asia/Kolkata" in source
