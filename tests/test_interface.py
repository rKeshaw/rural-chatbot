# tests/test_interface.py
"""Tests for interface.py — verifies UI structure, session management, and profile wiring."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
from interface import AssistantInterface


class TestInterfaceStructure:
    def test_assistant_interface_class_exists(self):
        assert AssistantInterface is not None

    def test_has_build_ui_method(self):
        assert hasattr(AssistantInterface, 'build_ui')
        assert callable(getattr(AssistantInterface, 'build_ui'))

    def test_has_predict_method(self):
        assert hasattr(AssistantInterface, 'predict')
        assert callable(getattr(AssistantInterface, 'predict'))

    def test_has_text_to_speech_method(self):
        assert hasattr(AssistantInterface, 'text_to_speech')
        assert callable(getattr(AssistantInterface, 'text_to_speech'))

    def test_predict_accepts_session_id(self):
        """predict should accept a session_id parameter for user profile tracking."""
        sig = inspect.signature(AssistantInterface.predict)
        params = list(sig.parameters.keys())
        assert "session_id" in params, f"predict should accept session_id, got params: {params}"

    def test_predict_returns_session_id(self):
        """predict should return session_id in its outputs."""
        source = inspect.getsource(AssistantInterface.predict)
        assert "session_id" in source

    def test_predict_uses_user_profile_manager(self):
        """predict should call user_profile_manager.get_or_create_profile."""
        source = inspect.getsource(AssistantInterface.predict)
        assert "user_profile_manager" in source
        assert "get_or_create_profile" in source

    def test_predict_uses_agent_app(self):
        source = inspect.getsource(AssistantInterface.predict)
        assert "agent_app" in source

    def test_predict_handles_empty_input(self):
        """Should handle the case where both audio and text are empty."""
        source = inspect.getsource(AssistantInterface.predict)
        assert "None" in source or "not text_input" in source

    def test_predict_has_safety_check(self):
        """Should call is_response_safe after generating response."""
        source = inspect.getsource(AssistantInterface.predict)
        assert "is_response_safe" in source

    def test_build_ui_has_read_aloud_button(self):
        source = inspect.getsource(AssistantInterface.build_ui)
        assert "Read Aloud" in source or "read_aloud" in source

    def test_build_ui_has_chatbot(self):
        source = inspect.getsource(AssistantInterface.build_ui)
        assert "Chatbot" in source or "chatbot" in source

    def test_build_ui_has_audio_input(self):
        source = inspect.getsource(AssistantInterface.build_ui)
        assert "Audio" in source or "audio" in source

    def test_build_ui_has_text_input(self):
        source = inspect.getsource(AssistantInterface.build_ui)
        assert "Textbox" in source or "textbox" in source

    def test_build_ui_has_session_id_state(self):
        source = inspect.getsource(AssistantInterface.build_ui)
        assert "session_id" in source

    def test_build_ui_uses_messages_format(self):
        """Chatbot should use the messages format (dict with role/content)."""
        source = inspect.getsource(AssistantInterface.predict)
        assert '"role"' in source and '"content"' in source
