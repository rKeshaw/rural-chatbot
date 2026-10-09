# tests/test_user_profile_manager.py
"""Tests for user_profile_manager.py — Supabase-backed profile CRUD."""
import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_profile_manager import UserProfileManager


class TestUserProfileManager:
    def setup_method(self):
        self.manager = UserProfileManager()
        self.test_session_id = f"test-{uuid.uuid4()}"

    def test_get_or_create_creates_new_profile(self):
        profile = self.manager.get_or_create_profile(self.test_session_id)
        assert profile is not None
        assert profile["session_id"] == self.test_session_id
        assert profile["language"] == "Hinglish"
        assert profile["interests"] == []

    def test_get_or_create_retrieves_existing(self):
        self.manager.get_or_create_profile(self.test_session_id)
        profile = self.manager.get_or_create_profile(self.test_session_id)
        assert profile["session_id"] == self.test_session_id

    def test_update_profile_location(self):
        self.manager.get_or_create_profile(self.test_session_id)
        updated = self.manager.update_profile(self.test_session_id, {"location": "Patna"})
        assert updated["location"] == "Patna"

    def test_update_profile_interests(self):
        self.manager.get_or_create_profile(self.test_session_id)
        updated = self.manager.update_profile(self.test_session_id, {"interests": ["agriculture", "weather"]})
        assert "agriculture" in updated["interests"]
        assert "weather" in updated["interests"]

    def test_update_profile_language(self):
        self.manager.get_or_create_profile(self.test_session_id)
        updated = self.manager.update_profile(self.test_session_id, {"language": "Hindi"})
        assert updated["language"] == "Hindi"

    def cleanup_method(self):
        """Clean up test profiles."""
        try:
            self.manager.client.table("user_profiles").delete().eq("session_id", self.test_session_id).execute()
        except Exception:
            pass
