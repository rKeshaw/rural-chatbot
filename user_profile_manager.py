# user_profile_manager.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class UserProfileManager:
    """Manages user profiles using Supabase for persistence."""

    def __init__(self):
        url = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        key = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("Supabase URL or key not found in environment.")
        self.client: Client = create_client(url, key)
        print("✅ UserProfileManager connected to Supabase.")

    def get_or_create_profile(self, session_id: str) -> dict:
        """Finds a user profile by session_id or creates a new one."""
        result = self.client.table("user_profiles").select("*").eq("session_id", session_id).maybe_single().execute()

        if result and result.data:
            return result.data

        profile_data = {
            "session_id": session_id,
            "location": None,
            "language": "Hinglish",
            "interests": [],
        }
        insert_result = self.client.table("user_profiles").insert(profile_data).execute()
        return insert_result.data[0] if insert_result.data else profile_data

    def update_profile(self, session_id: str, new_data: dict) -> dict:
        """Updates a user's profile with new data."""
        new_data["updated_at"] = "now()"
        self.client.table("user_profiles").update(new_data).eq("session_id", session_id).execute()
        updated = self.get_or_create_profile(session_id)
        print(f"Updated profile for {session_id}: {updated}")
        return updated


user_profile_manager = UserProfileManager()
