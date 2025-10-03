# user_profile_manager.py
from tinydb import TinyDB, Query

# Initialize the database. This will create a 'user_db.json' file.
db = TinyDB('user_db.json')
User = Query()

def get_or_create_profile(session_id: str) -> dict:
    """Finds a user profile by session_id or creates a new one."""
    result = db.search(User.session_id == session_id)
    if not result:
        # If no profile exists, create a default one
        profile_data = {
            "session_id": session_id,
            "location": None,
            "language": "Hinglish",
            "interests": []
        }
        db.insert(profile_data)
        return profile_data
    return result[0]

def update_profile(session_id: str, new_data: dict):
    """Updates a user's profile with new data."""
    profile = get_or_create_profile(session_id)
    profile.update(new_data)
    db.update(profile, User.session_id == session_id)
    print(f"Updated profile for {session_id}: {profile}")
