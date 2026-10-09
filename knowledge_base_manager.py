# knowledge_base_manager.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class KnowledgeBaseManager:
    """Manages knowledge base entries using Supabase for persistence."""

    def __init__(self):
        url = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        key = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("Supabase URL or key not found in environment.")
        self.client: Client = create_client(url, key)
        self._entries: list[dict] = []
        print("✅ KnowledgeBaseManager connected to Supabase.")

    def _load_entries(self) -> list[dict]:
        """Loads all knowledge base entries from Supabase."""
        result = self.client.table("knowledge_base_entries").select("*").execute()
        if result.data:
            return result.data
        return []

    def search(self, query: str, k: int = 3) -> str:
        """Searches the knowledge base for relevant entries using keyword matching.
        Returns concatenated content from the top k matches."""
        entries = self._entries or self._load_entries()
        self._entries = entries

        if not entries:
            return ""

        query_words = set(query.lower().split())
        scored = []
        for entry in entries:
            content_lower = entry["content"].lower()
            content_words = set(content_lower.split())
            score = len(query_words & content_words)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        if not top:
            return ""

        return "\n\n".join(entry["content"] for _, entry in top)

    def add_entry(self, source: str, content: str, category: str = None) -> dict:
        """Adds a new knowledge base entry."""
        data = {"source": source, "content": content, "category": category}
        result = self.client.table("knowledge_base_entries").insert(data).execute()
        if result.data:
            self._entries = []
        return result.data[0] if result.data else {}

    def get_all_entries(self) -> list[dict]:
        """Returns all knowledge base entries."""
        return self._entries or self._load_entries()


kb_manager = KnowledgeBaseManager()
