# tests/test_knowledge_base_manager.py
"""Tests for knowledge_base_manager.py — Supabase-backed knowledge search."""
import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base_manager import KnowledgeBaseManager


class TestKnowledgeBaseManager:
    def setup_method(self):
        self.manager = KnowledgeBaseManager()
        self.test_source = f"test_source_{uuid.uuid4()}"

    def test_search_returns_string(self):
        result = self.manager.search("PM Kisan yojana", k=3)
        assert isinstance(result, str)

    def test_search_pm_kisan_finds_results(self):
        result = self.manager.search("PM Kisan scheme kya hai", k=3)
        assert len(result) > 0
        assert "PM Kisan" in result or "6000" in result

    def test_search_ayushman_bharat_finds_results(self):
        result = self.manager.search("Ayushman Bharat health insurance", k=3)
        assert len(result) > 0
        assert "Ayushman" in result or "5 lakh" in result

    def test_search_mgnrega_finds_results(self):
        result = self.manager.search("MGNREGA 100 days employment", k=3)
        assert len(result) > 0
        assert "MGNREGA" in result or "100 days" in result or "NREGA" in result

    def test_search_irrelevant_query_returns_empty(self):
        result = self.manager.search("xyzqwerty unrelated nonsense query", k=3)
        assert result == ""

    def test_search_kcc_finds_results(self):
        result = self.manager.search("Kisan Credit Card loan", k=3)
        assert len(result) > 0
        assert "Kisan Credit Card" in result or "KCC" in result

    def test_add_and_retrieve_entry(self):
        entry = self.manager.add_entry(self.test_source, "This is a test knowledge entry about test schemes.", "test")
        assert entry is not None
        assert "content" in entry

        # Force reload and search
        self.manager._entries = []
        result = self.manager.search("test knowledge entry about test schemes", k=1)
        assert "test knowledge entry" in result

    def test_get_all_entries(self):
        entries = self.manager.get_all_entries()
        assert isinstance(entries, list)
        assert len(entries) > 0

    def cleanup_method(self):
        try:
            self.manager.client.table("knowledge_base_entries").delete().eq("source", self.test_source).execute()
        except Exception:
            pass
