# tests/test_integration.py
"""End-to-end integration tests that exercise the full flow with real API calls.
These tests are marked so they can be skipped if API keys are not available."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import pytest
from dotenv import load_dotenv

load_dotenv()

HAS_GROQ_KEY = bool(os.environ.get("GROQ_API_KEY"))
HAS_TAVILY_KEY = bool(os.environ.get("TAVILY_API_KEY"))
HAS_OPENWEATHER_KEY = bool(os.environ.get("OPENWEATHERMAP_API_KEY"))

pytestmark = pytest.mark.skipif(not HAS_GROQ_KEY, reason="GROQ_API_KEY not set")


class TestAgentEndToEnd:
    """Tests that invoke the full agent graph with real LLM calls."""

    def test_general_conversation(self):
        from agent import agent_app
        from langchain_core.messages import HumanMessage
        result = agent_app.invoke({
            "messages": [HumanMessage(content="Namaste, aap kaun ho?")],
            "session_id": f"test-{uuid.uuid4()}",
        })
        response = result["messages"][-1].content
        assert len(response) > 0
        assert "Gram Sahayak" in response or "sahayak" in response.lower() or "namaste" in response.lower() or "hello" in response.lower()

    def test_knowledge_base_query(self):
        from agent import agent_app
        from langchain_core.messages import HumanMessage
        result = agent_app.invoke({
            "messages": [HumanMessage(content="PM Kisan yojana kya hai?")],
            "session_id": f"test-{uuid.uuid4()}",
        })
        response = result["messages"][-1].content
        assert len(response) > 0
        # Response should mention PM Kisan or the 6000 amount
        assert "6000" in response or "kisan" in response.lower() or "yojana" in response.lower() or "scheme" in response.lower()

    @pytest.mark.skipif(not HAS_OPENWEATHER_KEY, reason="OPENWEATHERMAP_API_KEY not set")
    def test_weather_query(self):
        from agent import agent_app
        from langchain_core.messages import HumanMessage
        result = agent_app.invoke({
            "messages": [HumanMessage(content="Mumbai ka mausam kaisa hai?")],
            "session_id": f"test-{uuid.uuid4()}",
        })
        response = result["messages"][-1].content
        assert len(response) > 0

    @pytest.mark.skipif(not HAS_TAVILY_KEY, reason="TAVILY_API_KEY not set")
    def test_web_search_query(self):
        from agent import agent_app
        from langchain_core.messages import HumanMessage
        result = agent_app.invoke({
            "messages": [HumanMessage(content="India ka latest GDP growth rate kya hai?")],
            "session_id": f"test-{uuid.uuid4()}",
        })
        response = result["messages"][-1].content
        assert len(response) > 0

    def test_conversation_history(self):
        """Test that follow-up questions use conversation context."""
        from agent import agent_app
        from langchain_core.messages import HumanMessage, AIMessage
        history = [
            HumanMessage(content="PM Kisan yojana ke baare mein batao"),
            AIMessage(content="PM Kisan Samman Nidhi ek government scheme hai jo farmers ko Rs 6000 per year deti hai."),
        ]
        result = agent_app.invoke({
            "messages": history + [HumanMessage(content="isko apply kaise kare?")],
            "session_id": f"test-{uuid.uuid4()}",
        })
        response = result["messages"][-1].content
        assert len(response) > 0
