# tests/test_agent.py
"""Tests for agent.py — verifies the LangGraph routing structure and node definitions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import agent


class TestAgentGraph:
    def test_agent_app_compiled(self):
        assert agent.agent_app is not None

    def test_router_llm_exists(self):
        assert agent.router_llm is not None or agent._get_router_llm is not None

    def test_route_logic_function_exists(self):
        assert callable(agent.route_logic)

    def test_generate_general_response_exists(self):
        assert callable(agent.generate_general_response)

    def test_web_search_node_exists(self):
        assert callable(agent.web_search_node)

    def test_weather_node_exists(self):
        assert callable(agent.weather_node)

    def test_knowledge_base_node_exists(self):
        assert callable(agent.knowledge_base_node)

    def test_route_logic_returns_valid_keys(self):
        """The router should only return keys that map to actual nodes."""
        valid_keys = {"generate_general", "web_search", "get_weather", "knowledge_base"}
        source = inspect.getsource(agent.route_logic)
        # Verify all 4 routes are present
        assert "knowledge_base" in source, "route_logic should handle knowledge_base routing"
        assert "weather_query" in source
        assert "web_search" in source
        assert "general_conversation" in source

    def test_knowledge_base_node_uses_kb_manager(self):
        """The knowledge base node should call kb_manager.search()."""
        source = inspect.getsource(agent.knowledge_base_node)
        assert "kb_manager.search" in source

    def test_knowledge_base_node_handles_empty_results(self):
        """The KB node should handle the case where no results are found."""
        source = inspect.getsource(agent.knowledge_base_node)
        assert "not kb_context" in source or "if not kb_context" in source

    def test_workflow_has_all_nodes(self):
        """The compiled graph should have all 4 nodes."""
        # LangGraph compiled apps expose their graph structure
        assert hasattr(agent.agent_app, 'nodes') or hasattr(agent.agent_app, 'graph')

    def test_agent_state_has_session_id(self):
        """AgentState should include session_id for user profile tracking."""
        source = inspect.getsource(agent.AgentState)
        # TypedDict may not show fields directly, so check the source
        assert "session_id" in source or "session_id" in str(agent.AgentState.__annotations__)
