# agent.py (Multi-Tool Version)
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from llm_handler import llm_handler

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# This LLM is for the router
router_llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")

def route_logic(state: AgentState):
    """The new router decides between general chat, weather, or web search."""
    print("---AGENT: Deciding next action---")
    last_message = state['messages'][-1].content
    
    router_prompt = f"""You are an expert router. Classify the user's query into one of the following categories: 'general_conversation', 'weather_query', or 'web_search'.
- 'weather_query': For any questions about weather or temperature.
- 'web_search': For questions that require up-to-date facts that are not about weather.
- 'general_conversation': For conversational questions, greetings, or questions about the AI itself.

Query: "{last_message}"
Category:"""
    
    router_response = router_llm.invoke(router_prompt)
    decision = router_response.content.strip().lower()
    print(f"Router decision: {decision}")
    
    if "weather_query" in decision:
        return "get_weather"
    elif "web_search" in decision:
        return "web_search"
    else:
        return "generate_general"

def generate_general_response(state: AgentState):
    """Handles general conversation."""
    # This function remains largely the same
    print("---AGENT: Generating General Response---")
    message_history = [{"role": m.type.replace('human', 'user').replace('ai', 'assistant'), "content": m.content} for m in state['messages']]
    system_prompt = "You are a helpful AI assistant named Gram Sahayak..." # Add your full prompt here
    response_generator = llm_handler.get_streaming_response(messages=message_history, custom_system_prompt=system_prompt)
    full_response = "".join(list(response_generator))
    return {"messages": [AIMessage(content=full_response)]}

def web_search_node(state: AgentState):
    """Handles web search queries."""
    # This is the renamed 'retrieve_web_knowledge' function
    print("---AGENT: Retrieving Web Knowledge---")
    query = state['messages'][-1].content # Simple query for now
    context = llm_handler.search_the_web(query)
    message_history = [{"role": m.type.replace('human', 'user').replace('ai', 'assistant'), "content": m.content} for m in state['messages']]
    response_generator = llm_handler.get_streaming_response(messages=message_history, context=context)
    full_response = "".join(list(response_generator))
    return {"messages": [AIMessage(content=full_response)]}

def weather_node(state: AgentState):
    """New node for handling weather queries."""
    print("---AGENT: Calling Weather Tool---")
    # First, extract the city from the last message
    extractor_prompt = f"From the following user query, extract only the city name. If no city is mentioned, use the context from the conversation history. Conversation: {state['messages']}. Last Query: {state['messages'][-1].content}"
    city_response = router_llm.invoke(extractor_prompt)
    city = city_response.content.strip()
    
    # Call the new weather tool
    weather_data = llm_handler.get_weather(city)
    
    # The tool returns a nicely formatted string, so we can just use that as the response
    return {"messages": [AIMessage(content=weather_data)]}


# Define the new graph structure
workflow = StateGraph(AgentState)

# Add all the worker nodes that perform actions
workflow.add_node("generate_general", generate_general_response)
workflow.add_node("web_search", web_search_node)
workflow.add_node("get_weather", weather_node)

# Set the entry point as a conditional router
# The 'route_logic' function will be called first to decide which worker node to run.
workflow.set_conditional_entry_point(
    route_logic,
    {
        "generate_general": "generate_general",
        "web_search": "web_search",
        "get_weather": "get_weather",
    },
)

# Add the edges from the worker nodes to the end state
workflow.add_edge("generate_general", END)
workflow.add_edge("web_search", END)
workflow.add_edge("get_weather", END)

# Compile the graph
agent_app = workflow.compile()
print("✅ Multi-Tool Agent graph compiled successfully with correct routing.")