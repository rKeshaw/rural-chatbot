# agent.py (Multi-Tool Version with Knowledge Base + User Profiles)
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from llm_handler import llm_handler
from knowledge_base_manager import kb_manager
from user_profile_manager import user_profile_manager

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    session_id: str

router_llm = None

def _get_router_llm():
    global router_llm
    if router_llm is None:
        router_llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    return router_llm

def route_logic(state: AgentState):
    """The router decides between general chat, weather, web search, or knowledge base."""
    print("---AGENT: Deciding next action---")
    last_message = state['messages'][-1].content

    router_prompt = f"""You are an expert router. Classify the user's query into one of the following categories: 'general_conversation', 'weather_query', 'web_search', or 'knowledge_base'.
- 'weather_query': For any questions about weather or temperature.
- 'web_search': For questions that require up-to-date facts that are not about weather.
- 'knowledge_base': For questions about Indian government schemes like PM Kisan, Ayushman Bharat, MGNREGA, crop insurance, soil health card, Kisan Credit Card, or any agricultural government program.
- 'general_conversation': For conversational questions, greetings, or questions about the AI itself.

Query: "{last_message}"
Category:"""

    router_response = _get_router_llm().invoke(router_prompt)
    decision = router_response.content.strip().lower()
    print(f"Router decision: {decision}")

    if "weather_query" in decision:
        return "get_weather"
    elif "web_search" in decision:
        return "web_search"
    elif "knowledge_base" in decision:
        return "knowledge_base"
    else:
        return "generate_general"

def generate_general_response(state: AgentState):
    """Handles general conversation with user profile context."""
    print("---AGENT: Generating General Response---")
    message_history = [{"role": m.type.replace('human', 'user').replace('ai', 'assistant'), "content": m.content} for m in state['messages']]

    system_prompt = """You are 'Gram Sahayak', a helpful, patient, and knowledgeable AI assistant for Rural India. Your goal is to provide clear, direct, and useful answers in simple Hindi or Hinglish. Always be respectful and encouraging.

# YOUR INSTRUCTIONS
1. Read the entire conversation history to understand the user's need, especially for follow-up questions to resolve context (like 'waha' or 'uska').
2. Always use simple language. Avoid difficult or very formal words.
3. BE DIRECT AND CONFIDENT. Do not talk about your own process, limitations, or the quality of the information found.
4. If the user asks about something you don't know, politely say "Is vishay par mujhe sahi jaankari nahi mili."
"""
    response_generator = llm_handler.get_streaming_response(messages=message_history, custom_system_prompt=system_prompt)
    full_response = "".join(list(response_generator))
    return {"messages": [AIMessage(content=full_response)]}

def web_search_node(state: AgentState):
    """Handles web search queries."""
    print("---AGENT: Retrieving Web Knowledge---")
    query = state['messages'][-1].content
    context = llm_handler.search_the_web(query)
    message_history = [{"role": m.type.replace('human', 'user').replace('ai', 'assistant'), "content": m.content} for m in state['messages']]
    response_generator = llm_handler.get_streaming_response(messages=message_history, context=context)
    full_response = "".join(list(response_generator))
    return {"messages": [AIMessage(content=full_response)]}

def weather_node(state: AgentState):
    """Handles weather queries."""
    print("---AGENT: Calling Weather Tool---")
    extractor_prompt = f"From the following user query, extract only the city name. If no city is mentioned, use the context from the conversation history. Conversation: {state['messages']}. Last Query: {state['messages'][-1].content}"
    city_response = _get_router_llm().invoke(extractor_prompt)
    city = city_response.content.strip()

    weather_data = llm_handler.get_weather(city)
    return {"messages": [AIMessage(content=weather_data)]}

def knowledge_base_node(state: AgentState):
    """Handles queries about government schemes and programs using the knowledge base."""
    print("---AGENT: Searching Knowledge Base---")
    query = state['messages'][-1].content

    kb_context = kb_manager.search(query, k=3)
    print(f"Knowledge base returned {len(kb_context)} characters of context.")

    if not kb_context:
        message_history = [{"role": m.type.replace('human', 'user').replace('ai', 'assistant'), "content": m.content} for m in state['messages']]
        system_prompt = """You are 'Gram Sahayak', a helpful AI assistant for Rural India. Answer in simple Hindi or Hinglish.
The user is asking about a government scheme or program, but no matching information was found in the knowledge base.
Politely tell the user that you don't have information about this specific scheme right now, and suggest they visit the nearest Common Service Center (CSC) or check the official government website."""
        response_generator = llm_handler.get_streaming_response(messages=message_history, custom_system_prompt=system_prompt)
        full_response = "".join(list(response_generator))
        return {"messages": [AIMessage(content=full_response)]}

    message_history = [{"role": m.type.replace('human', 'user').replace('ai', 'assistant'), "content": m.content} for m in state['messages']]
    system_prompt = f"""You are 'Gram Sahayak', a helpful AI assistant for Rural India. Answer in simple Hindi or Hinglish.

# YOUR INSTRUCTIONS
1. Use the following knowledge base information to answer the user's question.
2. Summarize the information in simple, conversational Hindi or Hinglish.
3. Be direct and confident. Do not mention that you found this in a database or knowledge base.
4. If the knowledge base information is not relevant to the question, politely say "Is vishay par mujhe sahi jaankari nahi mili."

# KNOWLEDGE BASE INFORMATION
{kb_context}
"""
    response_generator = llm_handler.get_streaming_response(messages=message_history, custom_system_prompt=system_prompt)
    full_response = "".join(list(response_generator))
    return {"messages": [AIMessage(content=full_response)]}


workflow = StateGraph(AgentState)

workflow.add_node("generate_general", generate_general_response)
workflow.add_node("web_search", web_search_node)
workflow.add_node("get_weather", weather_node)
workflow.add_node("knowledge_base", knowledge_base_node)

workflow.set_conditional_entry_point(
    route_logic,
    {
        "generate_general": "generate_general",
        "web_search": "web_search",
        "get_weather": "get_weather",
        "knowledge_base": "knowledge_base",
    },
)

workflow.add_edge("generate_general", END)
workflow.add_edge("web_search", END)
workflow.add_edge("get_weather", END)
workflow.add_edge("knowledge_base", END)

agent_app = workflow.compile()
print("✅ Multi-Tool Agent graph compiled with general, web search, weather, and knowledge base routing.")
