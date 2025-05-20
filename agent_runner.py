from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

try:
    # Setup tools
    search = TavilySearchResults(max_results=2)
    tools = [search]

    # Setup model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Setup memory
    memory = MemorySaver()

    # Create agent executor
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # Config for memory/thread
    config = {"configurable": {"thread_id": "api-thread"}}
    setup_error = None
except Exception as setup_exc:
    agent_executor = None
    config = None
    setup_error = setup_exc

def run_agent(message: str):
    """Run the agent with a single message and return the response messages."""
    if setup_error is not None:
        raise RuntimeError(f"Agent setup failed: {setup_error}")
    try:
        response = agent_executor.invoke({"messages": [HumanMessage(content=message)]}, config)
        return response["messages"]
    except Exception as e:
        raise RuntimeError(f"Agent invocation failed: {e}")
