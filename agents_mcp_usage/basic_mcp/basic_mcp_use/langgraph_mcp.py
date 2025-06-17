import asyncio
import os
import logfire

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents_mcp_usage.utils import get_mcp_server_path

load_dotenv()

# Configure logging if LOGFIRE_TOKEN is set
logfire.configure(
    send_to_logfire="if-token-present", service_name="langgraph-basic-mcp"
)
logfire.instrument_mcp()


# Create server parameters for stdio connection
server = StdioServerParameters(
    command="uv",
    args=["run", str(get_mcp_server_path("example_server.py")), "stdio"],
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview-06-05", google_api_key=os.getenv("GEMINI_API_KEY")
)


async def main(query: str = "Greet Andrew and give him the current time") -> None:
    """Runs the LangGraph agent with a given query.

    This function connects to the MCP server, loads the tools, creates a
    LangGraph agent, and invokes it with the provided query.

    Args:
        query: The query to run the agent with.
    """
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialise the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke(
                {
                    "messages": query,
                }
            )
            print(agent_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
