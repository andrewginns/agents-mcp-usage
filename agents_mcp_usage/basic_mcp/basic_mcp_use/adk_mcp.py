import asyncio
import os
import logfire

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# Set API key for Google AI API from environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Configure logging if LOGFIRE_TOKEN is set
logfire.configure(send_to_logfire="if-token-present", service_name="adk-basic-mcp")
logfire.instrument_mcp()


async def main(query: str = "Greet Andrew and give him the current time") -> None:
    """
    Main function to run the agent

    Args:
        query (str): The query to run the agent with
    """
    # Set up MCP server connection
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "mcp_servers/example_server.py", "stdio"],
    )

    tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)
    print(f"Connected to MCP server. Found {len(tools)} tools.")

    # Create the agent
    root_agent = LlmAgent(
        model="gemini-2.5-pro-preview-03-25",
        name="mcp_pydantic_assistant",
        tools=tools,
    )

    # Set up session
    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name="mcp_pydantic_app",
        user_id="aginns",
    )

    # Create the runner
    runner = Runner(
        app_name="mcp_pydantic_app",
        agent=root_agent,
        session_service=session_service,
    )

    # Run the agent with a query
    content = types.Content(role="user", parts=[types.Part(text=query)])

    print("Running agent...")
    try:
        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )

        async for event in events_async:
            print(f"Event received: {event}")
    finally:
        print("Closing MCP server connection...")
        await exit_stack.aclose()
        print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
