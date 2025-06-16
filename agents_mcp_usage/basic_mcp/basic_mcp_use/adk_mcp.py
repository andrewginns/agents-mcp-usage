"""ADK-based agent using MCP protocol for tool access."""

import asyncio
import os
import logfire

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
    StdioConnectionParams,
)
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
    """Runs the agent with a given query.

    This function sets up the MCP server, creates an LLM agent, and runs it
    with a specified query. It also handles the cleanup of the MCP server
    connection.

    Args:
        query: The query to run the agent with.
    """
    toolset = None
    try:
        # Set up MCP server connection with enhanced error handling
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "mcp_servers/example_server.py", "stdio"],
        )

        connection_params = StdioConnectionParams(server_params=server_params)
        toolset = MCPToolset(connection_params=connection_params)
        tools = await toolset.get_tools()
        print(f"✓ Connected to MCP server. Found {len(tools)} tools.")

        # Create the agent
        root_agent = LlmAgent(
            model="gemini-2.0-flash",
            name="mcp_adk_assistant",
            tools=tools,
        )

        # Set up session with async service
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="mcp_adk_app",
            user_id="aginns",
        )

        # Create the runner
        runner = Runner(
            app_name="mcp_adk_app",
            agent=root_agent,
            session_service=session_service,
        )

        # Format the query as content
        content = types.Content(role="user", parts=[types.Part(text=query)])

        print("Running agent...")
        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )

        async for event in events_async:
            print(f"Event received: {event}")

    except Exception as e:
        print(f"Error during agent execution: {e}")
        print(f"Error type: {type(e).__name__}")
        raise
    finally:
        # Clean up MCP toolset to prevent asyncio shutdown errors
        if toolset:
            print("Closing MCP server connection...")
            try:
                await toolset.close()
                print("MCP connection closed successfully.")
            except asyncio.CancelledError:
                print("MCP cleanup cancelled - this is normal")
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
        print("Agent execution completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
