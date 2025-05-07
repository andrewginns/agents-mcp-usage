import os
from typing import Any

import logfire
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge

load_dotenv()

# Configure logging to logfire if LOGFIRE_TOKEN is set in environment
logfire.configure(
    send_to_logfire="if-token-present",
    service_name="evals",
)
logfire.instrument_mcp()

# Set API key for Google AI API from environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Grant the MCP server access to only a specific directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
test_dir = os.path.join(parent_dir, "mcp_allowed_dir")

# Model to use for the agent
model_name = "gemini-2.5-pro-preview-03-25"


# Define the input prompt schema
class InputPrompt(BaseModel):
    question: str


# Define the output response schema
class OutputResponse(BaseModel):
    output: str


async def find_target_quote(inputs: InputPrompt) -> OutputResponse:
    """Find information in files using the agent with a fresh MCP server for each evaluation.

    Args:
        inputs: The input prompt containing the question

    Returns:
        An OutputResponse with the agent's answer
    """
    # Create a new server instance for each evaluation
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", test_dir],
    )

    tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)

    try:
        # Create the agent
        root_agent = LlmAgent(
            model=model_name,
            name="mcp_adk_assistant",
            tools=tools,
        )

        # Set up session
        session_service = InMemorySessionService()
        session = session_service.create_session(
            app_name="mcp_adk_app",
            user_id="user",
        )

        # Create the runner
        runner = Runner(
            app_name="mcp_adk_app",
            agent=root_agent,
            session_service=session_service,
        )

        # Run the agent with the query
        content = types.Content(role="user", parts=[types.Part(text=inputs.question)])

        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )

        # Extract all text responses from the agent
        async for event in events_async:
            if hasattr(event, "content") and event.content:
                # Extract text content from the response
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        # Store only the last text response
                        result = part.text

        return OutputResponse(output=result)
    finally:
        # Ensure MCP server connection is properly closed
        await exit_stack.aclose()


# Define the dataset of cases to evaluate
quote_dataset = Dataset[InputPrompt, OutputResponse, Any](
    cases=[
        Case(
            name="2024_quote",
            inputs=InputPrompt(
                question=f"Read the contents of files in {test_dir} to find the line that the famous personality says in Motorway's 2024 advertising sponsorship feature. Give me just the line used."
            ),
            expected_output=OutputResponse(output="Oh yeah he's winning!"),
            metadata={"difficulty": "medium"},
        ),
        Case(
            name="find_reference",
            inputs=InputPrompt(
                question=f"Read the contents of files in {test_dir} to find the sub-heading number for 'Cycle lanes and cycle tracks'. Give me just the number."
            ),
            expected_output=OutputResponse(output="140"),
            metadata={"difficulty": "easy"},
        ),
    ],
    evaluators=[
        IsInstance(type_name="OutputResponse"),
        LLMJudge(
            rubric="Output should match expected",
            include_input=True,
            # LLM to use as the judge
            model="gemini-2.5-pro-preview-03-25",
        ),
    ],
)


def main():
    """Main function to run evaluations when module is imported or run directly."""
    # Run evaluations in parallel since each has its own server
    report = quote_dataset.evaluate_sync(
        find_target_quote, name=f"{model_name}-adk-find_target_quote-evals"
    )
    report.print(include_input=False, include_expected_output=True, include_output=True)


if __name__ == "__main__":
    main()
