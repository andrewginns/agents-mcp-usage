import os
from typing import Any

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge

load_dotenv()

# Configure logging to logfire if LOGFIRE_TOKEN is set in environment
logfire.configure(
    send_to_logfire="if-token-present",
    service_name="evals",
)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

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
    server = MCPServerStdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", test_dir],
    )

    # Create a new agent with its own MCP server instances
    agent = Agent(model_name, mcp_servers=[server])

    # Use the agent to accomplish the task
    async with agent.run_mcp_servers():
        result = await agent.run(inputs.question)

    return OutputResponse(output=result.output)


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
        find_target_quote, name=f"{model_name}-pydantic_ai-find_target_quote-evals"
    )
    report.print(
        include_input=False, include_expected_output=False, include_output=False
    )


# This block still allows the script to be run directly
if __name__ == "__main__":
    main()
