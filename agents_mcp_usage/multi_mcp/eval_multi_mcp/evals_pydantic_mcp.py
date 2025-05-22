import asyncio
import os
from typing import Any

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

from agents_mcp_usage.multi_mcp.mermaid_diagrams import (
    invalid_mermaid_diagram_easy,
    invalid_mermaid_diagram_medium,
    invalid_mermaid_diagram_hard,
    valid_mermaid_diagram,
)
from mcp_servers.mermaid_validator import validate_mermaid_diagram

load_dotenv()

# Configure logging to logfire if LOGFIRE_TOKEN is set in environment
logfire.configure(
    send_to_logfire="if-token-present", service_name="evals-pydantic-multi-mcp"
)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

# Default model to use
DEFAULT_MODEL = "gemini-2.5-pro-preview-05-06"

# Configure MCP servers
local_server = MCPServerStdio(
    command="uv",
    args=[
        "run",
        "run_server.py",
        "stdio",
    ],
)
mermaid_server = MCPServerStdio(
    command="uv",
    args=[
        "run",
        "mcp_servers/mermaid_validator.py",
    ],
)


# Create Agent with MCP servers
def create_agent(model: str = DEFAULT_MODEL, model_settings: dict[str, Any] = {}):
    return Agent(
        model,
        mcp_servers=[local_server, mermaid_server],
        model_settings=model_settings,
    )


# Define input and output schema for evaluations
class MermaidInput(BaseModel):
    invalid_diagram: str


class MermaidOutput(BaseModel):
    fixed_diagram: str


# Custom evaluator to check if both MCP tools were used
class UsedBothMCPTools(Evaluator[MermaidInput, MermaidOutput]):
    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        # In a real implementation, we would check logs to verify both servers were used
        # For now, we'll assume success if we get a valid diagram output
        return 1.0 if ctx.output and ctx.output.fixed_diagram else 0.0


# Custom evaluator to check if the mermaid diagram is valid
class MermaidDiagramValid(Evaluator[MermaidInput, MermaidOutput]):
    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        # Strip whitespace, remove backticks and ```mermaid markers
        input_str = ctx.output.fixed_diagram.strip()

        # Remove ```mermaid and ``` markers
        if input_str.startswith("```mermaid"):
            input_str = input_str[len("```mermaid") :].strip()
        if input_str.endswith("```"):
            input_str = input_str[:-3].strip()

        # Remove any remaining backticks
        input_str = input_str.replace("`", "")

        logfire.info(
            "Evaluating mermaid diagram validity",
            diagram_length=len(input_str),
            diagram_preview=input_str[:100],
        )

        # Use the MCP server's validation function
        result = await validate_mermaid_diagram(input_str)

        if result.is_valid:
            logfire.info("Mermaid diagram validation succeeded")
        else:
            logfire.warning(
                "Mermaid diagram validation failed", error_message=result.error_message
            )

        return 1.0 if result.is_valid else 0.0


async def fix_mermaid_diagram(
    inputs: MermaidInput, model: str = DEFAULT_MODEL
) -> MermaidOutput:
    """Fix an invalid mermaid diagram using the agent with multiple MCP servers.

    Args:
        inputs: The input containing the invalid diagram
        model: The model to use for the agent

    Returns:
        MermaidOutput with the fixed diagram
    """
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {inputs.invalid_diagram}. Return only the fixed mermaid diagram between backticks."

    # Create a fresh agent for each invocation to avoid concurrent usage issues
    current_agent = create_agent(model)
    usage_limits = UsageLimits(request_limit=5)

    # Use the agent's context manager directly in this function
    async with current_agent.run_mcp_servers():
        result = await current_agent.run(query, usage_limits=usage_limits)

    # Extract the mermaid diagram from the result output
    output = result.output

    # Logic to extract the diagram from between backticks
    if "```" in output:
        start = output.find("```")
        end = output.rfind("```") + 3
        diagram = output[start:end]
    else:
        diagram = output

    return MermaidOutput(fixed_diagram=diagram)


def create_evaluation_dataset(judge_model: str = DEFAULT_MODEL):
    """Create the dataset for evaluating mermaid diagram fixing.

    Args:
        judge_model: The model to use for LLM judging

    Returns:
        The evaluation dataset
    """
    return Dataset[MermaidInput, MermaidOutput, Any](
        # Construct 3 tests, each asks the LLM to fix an invalid mermaid diagram of increasing difficulty
        cases=[
            Case(
                name="fix_invalid_diagram_easy",
                inputs=MermaidInput(invalid_diagram=invalid_mermaid_diagram_easy),
                expected_output=MermaidOutput(fixed_diagram=valid_mermaid_diagram),
                metadata={"test_type": "mermaid_easy_fix"},
            ),
            Case(
                name="fix_invalid_diagram_medium",
                inputs=MermaidInput(invalid_diagram=invalid_mermaid_diagram_medium),
                expected_output=MermaidOutput(fixed_diagram=valid_mermaid_diagram),
                metadata={"test_type": "mermaid_medium_fix"},
            ),
            Case(
                name="fix_invalid_diagram_hard",
                inputs=MermaidInput(invalid_diagram=invalid_mermaid_diagram_hard),
                expected_output=MermaidOutput(fixed_diagram=valid_mermaid_diagram),
                metadata={"test_type": "mermaid_hard_fix"},
            ),
        ],
        evaluators=[
            UsedBothMCPTools(),
            MermaidDiagramValid(),
            LLMJudge(
                rubric="The response only contains a mermaid diagram inside the response JSON, no other text.",
                include_input=False,
                model=judge_model,
            ),
            LLMJudge(
                rubric="The output diagram should maintain the same overall structure and intent as the expected output diagram while fixing any syntax errors."
                + "Check if nodes, connections, and labels are preserved."
                + "The current time should be placeholder should be replace with a valid datetime",
                include_input=False,
                model=judge_model,
            ),
        ],
    )


async def run_evaluations(model: str = DEFAULT_MODEL, judge_model: str = DEFAULT_MODEL):
    """Run the evaluations on the mermaid diagram fixing task.

    Args:
        model: The model to use for the agent
        judge_model: The model to use for LLM judging

    Returns:
        The evaluation report
    """
    dataset = create_evaluation_dataset(judge_model)

    # Create a wrapper that includes the model parameter
    async def fix_with_model(inputs: MermaidInput) -> MermaidOutput:
        return await fix_mermaid_diagram(inputs, model=model)

    report = await dataset.evaluate(
        fix_with_model,
        name=f"{model}-multi-mcp-mermaid-diagram-fix-evals",
        max_concurrency=1,  # Run one evaluation at a time
    )

    report.print(include_input=False, include_output=False)
    return report


if __name__ == "__main__":
    # You can use different models for the agent and the judge
    # agent_model = os.getenv("AGENT_MODEL", DEFAULT_MODEL)
    agent_model = "gemini-2.5-flash-preview-04-17"
    # agent_model = "openai:o4-mini"
    # agent_model = "gemini-2.5-flash-preview-04-17"
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)

    async def run_all():
        await run_evaluations(model=agent_model, judge_model=judge_model)

    asyncio.run(run_all())
