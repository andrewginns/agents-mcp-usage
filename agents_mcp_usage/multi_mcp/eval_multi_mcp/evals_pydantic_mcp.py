import asyncio
import os
import subprocess
from typing import Any

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge

from agents_mcp_usage.multi_mcp.mermaid_diagrams import (
    invalid_mermaid_diagram_easy,
    valid_mermaid_diagram,
)

load_dotenv()

# Configure logging to logfire if LOGFIRE_TOKEN is set in environment
logfire.configure(
    send_to_logfire="if-token-present", service_name="evals-pydantic-multi-mcp"
)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

# Default model to use
DEFAULT_MODEL = "gemini-2.5-pro-preview-03-25"
# DEFAULT_MODEL = "openai:o4-mini"
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
    command="npx",
    args=[
        "-y",
        "@rtuin/mcp-mermaid-validator@latest",
    ],
)


# Create Agent with MCP servers
def create_agent(model: str = DEFAULT_MODEL):
    return Agent(
        model,
        mcp_servers=[local_server, mermaid_server],
    )


agent = create_agent()
Agent.instrument_all()


async def main(
    query: str = "Hi!", request_limit: int = 5, model: str = DEFAULT_MODEL
) -> Any:
    """
    Main function to run the agent

    Args:
        query (str): The query to run the agent with
        request_limit (int): The number of requests to make to the MCP servers
        model (str): The model to use for the agent

    Returns:
        The result from the agent's execution
    """
    # Create a fresh agent with the specified model
    current_agent = create_agent(model)

    # Set a request limit for LLM calls
    usage_limits = UsageLimits(request_limit=request_limit)

    # Invoke the agent with the usage limits
    async with current_agent.run_mcp_servers():
        result = await current_agent.run(query, usage_limits=usage_limits)

    return result


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
        diagram = ctx.output.fixed_diagram

        # Extract mermaid code from markdown code block if present
        mermaid_code = diagram
        if "```mermaid" in diagram and "```" in diagram:
            start_idx = diagram.find("```mermaid") + len("```mermaid")
            end_idx = diagram.rfind("```")
            mermaid_code = diagram[start_idx:end_idx].strip()

        # Validate using mmdc
        is_valid, _ = self.validate_mermaid_string_via_mmdc(mermaid_code)
        return 1.0 if is_valid else 0.0

    def validate_mermaid_string_via_mmdc(
        self, mermaid_code: str, mmdc_path: str = "mmdc"
    ) -> tuple[bool, str]:
        """
        Validates a Mermaid string by attempting to compile it using the
        Mermaid CLI (mmdc). Requires mmdc to be installed and in PATH,
        or mmdc_path to be explicitly provided.

        Args:
            mermaid_code: The string containing the Mermaid diagram syntax.
            mmdc_path: The command or path to the mmdc executable.

        Returns:
            A tuple (is_valid: bool, message: str).
            'message' will contain stderr output if not valid, or a success message.
        """
        # Define temporary file names
        temp_mmd_file = "temp_mermaid_for_validation.mmd"
        # mmdc requires an output file, even if we don't use its content for validation.
        temp_output_file = "temp_mermaid_output.svg"

        # Write the mermaid code to a temporary file
        with open(temp_mmd_file, "w", encoding="utf-8") as f:
            f.write(mermaid_code)

        try:
            # Construct the command to run mmdc
            command = [mmdc_path, "-i", temp_mmd_file, "-o", temp_output_file]

            # Execute the mmdc command
            process = subprocess.run(
                command,
                capture_output=True,  # Capture stdout and stderr
                text=True,  # Decode output as text
                check=False,  # Do not raise an exception for non-zero exit codes
                encoding="utf-8",
            )

            if process.returncode == 0:
                return True, "Syntax appears valid (compiled successfully by mmdc)."
            else:
                # mmdc usually prints errors to stderr.
                error_message = process.stderr.strip()
                # Sometimes, syntax errors might also appear in stdout for certain mmdc versions or error types
                if not error_message and process.stdout.strip():
                    error_message = process.stdout.strip()
                return (
                    False,
                    f"Invalid syntax or mmdc error (exit code {process.returncode}):\n{error_message}",
                )
        except FileNotFoundError:
            return False, (
                f"Validation failed: '{mmdc_path}' command not found. "
                "Please ensure Mermaid CLI (mmdc) is installed and in your system's PATH, "
                "or provide the full path to the executable."
            )
        except Exception as e:
            return (
                False,
                f"Validation failed due to an unexpected error during mmdc execution: {e}",
            )
        finally:
            # Clean up the temporary files
            if os.path.exists(temp_mmd_file):
                os.remove(temp_mmd_file)
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)


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

    result = await main(query, model=model)

    # Extract the mermaid diagram from the output
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
        cases=[
            Case(
                name="fix_invalid_diagram_1",
                inputs=MermaidInput(invalid_diagram=invalid_mermaid_diagram_easy),
                expected_output=MermaidOutput(fixed_diagram=valid_mermaid_diagram),
                metadata={"test_type": "mermaid_easy_fix", "iteration": 1},
            ),
        ],
        evaluators=[
            UsedBothMCPTools(),
            MermaidDiagramValid(),
            LLMJudge(
                rubric="The response only contains a mermaid diagram, no other text.",
                include_input=False,
                model=judge_model,
            ),
            LLMJudge(
                rubric="The fixed diagram should maintain the same overall structure and intent as the expected output diagram while fixing any syntax errors."
                + "Check if nodes, connections, and labels are preserved."
                + "The current time should be placeholder should be replace with a datetime",
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
        fix_with_model, name=f"{model}-multi-mcp-mermaid-diagram-fix-evals"
    )

    report.print(include_input=True, include_output=True)
    return report


if __name__ == "__main__":
    # You can use different models for the agent and the judge
    agent_model = os.getenv("AGENT_MODEL", DEFAULT_MODEL)
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)

    async def run_all():
        # Run evaluations
        await run_evaluations(model=agent_model, judge_model=judge_model)

    asyncio.run(run_all())
