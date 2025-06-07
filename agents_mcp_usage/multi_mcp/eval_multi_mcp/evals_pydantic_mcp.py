import asyncio
import csv
import os
from datetime import datetime
from typing import Any, Dict, List

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
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
        "mcp_servers/example_server.py",
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
    failure_reason: str = ""  # Add failure reason to track why a case failed
    metrics: Dict[str, Any] = {}  # Add metrics field to capture LLM usage metrics
    tools_used: List[str] = []  # Add field to track which MCP tools were called


# Custom evaluator to check if both MCP tools were used
class UsedBothMCPTools(Evaluator[MermaidInput, MermaidOutput]):
    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        if not ctx.output or not ctx.output.tools_used:
            return 0.0

        # Look for tools from both MCP servers
        has_example_server_tool = any(
            "example" in tool.lower() or "time" in tool.lower()
            for tool in ctx.output.tools_used
        )
        has_mermaid_server_tool = any(
            "mermaid" in tool.lower() or "validate" in tool.lower()
            for tool in ctx.output.tools_used
        )

        if has_example_server_tool and has_mermaid_server_tool:
            return 1.0
        elif has_example_server_tool or has_mermaid_server_tool:
            return 0.5  # Partial credit for using one server
        else:
            return 0.0


# Custom evaluator to detect usage limit failures
class UsageLimitNotExceeded(Evaluator[MermaidInput, MermaidOutput]):
    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        """Check if the case failed due to usage limits being exceeded."""
        if ctx.output and ctx.output.failure_reason == "usage_limit_exceeded":
            logfire.warning(
                "Case failed due to usage limit exceeded",
                case_name=getattr(ctx, "case_name", "unknown"),
            )
            return 0.0
        # Return 1.0 if no usage limit failure occurred
        return 1.0


# Custom evaluator to check if the mermaid diagram is valid
class MermaidDiagramValid(Evaluator[MermaidInput, MermaidOutput]):
    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        # Skip validation if there was a failure
        if ctx.output and ctx.output.failure_reason:
            logfire.info(
                "Skipping diagram validation due to failure",
                failure_reason=ctx.output.failure_reason,
            )
            return 0.0

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
        MermaidOutput with the fixed diagram and captured metrics
    """
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {inputs.invalid_diagram}. Return only the fixed mermaid diagram between backticks."

    # Create a fresh agent for each invocation to avoid concurrent usage issues
    current_agent = create_agent(model)
    usage_limits = UsageLimits(request_limit=5)

    try:
        # Use the agent's context manager directly in this function
        async with current_agent.run_mcp_servers():
            result = await current_agent.run(query, usage_limits=usage_limits)

        # Extract the mermaid diagram from the result output
        usage = result.usage()
        metrics = {
            "requests": usage.requests,
            "request_tokens": usage.request_tokens,
            "response_tokens": usage.response_tokens,
            "total_tokens": usage.total_tokens,
            "details": usage.details or {},
        }

        # Extract tool usage information from agent messages
        tools_used = []
        for message in result.all_messages():
            for part in message.parts:
                if hasattr(part, "tool_name") and part.tool_name:
                    tools_used.append(part.tool_name)

        tools_used = list(dict.fromkeys(tools_used))

        output = result.output

        # Logic to extract the diagram from between backticks
        if "```" in output:
            start = output.find("```")
            end = output.rfind("```") + 3
            diagram = output[start:end]
        else:
            diagram = output

        return MermaidOutput(
            fixed_diagram=diagram, metrics=metrics, tools_used=tools_used
        )

    except UsageLimitExceeded as e:
        logfire.warning(
            "Usage limit exceeded during mermaid diagram fix",
            error_message=str(e),
            model=model,
        )
        # Return empty diagram with failure reason to indicate usage limit failure
        return MermaidOutput(
            fixed_diagram="",
            failure_reason="usage_limit_exceeded",
            metrics={},
            tools_used=[],
        )

    except Exception as e:
        logfire.error(
            "Unexpected error during mermaid diagram fix",
            error_message=str(e),
            error_type=type(e).__name__,
            model=model,
        )
        # Return empty diagram with failure reason to indicate general failure
        return MermaidOutput(
            fixed_diagram="",
            failure_reason=f"error_{type(e).__name__}",
            metrics={},
            tools_used=[],
        )


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
                expected_output=MermaidOutput(
                    fixed_diagram=valid_mermaid_diagram,
                    failure_reason="",
                    metrics={},
                    tools_used=[],
                ),
                metadata={"test_type": "mermaid_easy_fix"},
            ),
            Case(
                name="fix_invalid_diagram_medium",
                inputs=MermaidInput(invalid_diagram=invalid_mermaid_diagram_medium),
                expected_output=MermaidOutput(
                    fixed_diagram=valid_mermaid_diagram,
                    failure_reason="",
                    metrics={},
                    tools_used=[],
                ),
                metadata={"test_type": "mermaid_medium_fix"},
            ),
            Case(
                name="fix_invalid_diagram_hard",
                inputs=MermaidInput(invalid_diagram=invalid_mermaid_diagram_hard),
                expected_output=MermaidOutput(
                    fixed_diagram=valid_mermaid_diagram,
                    failure_reason="",
                    metrics={},
                    tools_used=[],
                ),
                metadata={"test_type": "mermaid_hard_fix"},
            ),
        ],
        evaluators=[
            UsedBothMCPTools(),
            UsageLimitNotExceeded(),
            MermaidDiagramValid(),
            LLMJudge(
                rubric="The response only contains a mermaid diagram inside the fixed_diagram field, no other text. Ignore the metrics, failure_reason, and tools_used fields.",
                include_input=False,
                model=judge_model,
            ),
            LLMJudge(
                rubric="The fixed_diagram field should maintain the same overall structure and intent as the expected output diagram while fixing any syntax errors."
                + "Check if nodes, connections, and labels are preserved."
                + "The current time placeholder should be replaced with a valid datetime. Ignore the metrics, failure_reason, and tools_used fields.",
                include_input=False,
                model=judge_model,
            ),
        ],
    )


def get_timestamp_prefix():
    """Get a timestamp prefix in the format yyyy-mm-dd_H-M-s."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def write_mermaid_results_to_csv(report, model: str, output_dir="./mermaid_results"):
    """Write mermaid evaluation results with metrics to a CSV file.

    Args:
        report: The evaluation report from pydantic_evals
        model: The model name used for evaluation
        output_dir: Directory to write the CSV file

    Returns:
        Path to the created CSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = get_timestamp_prefix()
    filepath = os.path.join(
        output_dir, f"{timestamp}_mermaid_results_{model.replace(':', '_')}.csv"
    )

    all_evaluator_names = set()
    all_metric_names = set()

    for case in report.cases:
        all_evaluator_names.update(case.scores.keys())
        if hasattr(case.output, "metrics") and case.output.metrics:
            all_metric_names.update(case.output.metrics.keys())

    headers = [
        "Model",
        "Case",
        "Duration",
        "Fixed_Diagram_Length",
        "Failure_Reason",
        "Tools_Used",
    ]

    for evaluator in sorted(all_evaluator_names):
        headers.append(f"Score_{evaluator}")

    for metric in sorted(all_metric_names):
        headers.append(f"Metric_{metric}")

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for case in report.cases:
            row = [
                model,
                case.name,
                case.task_duration,
                len(case.output.fixed_diagram)
                if case.output and case.output.fixed_diagram
                else 0,
                case.output.failure_reason if case.output else "",
                "|".join(case.output.tools_used)
                if case.output and case.output.tools_used
                else "",
            ]

            for evaluator in sorted(all_evaluator_names):
                if evaluator in case.scores:
                    row.append(case.scores[evaluator].value)
                else:
                    row.append("")

            for metric in sorted(all_metric_names):
                if (
                    case.output
                    and hasattr(case.output, "metrics")
                    and case.output.metrics
                    and metric in case.output.metrics
                ):
                    metric_value = case.output.metrics[metric]
                    if isinstance(metric_value, dict):
                        row.append(str(metric_value))
                    else:
                        row.append(metric_value)
                else:
                    row.append("")

            writer.writerow(row)

    print(f"Mermaid evaluation results written to {filepath}")
    return filepath


async def run_evaluations(
    model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_MODEL,
    export_csv: bool = True,
):
    """Run the evaluations on the mermaid diagram fixing task.

    Args:
        model: The model to use for the agent
        judge_model: The model to use for LLM judging
        export_csv: Whether to export results to CSV

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

    if export_csv:
        csv_path = write_mermaid_results_to_csv(report, model)
        print(f"Results exported to: {csv_path}")

    return report


if __name__ == "__main__":
    # You can use different models for the agent and the judge
    # agent_model = os.getenv("AGENT_MODEL", DEFAULT_MODEL)
    agent_model = "gemini-2.5-pro-preview-06-05"
    # agent_model = "openai:o4-mini"
    # agent_model = "gemini-2.5-flash-preview-04-17"
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)

    async def run_all():
        await run_evaluations(
            model=agent_model, judge_model=judge_model, export_csv=True
        )

    asyncio.run(run_all())
