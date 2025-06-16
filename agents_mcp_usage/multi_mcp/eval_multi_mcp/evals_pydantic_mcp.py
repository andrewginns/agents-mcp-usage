"""
Single-Model Evaluation Module for Mermaid Diagram Fixing

This module provides the core functionality for evaluating LLM models
on mermaid diagram fixing tasks using multiple MCP servers. It includes:
- Schema definitions for inputs and outputs
- Custom evaluators for multi-MCP tool usage validation
- Agent creation and mermaid diagram fixing functions
- Dataset creation and evaluation utilities
- CSV export functionality
- Robust retry logic for handling transient API failures

This module is designed to be imported by multi-model evaluation scripts.
"""

import asyncio
import csv
import os
import random
from datetime import datetime
from typing import Any, Dict, List

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded, ModelHTTPError
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.reporting import EvaluationReport

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

# Default model configurations
DEFAULT_MODEL = "gemini-2.5-pro-preview-06-05"

# Retry configuration
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRY_ATTEMPTS = 3
BASE_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 30.0  # seconds

# ============================================================================
# Retry Utilities
# ============================================================================


def is_retryable_error(exception: Exception) -> bool:
    """Checks if an exception is retryable.

    This function checks if the given exception is a retryable HTTP error or a
    general connection error.

    Args:
        exception: The exception to check.

    Returns:
        True if the exception is retryable, False otherwise.
    """
    if isinstance(exception, ModelHTTPError):
        return exception.status_code in RETRYABLE_HTTP_STATUS_CODES

    # Also retry on general connection errors that might be transient
    if isinstance(exception, (ConnectionError, OSError)):
        return True

    return False


async def exponential_backoff_retry(
    func_call: callable,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = BASE_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    jitter: bool = True,
) -> Any:
    """Executes a function with exponential backoff retry logic.

    This function attempts to execute the given asynchronous function call,
    retrying with an exponential backoff delay if a retryable error occurs.

    Args:
        func_call: The async function to retry.
        max_attempts: The maximum number of retry attempts.
        base_delay: The base delay between retries in seconds.
        max_delay: The maximum delay between retries in seconds.
        jitter: Whether to add random jitter to the delays.

    Returns:
        The result of the function call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func_call()
        except Exception as e:
            last_exception = e

            if not is_retryable_error(e):
                logfire.warning(
                    "Non-retryable error encountered",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    attempt=attempt + 1,
                )
                raise

            if attempt == max_attempts - 1:
                logfire.error(
                    "Max retry attempts exhausted",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    attempts=max_attempts,
                )
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2**attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)  # Add 50% jitter

            logfire.warning(
                "Retryable error encountered, retrying",
                error_type=type(e).__name__,
                error_message=str(e),
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_seconds=delay,
            )

            await asyncio.sleep(delay)

    # Re-raise the last exception if all retries failed
    if last_exception:
        raise last_exception

    # This should never be reached, but just in case
    raise RuntimeError("Unexpected error in retry logic")


# ============================================================================
# MCP Server Configuration
# ============================================================================


def get_mcp_servers() -> List[MCPServerStdio]:
    """Gets the configured MCP servers for the evaluation.

    This function returns a list of MCP servers required for the evaluation,
    including the local example server and the mermaid validator server.

    Returns:
        A list of configured MCP servers.
    """
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
    return [local_server, mermaid_server]


def create_agent(
    model: str = DEFAULT_MODEL, model_settings: Dict[str, Any] = None
) -> Agent:
    """Creates an agent with MCP servers for the specified model.

    This function initializes and returns an agent with the necessary MCP
    servers and model settings.

    Args:
        model: The model to use for the agent.
        model_settings: Optional model-specific settings.

    Returns:
        A configured Agent instance.
    """
    if model_settings is None:
        model_settings = {}

    # Handle Bedrock models specifically
    if model.startswith("bedrock:"):
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockProvider

        # Extract the model name (remove "bedrock:" prefix)
        model_name = model.replace("bedrock:", "")

        # Create BedrockConverseModel with proper region and profile configuration
        bedrock_model = BedrockConverseModel(
            model_name,
            provider=BedrockProvider(
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                profile_name=os.getenv("AWS_PROFILE", "my-aws-profile"),
            ),
        )

        return Agent(
            bedrock_model,
            mcp_servers=get_mcp_servers(),
            model_settings=model_settings,
        )

    # For non-Bedrock models, use the original approach
    return Agent(
        model,
        mcp_servers=get_mcp_servers(),
        model_settings=model_settings,
    )


# ============================================================================
# Schema Definitions
# ============================================================================


class MermaidInput(BaseModel):
    """Input schema for mermaid diagram fixing."""

    invalid_diagram: str


class MermaidOutput(BaseModel):
    """Output schema for mermaid diagram fixing with comprehensive metrics."""

    fixed_diagram: str
    failure_reason: str = ""  # Track why a case failed
    metrics: Dict[str, Any] = {}  # Capture LLM usage metrics
    tools_used: List[str] = []  # Track which MCP tools were called


# ============================================================================
# Custom Evaluators
# ============================================================================


class UsedBothMCPTools(Evaluator[MermaidInput, MermaidOutput]):
    """Evaluator to check if both MCP tools were used."""

    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        """Evaluates if both MCP tools were used in the given context.

        This method checks the tools used in the output and returns a score
        based on whether tools from both MCP servers were utilized.

        Args:
            ctx: The evaluator context containing the input and output.

        Returns:
            A score of 1.0 if both tools were used, 0.5 if one was used,
            and 0.0 otherwise.
        """
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


class UsageLimitNotExceeded(Evaluator[MermaidInput, MermaidOutput]):
    """Evaluator to detect usage limit failures."""

    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        """Checks if the case failed due to usage limits being exceeded.

        This method examines the output for a usage limit failure reason and
        returns a score accordingly.

        Args:
            ctx: The evaluator context.

        Returns:
            0.0 if a usage limit failure occurred, 1.0 otherwise.
        """
        if ctx.output and ctx.output.failure_reason == "usage_limit_exceeded":
            logfire.warning(
                "Case failed due to usage limit exceeded",
                case_name=getattr(ctx, "case_name", "unknown"),
            )
            return 0.0
        # Return 1.0 if no usage limit failure occurred
        return 1.0


class MermaidDiagramValid(Evaluator[MermaidInput, MermaidOutput]):
    """Evaluator to check if the mermaid diagram is valid."""

    async def evaluate(
        self, ctx: EvaluatorContext[MermaidInput, MermaidOutput]
    ) -> float:
        """Evaluates if the generated mermaid diagram is valid.

        This method validates the mermaid diagram in the output, handling
        retries and logging the results.

        Args:
            ctx: The evaluator context.

        Returns:
            1.0 if the diagram is valid, 0.0 otherwise.
        """
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

        # Use the MCP server's validation function with retry logic
        try:
            result = await exponential_backoff_retry(
                lambda: validate_mermaid_diagram(input_str)
            )
        except Exception as e:
            logfire.error(
                "Failed to validate mermaid diagram after retries",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return 0.0

        if result.is_valid:
            logfire.info("Mermaid diagram validation succeeded")
        else:
            logfire.warning(
                "Mermaid diagram validation failed", error_message=result.error_message
            )

        return 1.0 if result.is_valid else 0.0


# ============================================================================
# Core Evaluation Functions
# ============================================================================


async def fix_mermaid_diagram(
    inputs: MermaidInput, model: str = DEFAULT_MODEL
) -> MermaidOutput:
    """Fixes an invalid mermaid diagram using an agent with multiple MCP servers.

    This function runs an agent to fix a given mermaid diagram, handling
    various exceptions and capturing metrics.

    Args:
        inputs: The input containing the invalid diagram.
        model: The model to use for the agent.

    Returns:
        A MermaidOutput object with the fixed diagram and captured metrics.
    """
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {inputs.invalid_diagram}. Return only the fixed mermaid diagram between backticks."

    # Create a fresh agent for each invocation to avoid concurrent usage issues
    current_agent = create_agent(model)
    usage_limits = UsageLimits(request_limit=5)

    async def _run_agent():
        async with current_agent.run_mcp_servers():
            return await current_agent.run(query, usage_limits=usage_limits)

    try:
        # Use retry logic for the agent run
        result = await exponential_backoff_retry(_run_agent)

        # Extract usage metrics
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

        tools_used = list(
            dict.fromkeys(tools_used)
        )  # Remove duplicates while preserving order
        output = result.output

        # Extract the diagram from between backticks
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

    except ModelHTTPError as e:
        logfire.error(
            "HTTP error during mermaid diagram fix after retries",
            error_message=str(e),
            status_code=e.status_code,
            model_name=e.model_name,
            model=model,
        )
        # Return empty diagram with failure reason to indicate HTTP error
        return MermaidOutput(
            fixed_diagram="",
            failure_reason=f"http_error_{e.status_code}",
            metrics={},
            tools_used=[],
        )

    except ValidationError as e:
        logfire.error(
            "Response validation error during mermaid diagram fix",
            error_message=str(e),
            model=model,
        )
        # Return empty diagram with failure reason to indicate validation failure
        return MermaidOutput(
            fixed_diagram="",
            failure_reason="response_validation_failed",
            metrics={},
            tools_used=[],
        )

    except asyncio.TimeoutError as e:
        logfire.error(
            "Timeout error during mermaid diagram fix",
            error_message=str(e),
            model=model,
        )
        # Return empty diagram with failure reason to indicate timeout
        return MermaidOutput(
            fixed_diagram="",
            failure_reason="agent_timeout",
            metrics={},
            tools_used=[],
        )

    except Exception as e:
        # Provide more specific error categorization
        error_type = type(e).__name__
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            failure_reason = "timeout_error"
        elif "connection" in str(e).lower() or "network" in str(e).lower():
            failure_reason = "connection_error"
        elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
            failure_reason = "rate_limit_error"
        else:
            failure_reason = f"error_{error_type}"

        logfire.error(
            "Unexpected error during mermaid diagram fix after retries",
            error_message=str(e),
            error_type=error_type,
            categorized_failure_reason=failure_reason,
            model=model,
        )
        # Return empty diagram with failure reason to indicate general failure
        return MermaidOutput(
            fixed_diagram="",
            failure_reason=failure_reason,
            metrics={},
            tools_used=[],
        )


def create_evaluation_dataset(
    judge_model: str = DEFAULT_MODEL,
) -> Dataset[MermaidInput, MermaidOutput, Any]:
    """Creates the dataset for evaluating mermaid diagram fixing.

    This function constructs a dataset with test cases of varying difficulty
    and a set of evaluators for judging the results.

    Args:
        judge_model: The model to use for LLM judging.

    Returns:
        The evaluation dataset.
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
                rubric="The fixed_diagram field should maintain the same overall structure and intent as the expected output diagram while fixing any syntax errors. Check if nodes, connections, and labels are preserved. The current time placeholder should be replaced with a valid datetime. Ignore the metrics, failure_reason, and tools_used fields.",
                include_input=False,
                model=judge_model,
            ),
        ],
    )


# ============================================================================
# Utility Functions
# ============================================================================


def get_timestamp_prefix() -> str:
    """Gets a timestamp prefix in the format yyyy-mm-dd_H-M-s.

    Returns:
        A string representing the current timestamp.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def write_mermaid_results_to_csv(
    report: EvaluationReport, model: str, output_dir: str = "./mermaid_results"
) -> str:
    """Writes mermaid evaluation results with metrics to a CSV file.

    This function takes an evaluation report and writes the results to a CSV
    file, including scores and metrics.

    Args:
        report: The evaluation report from pydantic_evals.
        model: The model name used for evaluation.
        output_dir: The directory to write the CSV file to.

    Returns:
        The path to the created CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = get_timestamp_prefix()
    filepath = os.path.join(
        output_dir, f"{timestamp}_mermaid_results_{model.replace(':', '_')}.csv"
    )

    # Collect all unique evaluator and metric names
    all_evaluator_names = set()
    all_metric_names = set()

    for case in report.cases:
        all_evaluator_names.update(case.scores.keys())
        if hasattr(case.output, "metrics") and case.output.metrics:
            all_metric_names.update(case.output.metrics.keys())

    # Build CSV headers
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

    # Write the CSV file
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

            # Add evaluator scores
            for evaluator in sorted(all_evaluator_names):
                if evaluator in case.scores:
                    row.append(case.scores[evaluator].value)
                else:
                    row.append("")

            # Add metrics
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


# ============================================================================
# Single Model Evaluation Function
# ============================================================================


async def run_evaluations(
    model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_MODEL,
    export_csv: bool = True,
    output_dir: str = "./mermaid_results",
) -> EvaluationReport:
    """Runs the evaluations on the mermaid diagram fixing task.

    This function sets up the evaluation dataset, runs the evaluation for a
    given model, and exports the results to a CSV file.

    Args:
        model: The model to use for the agent.
        judge_model: The model to use for LLM judging.
        export_csv: Whether to export the results to a CSV file.
        output_dir: The directory to save the results to.

    Returns:
        The evaluation report.
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
        csv_path = write_mermaid_results_to_csv(report, model, output_dir)
        print(f"Results exported to: {csv_path}")

    return report


# ============================================================================
# Main Execution (for standalone use)
# ============================================================================

if __name__ == "__main__":
    # You can use different models for the agent and the judge
    # agent_model = os.getenv("AGENT_MODEL", DEFAULT_MODEL)
    # agent_model = "gemini-2.5-pro-preview-06-05"
    # agent_model = "openai:o4-mini"
    agent_model = "gemini-2.5-flash-preview-04-17"
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)

    async def run_all():
        await run_evaluations(
            model=agent_model, judge_model=judge_model, export_csv=True
        )

    asyncio.run(run_all())
