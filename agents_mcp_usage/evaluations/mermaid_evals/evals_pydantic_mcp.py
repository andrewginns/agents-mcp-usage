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
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from agents_mcp_usage.evaluations.mermaid_evals.mermaid_diagrams import (
    invalid_mermaid_diagram_easy,
    invalid_mermaid_diagram_hard,
    invalid_mermaid_diagram_medium,
    valid_mermaid_diagram,
)
from agents_mcp_usage.factory.model_factory import (
    create_agent as create_agent_with_model,
)
from agents_mcp_usage.utils import get_mcp_server_path

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

# Request limit configuration for calls to the model
REQUEST_LIMIT = 500
REQUEST_USAGE_COLUMN = "Requests_Used"

# Defaults tuned for evaluation runs (avoid hammering endpoints on 429/5xx)
MAX_RETRY_ATTEMPTS = 20
BASE_RETRY_DELAY = 2.0  # seconds
MAX_RETRY_DELAY = 60.0  # seconds

# Dynamically import validate_mermaid_diagram from the mermaid_validator.py file
mermaid_validator_path = str(get_mcp_server_path("mermaid_validator.py"))
spec = importlib.util.spec_from_file_location("mermaid_validator", mermaid_validator_path)
mermaid_validator = importlib.util.module_from_spec(spec)
sys.modules["mermaid_validator"] = mermaid_validator
spec.loader.exec_module(mermaid_validator)
validate_mermaid_diagram = mermaid_validator.validate_mermaid_diagram

# ============================================================================
# Retry Utilities
# ============================================================================


class MermaidValidatorUnavailableError(RuntimeError):
    """Raised when Mermaid validation cannot be performed.

    This is treated as *fatal* since it invalidates the evaluation (fail-fast).
    """


def _iter_base_exceptions(exception: BaseException) -> Iterable[BaseException]:
    """Yield underlying exceptions, flattening `ExceptionGroup`s."""
    if isinstance(exception, BaseExceptionGroup):
        for exc in exception.exceptions:
            yield from _iter_base_exceptions(exc)
    else:
        yield exception


MERMAID_VALIDATOR_ERROR_HINTS = (
    "mermaid_validator",
    "mermaid-validator",
    "validate_mermaid_diagram",
)


def is_mermaid_validator_unavailable_error(exception: BaseException) -> bool:
    """Best-effort detection for Mermaid validator MCP/server failures."""
    if isinstance(exception, MermaidValidatorUnavailableError):
        return True
    text = f"{type(exception).__name__}: {exception}".lower()
    return any(hint in text for hint in MERMAID_VALIDATOR_ERROR_HINTS)


def compute_exponential_backoff_delay(
    attempt: int,
    base_delay: float = BASE_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    jitter: bool = True,
) -> float:
    """Compute exponential backoff delay for an attempt (0-indexed)."""
    delay = min(base_delay * (2**attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # 50% jitter
    return delay


def is_retryable_error(exception: Exception) -> bool:
    """Checks if an exception is retryable.

    This function checks if the given exception is a retryable HTTP error or a
    general connection error.

    For `ExceptionGroup`s we inspect contained exceptions and default to
    retryable *unless* the group clearly contains non-retryable signals (usage
    limits, per-case timeouts) or a fatal Mermaid validator failure.
    """
    if isinstance(exception, MermaidValidatorUnavailableError):
        return False

    # Per-case timeouts should not be retried (run-level timeouts are handled separately)
    if isinstance(exception, asyncio.TimeoutError):
        return False

    if isinstance(exception, BaseExceptionGroup):
        inner_exceptions = list(_iter_base_exceptions(exception))

        # Fail-fast (fatal)
        if any(is_mermaid_validator_unavailable_error(exc) for exc in inner_exceptions):
            return False

        # Non-retryable (policy)
        if any(isinstance(exc, UsageLimitExceeded) for exc in inner_exceptions):
            return False
        if any(isinstance(exc, asyncio.TimeoutError) for exc in inner_exceptions):
            return False

        # Explicit retryables
        if any(
            isinstance(exc, ModelHTTPError)
            and exc.status_code in RETRYABLE_HTTP_STATUS_CODES
            for exc in inner_exceptions
        ):
            return True
        if any(isinstance(exc, (ConnectionError, OSError)) for exc in inner_exceptions):
            return True

        # Default for groups: treat as retryable (matches `error_ExceptionGroup` policy)
        return True

    if isinstance(exception, ModelHTTPError):
        return exception.status_code in RETRYABLE_HTTP_STATUS_CODES

    # Also retry on general connection errors that might be transient
    if isinstance(exception, (ConnectionError, OSError)):
        return True

    return False


async def exponential_backoff_retry(
    func_call: Callable[[], Awaitable[Any]],
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = BASE_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    jitter: bool = True,
    is_retryable: Callable[[Exception], bool] = is_retryable_error,
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

            if not is_retryable(e):
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

            delay = compute_exponential_backoff_delay(
                attempt, base_delay=base_delay, max_delay=max_delay, jitter=jitter
            )

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
            str(get_mcp_server_path("example_server.py")),
            "stdio",
        ],
    )
    mermaid_server = MCPServerStdio(
        command="uv",
        args=[
            "run",
            str(get_mcp_server_path("mermaid_validator.py")),
        ],
    )
    return [local_server, mermaid_server]


def create_agent(
    model: str = DEFAULT_MODEL, model_settings: Dict[str, Any] = None
) -> Agent:
    """Creates an agent with MCP servers for the specified model.

    This function initializes and returns an agent with the necessary MCP
    servers and model settings using the new model factory.

    Args:
        model: The model to use for the agent.
        model_settings: Optional model-specific settings.

    Returns:
        A configured Agent instance.
    """
    if model_settings is None:
        model_settings = {}

    # Use the new model factory for all models
    return create_agent_with_model(
        model=model,
        mcp_servers=get_mcp_servers(),
        model_settings=model_settings,
    )


# ============================================================================
# Schema Definitions
# ============================================================================


class MermaidInput(BaseModel):
    """Input schema for mermaid diagram fixing."""

    invalid_diagram: str
    case_name: Optional[str] = None


class MermaidOutput(BaseModel):
    """Output schema for mermaid diagram fixing with comprehensive metrics."""

    fixed_diagram: str
    failure_reason: str = ""  # Track why a case failed
    metrics: Dict[str, Any] = {}  # Capture LLM usage metrics
    tools_used: List[str] = []  # Track which MCP tools were called


# ============================================================================
# Debug Trace Capture
# ============================================================================

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value).strip("_")
    return slug or "unknown"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _usage_limits_to_dict(usage_limits: UsageLimits) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key in (
        "request_limit",
        "request_tokens_limit",
        "response_tokens_limit",
        "total_tokens_limit",
    ):
        if hasattr(usage_limits, key):
            value = getattr(usage_limits, key)
            if value is not None:
                data[key] = value
    return data


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    """Best-effort conversion of PydanticAI usage objects to primitives for JSON."""
    details = getattr(usage, "details", None) or {}
    return {
        "requests": getattr(usage, "requests", None),
        "request_tokens": getattr(usage, "request_tokens", None),
        "response_tokens": getattr(usage, "response_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "details": details,
    }


def _exception_to_dict(exception: BaseException) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback.format_exception(exception),
    }

    if isinstance(exception, ModelHTTPError):
        data.update(
            {
                "status_code": exception.status_code,
                "model_name": exception.model_name,
                "body": exception.body,
            }
        )

    if isinstance(exception, BaseExceptionGroup):
        data["exceptions"] = [_exception_to_dict(e) for e in exception.exceptions]

    return data


def _summarize_messages(messages: Any) -> Dict[str, Any]:
    """Create small derived fields to quickly spot tool loops / excessive turn usage.

    Accepts the parsed output of `all_messages_json()`.
    """
    if not isinstance(messages, list):
        return {}

    summary: Dict[str, Any] = {
        "message_count": len(messages),
        "request_count": 0,
        "response_count": 0,
        "tool_call_count": 0,
        "tool_return_count": 0,
        "tool_names_unique": [],
        "tool_calls_by_name": {},
        "tool_sequence": [],
    }

    tools_seen_ordered: List[str] = []
    tool_calls_by_name: Dict[str, int] = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        kind = msg.get("kind")
        if kind == "request":
            summary["request_count"] += 1
        elif kind == "response":
            summary["response_count"] += 1

        parts = msg.get("parts") or []
        if not isinstance(parts, list):
            continue

        for part in parts:
            if not isinstance(part, dict):
                continue
            part_kind = part.get("part_kind")
            tool_name = part.get("tool_name")
            tool_call_id = part.get("tool_call_id")

            if tool_name and tool_name not in tools_seen_ordered:
                tools_seen_ordered.append(tool_name)

            if part_kind in {"tool-call", "builtin-tool-call"}:
                summary["tool_call_count"] += 1
                if tool_name:
                    tool_calls_by_name[tool_name] = tool_calls_by_name.get(tool_name, 0) + 1
                summary["tool_sequence"].append(
                    {"event": "call", "tool_name": tool_name, "tool_call_id": tool_call_id}
                )

            if part_kind in {"tool-return", "builtin-tool-return"}:
                summary["tool_return_count"] += 1
                summary["tool_sequence"].append(
                    {"event": "return", "tool_name": tool_name, "tool_call_id": tool_call_id}
                )

    summary["tool_names_unique"] = tools_seen_ordered
    summary["tool_calls_by_name"] = tool_calls_by_name
    return summary


@dataclass(slots=True)
class MermaidEvalTraceContext:
    """Per-case debug trace capture config passed down from evaluation runners."""

    enabled: bool
    trace_dir: str
    model: str
    planned_run_index: Optional[int] = None  # 0-based planned run index
    run_attempt: Optional[int] = None  # 1-based attempt counter at runner-level
    evaluation_name: Optional[str] = None


def _build_trace_path(trace_ctx: MermaidEvalTraceContext, inputs: MermaidInput) -> str:
    model_slug = _slugify(trace_ctx.model)
    case_slug = _slugify(inputs.case_name) if inputs.case_name else f"hash_{_short_hash(inputs.invalid_diagram)}"
    run_part = (
        f"run{(trace_ctx.planned_run_index + 1):03d}"
        if trace_ctx.planned_run_index is not None
        else "run000"
    )
    attempt_part = (
        f"attempt{trace_ctx.run_attempt:03d}"
        if trace_ctx.run_attempt is not None
        else "attempt000"
    )

    model_dir = os.path.join(trace_ctx.trace_dir, model_slug)
    filename = f"model={model_slug}__{run_part}__{attempt_part}__case={case_slug}.json"
    return os.path.join(model_dir, filename)


def _write_trace(trace_path: str, trace_record: Dict[str, Any]) -> None:
    """Best-effort writer; never raise (debug traces should not break evals)."""
    try:
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        tmp_path = f"{trace_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(trace_record, f, indent=2, ensure_ascii=False, sort_keys=False)
        os.replace(tmp_path, trace_path)
    except Exception as e:  # pragma: no cover
        logfire.warning(
            "Failed to write debug trace file",
            trace_path=trace_path,
            error_type=type(e).__name__,
            error=str(e),
        )


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
        except MermaidValidatorUnavailableError:
            raise
        except Exception as e:
            logfire.error(
                "Failed to validate mermaid diagram after retries",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise MermaidValidatorUnavailableError(
                "Mermaid validator MCP server unavailable"
            ) from e

        if (
            not result.is_valid
            and result.error_message
            and result.error_message.startswith("Error validating mermaid diagram:")
        ):
            # If the validator itself is failing, results are invalid → fail-fast.
            raise MermaidValidatorUnavailableError(result.error_message)

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
    inputs: MermaidInput,
    model: str = DEFAULT_MODEL,
    *,
    trace_ctx: MermaidEvalTraceContext | None = None,
) -> MermaidOutput:
    """Fixes an invalid mermaid diagram using an agent with multiple MCP servers.

    In normal mode, we return aggregate usage metrics and tool names.
    In debug trace mode, we additionally write a per-case JSON file containing:
    input, prompt, full message history (incl. tool call args + tool return payloads),
    and usage.

    Args:
        inputs: The input containing the invalid diagram.
        model: The model to use for the agent.
        trace_ctx: Optional debug trace config (used by multi-model runner).

    Returns:
        A MermaidOutput object with the fixed diagram and captured metrics.
    """
    query = (
        "Add the current time and fix the mermaid diagram syntax using the validator: "
        f"{inputs.invalid_diagram}. Return only the fixed mermaid diagram between backticks."
    )

    trace_enabled = bool(trace_ctx and trace_ctx.enabled)

    # Create a fresh agent for each invocation to avoid concurrent usage issues
    current_agent = create_agent(model)

    # NOTE: request_limit is the most common limiter we hit in tool-loop failure modes.
    usage_limits = UsageLimits(request_limit=REQUEST_LIMIT)

    captured_messages_json: str | None = None
    captured_usage: Any | None = None
    captured_agent_run_id: str | None = None

    def _maybe_write_trace(
        *,
        status: str,
        failure_reason: str,
        output_text: str | None,
        extracted_diagram: str | None,
        exception: BaseException | None = None,
    ) -> None:
        nonlocal captured_messages_json, captured_usage, captured_agent_run_id

        if not trace_enabled or trace_ctx is None:
            return

        trace_path = _build_trace_path(trace_ctx, inputs)

        messages_parsed: Any = None
        if captured_messages_json:
            try:
                messages_parsed = json.loads(captured_messages_json)
            except Exception:
                messages_parsed = captured_messages_json

        trace_record: Dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "model": model,
            "runner": {
                "evaluation_name": trace_ctx.evaluation_name,
                "planned_run_index": trace_ctx.planned_run_index,
                "planned_run_number": (
                    trace_ctx.planned_run_index + 1
                    if trace_ctx.planned_run_index is not None
                    else None
                ),
                "run_attempt": trace_ctx.run_attempt,
            },
            "case": {
                "case_name": inputs.case_name,
                "invalid_diagram": inputs.invalid_diagram,
                "invalid_diagram_len": len(inputs.invalid_diagram),
                "invalid_diagram_sha256": hashlib.sha256(
                    inputs.invalid_diagram.encode("utf-8")
                ).hexdigest(),
            },
            "prompt": {"query": query},
            "usage_limits": _usage_limits_to_dict(usage_limits),
            "agent": {
                "run_id": captured_agent_run_id,
                "all_messages": messages_parsed,
                "messages_summary": _summarize_messages(messages_parsed),
            },
            "result": {
                "output_text": output_text,
                "extracted_diagram": extracted_diagram,
                "failure_reason": failure_reason,
                "usage": _usage_to_dict(captured_usage) if captured_usage is not None else {},
            },
        }

        if exception is not None:
            trace_record["error"] = _exception_to_dict(exception)

        _write_trace(trace_path, trace_record)

    async def _run_agent():
        nonlocal captured_messages_json, captured_usage, captured_agent_run_id

        async with current_agent.run_mcp_servers():
            if trace_enabled:
                agent_run = None
                try:
                    async with current_agent.iter(
                        query, usage_limits=usage_limits, infer_name=False
                    ) as _agent_run:
                        agent_run = _agent_run
                        async for _node in agent_run:
                            # We don't persist nodes (they're not stable/JSON-serializable),
                            # all useful debug info is in `all_messages_json()`.
                            pass
                        # NOTE: this may raise if the run failed during finalisation;
                        # our `finally` block below will still capture partial messages/usage.
                        return agent_run.result
                finally:
                    if agent_run is not None:
                        captured_agent_run_id = agent_run.run_id
                        try:
                            captured_messages_json = agent_run.all_messages_json()
                        except Exception:
                            captured_messages_json = None
                        try:
                            captured_usage = agent_run.usage()
                        except Exception:
                            captured_usage = None

            # Non-trace mode: keep the existing behaviour.
            result = await current_agent.run(query, usage_limits=usage_limits)
            return result

    try:
        # Use retry logic for the agent run
        result = await exponential_backoff_retry(_run_agent)

        # Extract usage metrics
        usage = captured_usage or result.usage()
        metrics = _usage_to_dict(usage)

        # Extract tool usage information from agent messages
        tools_used: List[str] = []
        for message in result.all_messages():
            for part in message.parts:
                tool_name = getattr(part, "tool_name", None)
                if tool_name:
                    tools_used.append(tool_name)

        tools_used = list(dict.fromkeys(tools_used))  # unique, preserve order
        output = result.output

        # Extract the diagram from between backticks
        if "```" in output:
            start = output.find("```")
            end = output.rfind("```") + 3
            diagram = output[start:end]
        else:
            diagram = output

        _maybe_write_trace(
            status="success",
            failure_reason="",
            output_text=output,
            extracted_diagram=diagram,
        )

        return MermaidOutput(fixed_diagram=diagram, metrics=metrics, tools_used=tools_used)

    except MermaidValidatorUnavailableError as e:
        _maybe_write_trace(
            status="error",
            failure_reason="mermaid_validator_unavailable",
            output_text=None,
            extracted_diagram=None,
            exception=e,
        )
        raise

    except UsageLimitExceeded as e:
        logfire.warning(
            "Usage limit exceeded during mermaid diagram fix",
            error_message=str(e),
            model=model,
        )
        _maybe_write_trace(
            status="failure",
            failure_reason="usage_limit_exceeded",
            output_text=None,
            extracted_diagram=None,
            exception=e,
        )
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
        _maybe_write_trace(
            status="failure",
            failure_reason=f"http_error_{e.status_code}",
            output_text=None,
            extracted_diagram=None,
            exception=e,
        )
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
        _maybe_write_trace(
            status="failure",
            failure_reason="response_validation_failed",
            output_text=None,
            extracted_diagram=None,
            exception=e,
        )
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
        _maybe_write_trace(
            status="failure",
            failure_reason="agent_timeout",
            output_text=None,
            extracted_diagram=None,
            exception=e,
        )
        return MermaidOutput(
            fixed_diagram="",
            failure_reason="agent_timeout",
            metrics={},
            tools_used=[],
        )

    except Exception as e:
        # Mermaid validator MCP failures are fatal for evaluations (fail-fast)
        if is_mermaid_validator_unavailable_error(e):
            wrapped = MermaidValidatorUnavailableError(
                "Mermaid validator MCP server unavailable"
            )
            _maybe_write_trace(
                status="error",
                failure_reason="mermaid_validator_unavailable",
                output_text=None,
                extracted_diagram=None,
                exception=wrapped,
            )
            raise wrapped from e

        # Provide more specific error categorization
        error_type = type(e).__name__
        failure_reason = ""

        if isinstance(e, BaseExceptionGroup):
            inner_exceptions = list(_iter_base_exceptions(e))

            if any(
                is_mermaid_validator_unavailable_error(exc) for exc in inner_exceptions
            ):
                raise MermaidValidatorUnavailableError(
                    "Mermaid validator MCP server unavailable"
                ) from e

            if any(isinstance(exc, UsageLimitExceeded) for exc in inner_exceptions):
                failure_reason = "usage_limit_exceeded"
            elif any(isinstance(exc, asyncio.TimeoutError) for exc in inner_exceptions):
                # Policy: per-case timeouts should not be retried
                failure_reason = "agent_timeout"
            else:
                http_exc = next(
                    (
                        exc
                        for exc in inner_exceptions
                        if isinstance(exc, ModelHTTPError)
                    ),
                    None,
                )
                if http_exc:
                    failure_reason = f"http_error_{http_exc.status_code}"
                elif any(
                    isinstance(exc, (ConnectionError, OSError))
                    for exc in inner_exceptions
                ):
                    failure_reason = "connection_error"
                else:
                    # Default: ambiguous ExceptionGroup (treated as retryable at run-level)
                    failure_reason = "error_ExceptionGroup"
        else:
            error_lower = str(e).lower()
            if "timeout" in error_lower or "timed out" in error_lower:
                failure_reason = "timeout_error"
            elif "connection" in error_lower or "network" in error_lower:
                failure_reason = "connection_error"
            elif "rate limit" in error_lower or "quota" in error_lower:
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

        _maybe_write_trace(
            status="failure",
            failure_reason=failure_reason,
            output_text=None,
            extracted_diagram=None,
            exception=e,
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
                inputs=MermaidInput(
                    invalid_diagram=invalid_mermaid_diagram_easy,
                    case_name="fix_invalid_diagram_easy",
                ),
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
                inputs=MermaidInput(
                    invalid_diagram=invalid_mermaid_diagram_medium,
                    case_name="fix_invalid_diagram_medium",
                ),
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
                inputs=MermaidInput(
                    invalid_diagram=invalid_mermaid_diagram_hard,
                    case_name="fix_invalid_diagram_hard",
                ),
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
            # LLMJudge(
            #     rubric="The response only contains a mermaid diagram inside the fixed_diagram field, no other text. Ignore the metrics, failure_reason, and tools_used fields.",
            #     include_input=False,
            #     model=judge_model,
            # ),
            # LLMJudge(
            #     rubric="The fixed_diagram field should maintain the same overall structure and intent as the expected output diagram while fixing any syntax errors. Check if nodes, connections, and labels are preserved. The current time placeholder should be replaced with a valid datetime. Ignore the metrics, failure_reason, and tools_used fields.",
            #     include_input=False,
            #     model=judge_model,
            # ),
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
    report: EvaluationReport, model: str, output_dir: str = "./mermaid_eval_results"
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

    # Track how many model invocations were used (e.g., "87/500")
    headers.append(REQUEST_USAGE_COLUMN)

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

            # Add request-usage summary as the last column
            requests_used = None
            if (
                case.output
                and hasattr(case.output, "metrics")
                and isinstance(case.output.metrics, dict)
            ):
                requests_used = case.output.metrics.get("requests")

            row.append(requests_used if requests_used is not None else "na")

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
    output_dir: str = "./mermaid_eval_results",
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
    agent_model = "gemini-2.5-flash"
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)

    async def run_all():
        await run_evaluations(
            model=agent_model, judge_model=judge_model, export_csv=True
        )

    asyncio.run(run_all())
