#!/usr/bin/env python3
"""
Multi-Model Evaluation Script for Mermaid Diagram Fixing

This script extends single-model evaluation to handle multiple LLM models with:
- Robust failure handling and recovery
- Individual model results written to disk before combining
- Configurable number of runs per model
- Combined metrics output to single CSV file
- Handling of variable response metrics from different models
- Parallel and sequential execution modes
"""

import argparse
import asyncio
import csv
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import logfire
from dotenv import load_dotenv
from pydantic_evals.reporting import EvaluationReport
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table

# Import shared functionality from the improved evals module
from agents_mcp_usage.evaluations.mermaid_evals.evals_pydantic_mcp import (
    BASE_RETRY_DELAY,
    MAX_RETRY_ATTEMPTS,
    MAX_RETRY_DELAY,
    MermaidEvalTraceContext,
    MermaidInput,
    MermaidOutput,
    MermaidValidatorUnavailableError,
    RETRYABLE_HTTP_STATUS_CODES,
    REQUEST_USAGE_COLUMN,
    compute_exponential_backoff_delay,
    create_evaluation_dataset,
    fix_mermaid_diagram,
    get_timestamp_prefix,
    is_mermaid_validator_unavailable_error,
    is_retryable_error,
)

load_dotenv()

DEFAULT_MODELS = [
    # "openai:gpt-5 (minimal)",
    # "openai:gpt-5.1 (none)",
    "openai:gpt-5.2 (none)",
    # "openai:gpt-5.2-codex (none)",
]

logfire.configure(
    send_to_logfire="if-token-present", service_name="multi-model-mermaid-evals"
)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

_FILENAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_filename_component(value: str) -> str:
    """Return a filesystem-safe slug for `value`.

    This is used when writing per-model CSV files so model IDs like
    `openrouter:anthropic/claude-3.7-sonnet` don't create invalid paths.
    """
    sanitized = _FILENAME_SAFE_RE.sub("_", value).strip("._-")
    return sanitized or "unknown"


@dataclass(slots=True)
class EvaluationRunResult:
    """Result of a single *planned* evaluation run (including retries)."""

    run_index: int  # 0-based planned run index
    attempts: int
    last_retry_reason: str
    report: EvaluationReport


class ModelEvaluationResults:
    """Container for storing and managing evaluation results for a single model."""

    def __init__(self, model: str):
        self.model = model
        self.successful_runs: List[EvaluationRunResult] = []
        self.failed_runs: List[Dict[str, Any]] = []

    def add_successful_run(self, run_result: EvaluationRunResult) -> None:
        """Adds a successful evaluation run result.

        Args:
            run_result: The successful run result to add.
        """
        self.successful_runs.append(run_result)

    def add_failed_run(self, run_index: int, error: str, attempts: int) -> None:
        """Adds information about a failed run.

        Args:
            run_index: The index of the failed run.
            error: The error category.
            attempts: How many attempts were made for this run.
        """
        self.failed_runs.append(
            {
                "run_index": run_index,
                "attempts": attempts,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_success_rate(self) -> float:
        """Calculates the success rate for this model.

        Returns:
            The success rate as a float.
        """
        total_runs = len(self.successful_runs) + len(self.failed_runs)
        if total_runs == 0:
            return 0.0
        return len(self.successful_runs) / total_runs

    def write_individual_results(self, output_dir: str) -> Optional[str]:
        """Writes individual model results to a CSV file.

        Args:
            output_dir: The directory to write the CSV file to.

        Returns:
            The path to the created CSV file, or None if no results were written.
        """
        if not self.successful_runs:
            return None

        os.makedirs(output_dir, exist_ok=True)
        safe_model = sanitize_filename_component(self.model)
        filepath = os.path.join(output_dir, f"individual_{safe_model}.csv")

        all_evaluator_names = set()
        all_metric_names = set()

        for run in self.successful_runs:
            report = run.report
            for case in report.cases:
                all_evaluator_names.update(case.scores.keys())
                if hasattr(case.output, "metrics") and case.output.metrics:
                    all_metric_names.update(case.output.metrics.keys())

        headers = [
            "Model",
            "Run",
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

        headers.append(REQUEST_USAGE_COLUMN)

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for run in self.successful_runs:
                report = run.report
                for case in report.cases:
                    row = [
                        self.model,
                        run.run_index + 1,
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

                    requests_used = None
                    if (
                        case.output
                        and hasattr(case.output, "metrics")
                        and isinstance(case.output.metrics, dict)
                    ):
                        requests_used = case.output.metrics.get("requests")

                    row.append(requests_used if requests_used is not None else "na")

                    writer.writerow(row)

        return filepath


class MultiModelEvaluator:
    """Main class for running evaluations across multiple models."""

    def __init__(
        self,
        models: List[str],
        judge_model: str,
        output_dir: str = "./results",
        *,
        debug_traces: bool = False,
        trace_dir: Optional[str] = None,
    ):
        self.models = models
        self.judge_model = judge_model
        self.output_dir = output_dir
        self.console = Console()
        self.results: Dict[str, ModelEvaluationResults] = {}

        self._run_output_dir: Optional[str] = None
        self._retry_log_lock = asyncio.Lock()

        # Pre-create the per-invocation output directory to avoid races when running
        # evaluations in parallel.
        self._ensure_run_output_dir()

        self.debug_traces = debug_traces
        if trace_dir is not None:
            self.trace_dir = trace_dir
        elif debug_traces:
            self.trace_dir = os.path.join(self._ensure_run_output_dir(), "debug_traces")
        else:
            self.trace_dir = os.path.join(output_dir, "debug_traces")

        for model in models:
            self.results[model] = ModelEvaluationResults(model)

    def _ensure_run_output_dir(self) -> str:
        """Return the per-invocation timestamped output directory, creating it if needed."""
        if self._run_output_dir is not None:
            return self._run_output_dir

        timestamp = get_timestamp_prefix()
        base_dir = os.path.join(self.output_dir, timestamp)
        run_dir = base_dir
        suffix = 1
        while os.path.exists(run_dir):
            suffix += 1
            run_dir = f"{base_dir}_{suffix}"

        os.makedirs(run_dir, exist_ok=True)
        self._run_output_dir = run_dir

        # Create the retry log with a header so downstream parsing is easy.
        retry_log_path = os.path.join(run_dir, "retries.log")
        if not os.path.exists(retry_log_path):
            with open(retry_log_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "timestamp",
                        "model",
                        "planned_run",
                        "attempt",
                        "retry_reason",
                        "delay_seconds",
                    ]
                )

        return run_dir

    async def _log_retry(
        self,
        *,
        model: str,
        run_index: int,
        attempt: int,
        retry_reason: str,
        delay_seconds: float,
    ) -> None:
        """Append a retry event to the per-invocation retry log."""
        retry_log_path = os.path.join(self._ensure_run_output_dir(), "retries.log")

        async with self._retry_log_lock:
            file_exists = os.path.exists(retry_log_path)
            with open(retry_log_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "model",
                            "planned_run",
                            "attempt",
                            "retry_reason",
                            "delay_seconds",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        model,
                        run_index + 1,
                        attempt,
                        retry_reason,
                        f"{delay_seconds:.3f}",
                    ]
                )

    def _categorize_evaluation_exception(self, exception: Exception) -> str:
        """Categorize evaluation-level exceptions for logging/reporting."""
        error_type = type(exception).__name__
        error_lower = str(exception).lower()

        if "ValidationError" in error_type:
            return "evaluation_validation_failed"

        if isinstance(exception, BaseExceptionGroup):
            return "evaluation_error_ExceptionGroup"

        if "timeout" in error_lower or "timed out" in error_lower:
            return "evaluation_timeout"

        if "ModelHTTPError" in error_type:
            status_code = getattr(exception, "status_code", None)
            if isinstance(status_code, int):
                return f"http_error_{status_code}"
            return "model_api_error"

        if "ConnectionError" in error_type or "network" in error_lower:
            return "network_error"

        return f"evaluation_error_{error_type}"

    def _get_retry_reason_from_report(self, report: EvaluationReport) -> Optional[str]:
        """Return a retry reason if the report contains retryable case failures."""
        retry_reason: Optional[str] = None
        has_usage_limit = False

        for case in report.cases:
            output = getattr(case, "output", None)
            if not output:
                continue

            failure_reason = getattr(output, "failure_reason", "") or ""
            if not failure_reason:
                continue

            if failure_reason == "usage_limit_exceeded":
                # Policy: do not retry on usage limit failures
                has_usage_limit = True
                continue

            if failure_reason in {"agent_timeout", "timeout_error"}:
                # Policy: do not retry per-case model timeouts
                continue

            if failure_reason == "error_ExceptionGroup":
                retry_reason = retry_reason or failure_reason
                continue

            if failure_reason in {"connection_error", "network_error", "rate_limit_error"}:
                retry_reason = retry_reason or failure_reason
                continue

            if failure_reason.startswith("http_error_"):
                try:
                    status_code = int(failure_reason.split("_")[-1])
                except ValueError:
                    continue
                if status_code in RETRYABLE_HTTP_STATUS_CODES:
                    retry_reason = retry_reason or failure_reason

        # If usage limits were hit, don't retry (even if other retryables are present)
        if has_usage_limit:
            return None

        return retry_reason

    async def run_single_evaluation(
        self, model: str, run_index: int, dataset, timeout: int = 120
    ) -> Optional[EvaluationRunResult]:
        """Runs a single evaluation for a model with retries + exponential backoff.

        A "run" corresponds to one planned report slot, and may be attempted
        multiple times to mitigate transient provider/network issues.

        Args:
            model: The model to evaluate.
            run_index: The index of the run (0-based).
            dataset: The evaluation dataset.
            timeout: The timeout in seconds for the evaluation (outer).

        Returns:
            An EvaluationRunResult if a report was produced, otherwise None.
        """

        attempts = 0
        last_retry_reason = ""

        while attempts < MAX_RETRY_ATTEMPTS:
            attempts += 1
            evaluation_name = (
                f"{model}-multi-mcp-mermaid-diagram-fix-run{run_index + 1}"
                f"-attempt{attempts}"
            )

            async def fix_with_model(inputs: MermaidInput) -> MermaidOutput:
                trace_ctx = None
                if self.debug_traces:
                    trace_ctx = MermaidEvalTraceContext(
                        enabled=True,
                        trace_dir=self.trace_dir,
                        model=model,
                        planned_run_index=run_index,
                        run_attempt=attempts,
                        evaluation_name=evaluation_name,
                    )
                return await fix_mermaid_diagram(inputs, model=model, trace_ctx=trace_ctx)

            try:
                report = await asyncio.wait_for(
                    dataset.evaluate(
                        fix_with_model,
                        name=evaluation_name,
                        max_concurrency=1,
                    ),
                    timeout=timeout,
                )

            except MermaidValidatorUnavailableError:
                # Fatal: evaluation is invalid without validator
                logfire.error(
                    "Mermaid validator unavailable; aborting evaluation",
                    model=model,
                    run_index=run_index,
                    attempts=attempts,
                )
                raise

            except asyncio.TimeoutError:
                # Policy: outer evaluation timeout is retryable
                last_retry_reason = "evaluation_timeout"

                if attempts < MAX_RETRY_ATTEMPTS:
                    delay = compute_exponential_backoff_delay(
                        attempts - 1,
                        base_delay=BASE_RETRY_DELAY,
                        max_delay=MAX_RETRY_DELAY,
                        jitter=True,
                    )
                    logfire.warning(
                        "Evaluation timeout; retrying",
                        model=model,
                        run_index=run_index,
                        attempts=attempts,
                        max_attempts=MAX_RETRY_ATTEMPTS,
                        timeout=timeout,
                        delay_seconds=delay,
                    )
                    await self._log_retry(
                        model=model,
                        run_index=run_index,
                        attempt=attempts,
                        retry_reason=last_retry_reason,
                        delay_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logfire.warning(
                    "Evaluation timeout; max attempts exhausted",
                    model=model,
                    run_index=run_index,
                    attempts=attempts,
                    timeout=timeout,
                )
                self.results[model].add_failed_run(
                    run_index, "evaluation_timeout", attempts
                )
                return None

            except Exception as e:
                # Fatal: Mermaid validator MCP/server outage (best-effort detection)
                if is_mermaid_validator_unavailable_error(e):
                    raise MermaidValidatorUnavailableError(
                        "Mermaid validator MCP server unavailable"
                    ) from e

                categorized_error = self._categorize_evaluation_exception(e)
                last_retry_reason = categorized_error

                retryable = is_retryable_error(e)
                if retryable and attempts < MAX_RETRY_ATTEMPTS:
                    delay = compute_exponential_backoff_delay(
                        attempts - 1,
                        base_delay=BASE_RETRY_DELAY,
                        max_delay=MAX_RETRY_DELAY,
                        jitter=True,
                    )
                    logfire.warning(
                        "Retryable evaluation error; retrying",
                        model=model,
                        run_index=run_index,
                        attempts=attempts,
                        max_attempts=MAX_RETRY_ATTEMPTS,
                        categorized_error=categorized_error,
                        error_type=type(e).__name__,
                        error=str(e),
                        delay_seconds=delay,
                    )
                    await self._log_retry(
                        model=model,
                        run_index=run_index,
                        attempt=attempts,
                        retry_reason=categorized_error,
                        delay_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logfire.error(
                    "Evaluation error; not retrying",
                    model=model,
                    run_index=run_index,
                    attempts=attempts,
                    categorized_error=categorized_error,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                self.results[model].add_failed_run(run_index, categorized_error, attempts)
                return None

            retry_reason = self._get_retry_reason_from_report(report)

            if retry_reason and attempts < MAX_RETRY_ATTEMPTS:
                last_retry_reason = retry_reason
                delay = compute_exponential_backoff_delay(
                    attempts - 1,
                    base_delay=BASE_RETRY_DELAY,
                    max_delay=MAX_RETRY_DELAY,
                    jitter=True,
                )
                logfire.warning(
                    "Retryable case failure detected; retrying run",
                    model=model,
                    run_index=run_index,
                    attempts=attempts,
                    max_attempts=MAX_RETRY_ATTEMPTS,
                    retry_reason=retry_reason,
                    delay_seconds=delay,
                )
                await self._log_retry(
                    model=model,
                    run_index=run_index,
                    attempt=attempts,
                    retry_reason=retry_reason,
                    delay_seconds=delay,
                )
                await asyncio.sleep(delay)
                continue

            if retry_reason:
                logfire.warning(
                    "Retryable case failure detected; accepting report (max attempts exhausted)",
                    model=model,
                    run_index=run_index,
                    attempts=attempts,
                    retry_reason=retry_reason,
                )

            return EvaluationRunResult(
                run_index=run_index,
                attempts=attempts,
                last_retry_reason=last_retry_reason,
                report=report,
            )

        # Defensive: should be unreachable
        self.results[model].add_failed_run(run_index, "evaluation_unknown_failure", attempts)
        return None

    async def run_model_evaluations(
        self,
        model: str,
        n_runs: int,
        dataset,
        parallel: bool = True,
        timeout: int = 600,
    ) -> None:
        """Runs multiple evaluations for a single model.

        Args:
            model: The model to evaluate.
            n_runs: The number of runs to perform.
            dataset: The evaluation dataset.
            parallel: Whether to run the evaluations in parallel.
            timeout: The timeout in seconds for each evaluation.
        """
        self.console.print(f"\n[bold cyan]Evaluating model: {model}[/bold cyan]")

        if parallel:
            tasks = [
                self.run_single_evaluation(model, i, dataset, timeout) for i in range(n_runs)
            ]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    f"Running {n_runs} evaluations for {model}", total=n_runs
                )

                try:
                    results = await asyncio.gather(*tasks)
                except MermaidValidatorUnavailableError:
                    raise
                except Exception as e:
                    logfire.error(
                        "Unexpected error during parallel evaluation execution",
                        model=model,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    raise

                for result in results:
                    if result is not None:
                        self.results[model].add_successful_run(result)
                    progress.advance(task)
        else:
            for i in range(n_runs):
                self.console.print(f"[yellow]Run {i + 1}/{n_runs} for {model}[/yellow]")

                result = await self.run_single_evaluation(model, i, dataset, timeout)
                if result is not None:
                    self.results[model].add_successful_run(result)

        individual_file = self.results[model].write_individual_results(
            self._ensure_run_output_dir()
        )
        if individual_file:
            self.console.print(
                f"[green]Individual results saved: {individual_file}[/green]"
            )

        success_rate = self.results[model].get_success_rate()
        successful_runs = len(self.results[model].successful_runs)
        failed_runs = len(self.results[model].failed_runs)

        self.console.print(f"[bold]Model {model} Summary:[/bold]")
        self.console.print(f"  Successful runs: {successful_runs}")
        self.console.print(f"  Failed runs: {failed_runs}")
        self.console.print(f"  Success rate: {success_rate:.1%}")

    def write_combined_results(self) -> str:
        """Writes combined results from all models to a single CSV file.

        Returns:
            The path to the combined results CSV file.
        """
        run_output_dir = self._ensure_run_output_dir()
        filepath = os.path.join(run_output_dir, "combined_results.csv")

        all_evaluator_names = set()
        all_metric_names = set()

        for model_results in self.results.values():
            for run in model_results.successful_runs:
                report = run.report
                for case in report.cases:
                    all_evaluator_names.update(case.scores.keys())
                    if hasattr(case.output, "metrics") and case.output.metrics:
                        all_metric_names.update(case.output.metrics.keys())

        headers = [
            "Model",
            "Run",
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

        headers.append(REQUEST_USAGE_COLUMN)

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for model, model_results in self.results.items():
                for run in model_results.successful_runs:
                    report = run.report
                    for case in report.cases:
                        row = [
                            model,
                            run.run_index + 1,
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

                        requests_used = None
                        if (
                            case.output
                            and hasattr(case.output, "metrics")
                            and isinstance(case.output.metrics, dict)
                        ):
                            requests_used = case.output.metrics.get("requests")

                        row.append(requests_used if requests_used is not None else "na")

                        writer.writerow(row)

        return filepath

    def print_final_summary(self) -> None:
        """Prints a comprehensive summary of all results."""
        table = Table(title="Multi-Model Evaluation Summary")

        table.add_column("Model", style="cyan")
        table.add_column("Successful Runs", style="green")
        table.add_column("Failed Runs", style="red")
        table.add_column("Success Rate", style="yellow")
        table.add_column("Avg Duration", style="blue")

        for model, model_results in self.results.items():
            successful_runs = len(model_results.successful_runs)
            failed_runs = len(model_results.failed_runs)
            success_rate = model_results.get_success_rate()

            if model_results.successful_runs:
                all_durations = []
                for run in model_results.successful_runs:
                    report = run.report
                    for case in report.cases:
                        all_durations.append(case.task_duration)
                avg_duration = statistics.mean(all_durations) if all_durations else 0
            else:
                avg_duration = 0

            table.add_row(
                model,
                str(successful_runs),
                str(failed_runs),
                f"{success_rate:.1%}",
                f"{avg_duration:.1f}s",
            )

        self.console.print(table)

    async def run_all_evaluations(
        self, n_runs: int, parallel: bool = True, timeout: int = 120
    ) -> str:
        """Runs evaluations for all models and returns the path to the combined results.

        Args:
            n_runs: The number of runs per model.
            parallel: Whether to run the evaluations in parallel.
            timeout: The timeout in seconds for each evaluation.

        Returns:
            The path to the combined results CSV file.
        """
        self.console.print("[bold green]Starting multi-model evaluation[/bold green]")
        self.console.print(f"Models: {', '.join(self.models)}")
        self.console.print(f"Runs per model: {n_runs}")
        self.console.print(f"Parallel execution: {parallel}")
        self.console.print(f"Timeout per run: {timeout}s")
        if self.debug_traces:
            self.console.print(f"Debug traces: enabled ({self.trace_dir})")
        else:
            self.console.print("Debug traces: disabled")

        dataset = create_evaluation_dataset(self.judge_model)

        for model in self.models:
            await self.run_model_evaluations(model, n_runs, dataset, parallel, timeout)

        combined_file = self.write_combined_results()

        self.print_final_summary()

        self.console.print("\n[bold green]All evaluations complete![/bold green]")
        self.console.print(f"Combined results: {combined_file}")

        return combined_file


async def main() -> None:
    """The main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run mermaid diagram evaluations across multiple LLM models"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of models to evaluate",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=15,
        help="Number of evaluation runs per model",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-pro",
        help="Model to use for LLM judging",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run evaluations in parallel",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run evaluations sequentially (overrides --parallel)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each evaluation run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mermaid_eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--debug-traces",
        action="store_true",
        default=False,
        help="Write per-case debug trace JSON files (prompt, full message history incl. tool calls/returns, usage)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Directory to save debug traces (default: <output-dir>/debug_traces)",
    )

    args = parser.parse_args()

    models = [model.strip() for model in args.models.split(",")]

    parallel = args.parallel and not args.sequential

    evaluator = MultiModelEvaluator(
        models=models,
        judge_model=args.judge_model,
        output_dir=args.output_dir,
        debug_traces=args.debug_traces,
        trace_dir=args.trace_dir,
    )

    combined_results_file = await evaluator.run_all_evaluations(
        n_runs=args.runs, parallel=parallel, timeout=args.timeout
    )

    print(f"\nEvaluation complete. Combined results saved to: {combined_results_file}")


if __name__ == "__main__":
    asyncio.run(main())
