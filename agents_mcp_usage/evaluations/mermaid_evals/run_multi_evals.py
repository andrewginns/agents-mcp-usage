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
import statistics
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
    MermaidInput,
    MermaidOutput,
    fix_mermaid_diagram,
    create_evaluation_dataset,
    get_timestamp_prefix,
)

load_dotenv()

DEFAULT_MODELS = [
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro",
    # "gemini-2.5-flash",
    # "gemini-2.5-flash-preview-04-17",
    # "openai:o4-mini",
    # "openai:gpt-4.1",
    # "openai:gpt-4.1-mini",
    # "openai:gpt-4.1-nano",
    # "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0",
    # "bedrock:us.anthropic.claude-opus-4-20250514-v1:0",
    # "gemini-2.5-flash-lite-preview-06-17"
    # "bedrock:us.anthropic.claude-3-7-sonnet-20240219-v1:0",
    # "bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    # "bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0",
]

logfire.configure(
    send_to_logfire="if-token-present", service_name="multi-model-mermaid-evals"
)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()


class ModelEvaluationResults:
    """Container for storing and managing evaluation results for a single model."""

    def __init__(self, model: str):
        self.model = model
        self.reports: List[EvaluationReport] = []
        self.failed_runs: List[Dict[str, Any]] = []

    def add_successful_run(self, report: EvaluationReport) -> None:
        """Adds a successful evaluation report.

        Args:
            report: The evaluation report to add.
        """
        self.reports.append(report)

    def add_failed_run(self, run_index: int, error: str) -> None:
        """Adds information about a failed run.

        Args:
            run_index: The index of the failed run.
            error: The error message.
        """
        self.failed_runs.append(
            {
                "run_index": run_index,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_success_rate(self) -> float:
        """Calculates the success rate for this model.

        Returns:
            The success rate as a float.
        """
        total_runs = len(self.reports) + len(self.failed_runs)
        if total_runs == 0:
            return 0.0
        return len(self.reports) / total_runs

    def write_individual_results(self, output_dir: str) -> Optional[str]:
        """Writes individual model results to a CSV file.

        Args:
            output_dir: The directory to write the CSV file to.

        Returns:
            The path to the created CSV file, or None if no results were written.
        """
        if not self.reports:
            return None

        os.makedirs(output_dir, exist_ok=True)
        timestamp = get_timestamp_prefix()
        filepath = os.path.join(
            output_dir, f"{timestamp}_individual_{self.model.replace(':', '_')}.csv"
        )

        all_evaluator_names = set()
        all_metric_names = set()

        for report in self.reports:
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

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for run_idx, report in enumerate(self.reports):
                for case in report.cases:
                    row = [
                        self.model,
                        run_idx + 1,
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

        return filepath


class MultiModelEvaluator:
    """Main class for running evaluations across multiple models."""

    def __init__(
        self, models: List[str], judge_model: str, output_dir: str = "./results"
    ):
        self.models = models
        self.judge_model = judge_model
        self.output_dir = output_dir
        self.console = Console()
        self.results: Dict[str, ModelEvaluationResults] = {}

        for model in models:
            self.results[model] = ModelEvaluationResults(model)

    async def run_single_evaluation(
        self, model: str, run_index: int, dataset, timeout: int = 120
    ) -> Optional[EvaluationReport]:
        """Runs a single evaluation for a model with timeout and error handling.

        Args:
            model: The model to evaluate.
            run_index: The index of the run.
            dataset: The evaluation dataset.
            timeout: The timeout in seconds for the evaluation.

        Returns:
            An EvaluationReport if successful, otherwise None.
        """
        try:

            async def fix_with_model(inputs: MermaidInput) -> MermaidOutput:
                return await fix_mermaid_diagram(inputs, model=model)

            report = await asyncio.wait_for(
                dataset.evaluate(
                    fix_with_model,
                    name=f"{model}-multi-mcp-mermaid-diagram-fix-run{run_index + 1}",
                    max_concurrency=1,
                ),
                timeout=timeout,
            )

            return report

        except asyncio.TimeoutError:
            error_msg = f"Evaluation timed out after {timeout}s"
            logfire.warning(
                "Evaluation timeout", model=model, run_index=run_index, timeout=timeout
            )
            self.results[model].add_failed_run(run_index, "evaluation_timeout")
            return None

        except Exception as e:
            # Categorize the error for better reporting
            error_type = type(e).__name__
            if "ValidationError" in error_type:
                categorized_error = "evaluation_validation_failed"
            elif "timeout" in str(e).lower() or "timed out" in str(e).lower():
                categorized_error = "evaluation_timeout"
            elif "ModelHTTPError" in error_type:
                categorized_error = "model_api_error"
            elif "ConnectionError" in error_type or "network" in str(e).lower():
                categorized_error = "network_error"
            else:
                categorized_error = f"evaluation_error_{error_type}"

            error_msg = f"Error during evaluation: {str(e)}"
            logfire.error(
                f"Evaluation error: {error_msg}",
                model=model,
                run_index=run_index,
                error=str(e),
                error_type=error_type,
                categorized_error=categorized_error,
            )
            self.results[model].add_failed_run(run_index, categorized_error)
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
                self.run_single_evaluation(model, i, dataset, timeout)
                for i in range(n_runs)
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

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, EvaluationReport):
                        self.results[model].add_successful_run(result)
                    progress.advance(task)
        else:
            for i in range(n_runs):
                self.console.print(f"[yellow]Run {i + 1}/{n_runs} for {model}[/yellow]")

                result = await self.run_single_evaluation(model, i, dataset, timeout)
                if result:
                    self.results[model].add_successful_run(result)

        individual_file = self.results[model].write_individual_results(self.output_dir)
        if individual_file:
            self.console.print(
                f"[green]Individual results saved: {individual_file}[/green]"
            )

        success_rate = self.results[model].get_success_rate()
        successful_runs = len(self.results[model].reports)
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
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = get_timestamp_prefix()
        filepath = os.path.join(self.output_dir, f"{timestamp}_combined_results.csv")

        all_evaluator_names = set()
        all_metric_names = set()

        for model_results in self.results.values():
            for report in model_results.reports:
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

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for model, model_results in self.results.items():
                for run_idx, report in enumerate(model_results.reports):
                    for case in report.cases:
                        row = [
                            model,
                            run_idx + 1,
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
            successful_runs = len(model_results.reports)
            failed_runs = len(model_results.failed_runs)
            success_rate = model_results.get_success_rate()

            if model_results.reports:
                all_durations = []
                for report in model_results.reports:
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
        default=5,
        help="Number of evaluation runs per model",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-pro-preview-06-05",
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

    args = parser.parse_args()

    models = [model.strip() for model in args.models.split(",")]

    parallel = args.parallel and not args.sequential

    evaluator = MultiModelEvaluator(
        models=models, judge_model=args.judge_model, output_dir=args.output_dir
    )

    combined_results_file = await evaluator.run_all_evaluations(
        n_runs=args.runs, parallel=parallel, timeout=args.timeout
    )

    print(f"\nEvaluation complete. Combined results saved to: {combined_results_file}")


if __name__ == "__main__":
    asyncio.run(main())