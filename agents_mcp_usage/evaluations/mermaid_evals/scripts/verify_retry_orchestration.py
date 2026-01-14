#!/usr/bin/env python3
"""
Offline verification for run-level retry + backoff orchestration.

This script uses a stub Dataset to simulate:
- evaluation timeouts (retryable at run-level)
- report-based retry triggers (e.g. http_error_429, error_ExceptionGroup)
- non-retry conditions (usage_limit_exceeded, agent_timeout)
- fatal conditions (Mermaid validator unavailable)

It patches delays to zero so it runs quickly and deterministically.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import agents_mcp_usage.evaluations.mermaid_evals.run_multi_evals as run_multi_evals
from agents_mcp_usage.evaluations.mermaid_evals.evals_pydantic_mcp import (
    MermaidValidatorUnavailableError,
)
from agents_mcp_usage.evaluations.mermaid_evals.run_multi_evals import MultiModelEvaluator


@dataclass(slots=True)
class StubOutput:
    failure_reason: str = ""
    fixed_diagram: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StubCase:
    name: str
    task_duration: float = 0.0
    output: StubOutput = field(default_factory=StubOutput)
    scores: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StubReport:
    cases: list[StubCase]


class StubDataset:
    """A minimal stub that matches the Dataset.evaluate() shape we need."""

    def __init__(self, sequence: Sequence[object]) -> None:
        self.sequence = list(sequence)
        self.calls = 0

    async def evaluate(self, *_args: Any, **_kwargs: Any) -> object:
        item = self.sequence[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


async def _no_sleep(_delay: float) -> None:
    return None


def _make_report(failure_reasons: List[str]) -> StubReport:
    cases = [
        StubCase(name=f"case_{idx}", output=StubOutput(failure_reason=reason))
        for idx, reason in enumerate(failure_reasons)
    ]
    return StubReport(cases=cases)


async def main() -> None:
    # Patch delay + sleep so retries are fast in this script
    run_multi_evals.compute_exponential_backoff_delay = lambda *_a, **_k: 0.0  # type: ignore[assignment]
    run_multi_evals.asyncio.sleep = _no_sleep  # type: ignore[assignment]

    model = "test-model"

    # 1) Evaluation timeout should retry up to MAX_RETRY_ATTEMPTS, then fail.
    evaluator = MultiModelEvaluator(models=[model], judge_model="judge", output_dir="./mermaid_eval_results")
    dataset = StubDataset([asyncio.TimeoutError()] * run_multi_evals.MAX_RETRY_ATTEMPTS)
    result = await evaluator.run_single_evaluation(model, 0, dataset, timeout=0.01)
    assert result is None
    assert evaluator.results[model].failed_runs[0]["attempts"] == run_multi_evals.MAX_RETRY_ATTEMPTS

    # 2) Retryable report failure_reason should trigger a retry, then succeed.
    evaluator = MultiModelEvaluator(models=[model], judge_model="judge", output_dir="./mermaid_eval_results")
    dataset = StubDataset([_make_report(["http_error_429"]), _make_report([""])])
    result = await evaluator.run_single_evaluation(model, 0, dataset, timeout=1)
    assert result is not None
    assert result.attempts == 2
    assert dataset.calls == 2

    # 3) usage_limit_exceeded should NOT trigger a retry.
    evaluator = MultiModelEvaluator(models=[model], judge_model="judge", output_dir="./mermaid_eval_results")
    dataset = StubDataset([_make_report(["usage_limit_exceeded"]), _make_report([""])])
    result = await evaluator.run_single_evaluation(model, 0, dataset, timeout=1)
    assert result is not None
    assert result.attempts == 1
    assert dataset.calls == 1

    # 4) agent_timeout should NOT trigger a retry.
    evaluator = MultiModelEvaluator(models=[model], judge_model="judge", output_dir="./mermaid_eval_results")
    dataset = StubDataset([_make_report(["agent_timeout"]), _make_report([""])])
    result = await evaluator.run_single_evaluation(model, 0, dataset, timeout=1)
    assert result is not None
    assert result.attempts == 1
    assert dataset.calls == 1

    # 5) Mermaid validator unavailability should fail-fast (raise).
    evaluator = MultiModelEvaluator(models=[model], judge_model="judge", output_dir="./mermaid_eval_results")
    dataset = StubDataset([MermaidValidatorUnavailableError("validator down")])
    try:
        await evaluator.run_single_evaluation(model, 0, dataset, timeout=1)
        raise AssertionError("Expected MermaidValidatorUnavailableError")
    except MermaidValidatorUnavailableError:
        pass

    print("verify_retry_orchestration.py: OK")


if __name__ == "__main__":
    asyncio.run(main())