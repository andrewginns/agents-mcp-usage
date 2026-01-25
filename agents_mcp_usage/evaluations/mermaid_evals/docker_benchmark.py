#!/usr/bin/env python3
"""Docker entrypoint for running Merbench evaluations.

This script is designed to be the container entrypoint. It reads configuration
from environment variables (set by the Makefile) and executes the existing
`run_multi_evals.py` script.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with a friendly error message."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        message = f"Environment variable {name} must be an integer, got: {value!r}"
        raise SystemExit(message) from exc


def main() -> int:
    """Run the benchmark using environment-driven configuration."""
    workspace = Path("/workspace")
    if workspace.exists():
        os.chdir(workspace)

    script_path = Path("agents_mcp_usage/evaluations/mermaid_evals/run_multi_evals.py")
    if not script_path.exists():
        print(
            "Could not find run_multi_evals.py under /workspace. "
            "Ensure the repository is bind-mounted to /workspace.",
            file=sys.stderr,
        )
        return 2

    model = os.getenv("MODEL", "").strip()
    if not model:
        print(
            "MODEL is required. Example: MODEL='openai:gpt-5.1 (none)'",
            file=sys.stderr,
        )
        return 2

    runs = env_int("RUNS", 15)
    judge_model = os.getenv("JUDGE_MODEL", "gemini-2.5-pro").strip()
    timeout = env_int("TIMEOUT", 600)
    output_dir = os.getenv("OUTPUT_DIR", "./mermaid_eval_results").strip()
    trace_dir = os.getenv("TRACE_DIR")

    parallel = env_bool("PARALLEL", default=False)
    sequential = env_bool("SEQUENTIAL", default=False)
    debug_traces = env_bool("DEBUG_TRACES", default=False)
    dry_run = env_bool("DRY_RUN", default=False)

    module_name = "agents_mcp_usage.evaluations.mermaid_evals.run_multi_evals"
    cmd = [
        sys.executable,
        "-m",
        module_name,
        "--models",
        model,
        "--runs",
        str(runs),
        "--judge-model",
        judge_model,
        "--timeout",
        str(timeout),
        "--output-dir",
        output_dir,
    ]

    if debug_traces:
        cmd.append("--debug-traces")
    if trace_dir:
        cmd.extend(["--trace-dir", trace_dir])

    if sequential:
        cmd.append("--sequential")
    elif parallel:
        cmd.append("--parallel")

    extra_flags = os.getenv("BENCHMARK_FLAGS", "").strip()
    if extra_flags:
        cmd.extend(shlex.split(extra_flags))

    print(f"[merbench] Executing: {shlex.join(cmd)}")
    if dry_run:
        print("[merbench] DRY_RUN=1, skipping execution.")
        return 0

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
