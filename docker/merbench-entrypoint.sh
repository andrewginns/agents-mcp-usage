#!/bin/sh
set -e

# Ensure results directory exists if bind-mounted
mkdir -p /app/mermaid_eval_results

exec uv run agents_mcp_usage/evaluations/mermaid_evals/run_multi_evals.py "$@"
