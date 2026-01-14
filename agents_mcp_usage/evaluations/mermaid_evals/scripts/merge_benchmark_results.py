#!/usr/bin/env python3
"""Merge multiple Merbench processed JSON result files.

This script combines one or more JSON files produced by
`preprocess_merbench_data.py`, optionally deduplicates runs for the
same (Model, Case) combination using different strategies, and then
recomputes all aggregate sections from the merged raw data.

The output JSON has the same schema as a single processed file:
- stats
- leaderboard
- pareto_data
- test_groups_data
- failure_analysis_data
- cost_breakdown_data
- raw_data
- config

Usage examples (as documented in README):

    python scripts/merge_benchmark_results.py \
        -i file1.json file2.json \
        -o merged.json

    python scripts/merge_benchmark_results.py \
        -i file1.json file2.json \
        -o merged.json \
        --dedup keep-first

    python scripts/merge_benchmark_results.py \
        -i file1.json file2.json \
        -o merged.json \
        --report merge_report.json \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import pandas as pd

# Add parent directory to path to import project modules (mirrors preprocess script)
sys.path.append(str(Path(__file__).parent.parent))

from agents_mcp_usage.utils import get_project_root


DedupStrategy = Literal["keep-all", "keep-first", "keep-last", "average"]


@dataclass
class InputFileSummary:
    path: str
    total_runs: int
    models: List[str]
    test_cases: List[str]


def load_json_file(path: Path, verbose: bool = False) -> Dict[str, Any]:
    if verbose:
        print(f"Loading {path}...")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_raw_data(
    datasets: Iterable[Dict[str, Any]],
    dedup_strategy: DedupStrategy = "keep-all",
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge raw_data arrays from multiple processed JSON datasets.

    Deduplication is performed on (Model, Case) pairs when requested.
    """

    all_rows: List[Dict[str, Any]] = []
    for data in datasets:
        raw = data.get("raw_data", [])
        if not isinstance(raw, list):
            raise ValueError("Each input JSON must contain a 'raw_data' list.")
        all_rows.extend(raw)

    if not all_rows:
        # Return empty DataFrame with no rows
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)

    # First, drop exact duplicate rows across all columns to avoid double-counting
    # when the same processed JSON is merged multiple times.
    df_all = df_all.drop_duplicates().reset_index(drop=True)

    # Ensure key columns exist
    for col in ["Model", "Case"]:
        if col not in df_all.columns:
            raise KeyError(f"Expected column '{col}' to be present in raw_data")

    if dedup_strategy == "keep-all":
        if verbose:
            print("Deduplication strategy 'keep-all': keeping all unique rows (no per-(Model, Case) deduplication).")
        return df_all.reset_index(drop=True)

    key_cols = ["Model", "Case"]

    if dedup_strategy in {"keep-first", "keep-last"}:
        keep_arg = "first" if dedup_strategy == "keep-first" else "last"
        if verbose:
            print(f"Deduplication strategy '{dedup_strategy}': dropping duplicates, keeping {keep_arg}.")
        df_dedup = df_all.drop_duplicates(subset=key_cols, keep=keep_arg).reset_index(drop=True)
        return df_dedup

    if dedup_strategy == "average":
        if verbose:
            print("Deduplication strategy 'average': averaging numeric fields for duplicate (Model, Case) pairs.")

        # Identify numeric vs non-numeric columns
        numeric_cols = df_all.select_dtypes(include="number").columns.tolist()

        agg_dict = {}
        for col in df_all.columns:
            if col in key_cols:
                # Key columns are included in groupby, no aggregation spec needed
                continue
            if col in numeric_cols:
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"

        grouped = df_all.groupby(key_cols, as_index=False).agg(agg_dict)
        return grouped.reset_index(drop=True)

    raise ValueError(f"Unknown deduplication strategy: {dedup_strategy}")


def calculate_failure_analysis_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calculate failure counts by model and failure type.

    This mirrors the logic in preprocess_merbench_data.calculate_failure_analysis_data,
    but operates on the merged DataFrame.
    """

    failure_series = [
        {"name": "Invalid Diagram", "column": "Score_MermaidDiagramValid", "condition": "== 0"},
        {"name": "MCP Tool Failure", "column": "Score_UsedBothMCPTools", "condition": "< 1"},
        {"name": "Usage Limit Exceeded", "column": "Score_UsageLimitNotExceeded", "condition": "== 0"},
    ]

    if df.empty:
        return []

    models = sorted(df["Model"].unique())
    failure_data: List[Dict[str, Any]] = []

    for model in models:
        model_data = df[df["Model"] == model]
        failure_counts: Dict[str, Any] = {"Model": model}

        for series in failure_series:
            column = series["column"]
            if column not in model_data.columns:
                # If a score column is missing entirely, treat as zero failures
                failure_counts[series["name"]] = 0
                continue
            condition_str = f"`{column}` {series['condition']}"
            count = model_data.eval(condition_str).sum()
            failure_counts[series["name"]] = int(count)

        failure_data.append(failure_counts)

    return failure_data


def build_aggregates(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build all aggregate sections from merged raw_data DataFrame.

    This mirrors the logic in preprocess_merbench_data.process_csv_for_static_site,
    but assumes costs and tokens are already computed in raw_data.
    """

    if df.empty:
        # Return an empty-but-well-formed structure
        stats = {
            "total_runs": 0,
            "models_evaluated": 0,
            "test_cases": 0,
            "test_groups": [],
            "providers": [],
            "models": [],
            "total_cost": 0.0,
            "avg_cost_per_run": 0.0,
        }

        return {
            "stats": stats,
            "leaderboard": [],
            "pareto_data": [],
            "test_groups_data": [],
            "failure_analysis_data": [],
            "cost_breakdown_data": [],
            "raw_data": [],
            "config": config or {},
        }

    # Ensure expected columns exist; fill missing numeric columns with 0
    numeric_defaults = [
        "Duration",
        "Score_MermaidDiagramValid",
        "Score_UsageLimitNotExceeded",
        "Score_UsedBothMCPTools",
        "total_tokens",
        "Metric_request_tokens",
        "Metric_response_tokens",
        "total_cost",
        "input_cost",
        "output_cost",
    ]

    for col in numeric_defaults:
        if col not in df.columns:
            df[col] = 0.0

    # Fill NaN with 0 for numeric columns
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Derive Success_Rate as in preprocess script
    df["Success_Rate"] = df["Score_MermaidDiagramValid"] * 100

    # Leaderboard data (group by Model)
    leaderboard = df.groupby("Model").agg({
        "Success_Rate": "mean",
        "Duration": "mean",
        "total_tokens": "mean",
        "total_cost": "mean",
        "input_cost": "mean",
        "output_cost": "mean",
        "Case": "count",  # Number of runs
        "provider": "first",
    }).reset_index()

    leaderboard.columns = [
        "Model",
        "Success_Rate",
        "Avg_Duration",
        "Avg_Tokens",
        "Avg_Cost",
        "Avg_Input_Cost",
        "Avg_Output_Cost",
        "Runs",
        "Provider",
    ]

    leaderboard = leaderboard.sort_values("Success_Rate", ascending=False)

    # Pareto data (group by Model)
    def _nonzero_mean(series: pd.Series) -> float:
        non_zero = series[series > 0]
        return float(non_zero.mean()) if not non_zero.empty else 0.0

    pareto_data = df.groupby("Model").agg({
        "Success_Rate": "mean",
        "Duration": "mean",
        "total_tokens": "mean",
        "total_cost": "mean",
        "input_cost": "mean",
        "output_cost": "mean",
        "Metric_request_tokens": _nonzero_mean,
        "Metric_response_tokens": _nonzero_mean,
    }).reset_index()

    pareto_data = pareto_data.fillna(0)

    # Test group performance data (group by Model + test_group)
    if "test_group" not in df.columns:
        # Derive a default test_group if missing
        df["test_group"] = "other"

    test_groups_data = df.groupby(["Model", "test_group"]).agg({
        "Score_MermaidDiagramValid": "mean",
        "Score_UsageLimitNotExceeded": "mean",
        "Score_UsedBothMCPTools": "mean",
        "total_cost": "mean",
        "input_cost": "mean",
        "output_cost": "mean",
        "total_tokens": "mean",
    }).reset_index()

    # Failure analysis data
    failure_analysis_data = calculate_failure_analysis_data(df)

    # Cost breakdown data (group by Model + test_group)
    cost_breakdown_data = df.groupby(["Model", "test_group"]).agg({
        "total_cost": ["mean", "sum", "count"],
        "input_cost": ["mean", "sum"],
        "output_cost": ["mean", "sum"],
    }).round(6)

    # Flatten multi-level columns
    cost_breakdown_data.columns = ["_".join(col).strip() for col in cost_breakdown_data.columns.values]
    cost_breakdown_data = cost_breakdown_data.reset_index()
    cost_breakdown_data = cost_breakdown_data.rename(
        columns={
            "total_cost_mean": "avg_total_cost",
            "total_cost_sum": "sum_total_cost",
            "total_cost_count": "run_count",
            "input_cost_mean": "avg_input_cost",
            "input_cost_sum": "sum_input_cost",
            "output_cost_mean": "avg_output_cost",
            "output_cost_sum": "sum_output_cost",
        }
    )

    # Aggregate statistics from merged raw data
    stats = {
        "total_runs": int(len(df)),
        "models_evaluated": int(df["Model"].nunique()),
        "test_cases": int(df["Case"].nunique()),
        "test_groups": sorted(df["test_group"].dropna().unique().tolist()),
        "providers": sorted(df["provider"].dropna().unique().tolist()) if "provider" in df.columns else [],
        "models": sorted(df["Model"].dropna().unique().tolist()),
        "total_cost": float(df["total_cost"].sum()),
        "avg_cost_per_run": float(df["total_cost"].mean() if len(df) > 0 else 0.0),
    }

    # Raw data to records (ensure stable column subset as per schema)
    raw_columns = [
        "Model",
        "Case",
        "test_group",
        "Duration",
        "Score_MermaidDiagramValid",
        "Score_UsageLimitNotExceeded",
        "Score_UsedBothMCPTools",
        "total_tokens",
        "provider",
        "Metric_request_tokens",
        "Metric_response_tokens",
        "total_cost",
        "input_cost",
        "output_cost",
    ]

    # Add any missing columns in the expected raw_data schema
    for col in raw_columns:
        if col not in df.columns:
            df[col] = 0 if col in numeric_defaults else None

    raw_data_records = df[raw_columns].to_dict(orient="records")

    return {
        "stats": stats,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "pareto_data": pareto_data.to_dict(orient="records"),
        "test_groups_data": test_groups_data.to_dict(orient="records"),
        "failure_analysis_data": failure_analysis_data,
        "cost_breakdown_data": cost_breakdown_data.to_dict(orient="records"),
        "raw_data": raw_data_records,
        "config": config or {},
    }


def summarise_input_file(path: Path, data: Dict[str, Any]) -> InputFileSummary:
    raw = data.get("raw_data", [])
    models = sorted({row.get("Model", "") for row in raw if "Model" in row})
    cases = sorted({row.get("Case", "") for row in raw if "Case" in row})

    return InputFileSummary(
        path=str(path),
        total_runs=len(raw),
        models=models,
        test_cases=cases,
    )


def build_merge_report(
    input_summaries: List[InputFileSummary],
    dedup_strategy: DedupStrategy,
    df_all: pd.DataFrame,
    df_merged: pd.DataFrame,
) -> Dict[str, Any]:
    key_cols = ["Model", "Case"]

    total_input_runs = int(len(df_all))
    unique_runs_before = int(df_all.drop_duplicates(subset=key_cols).shape[0]) if not df_all.empty else 0
    unique_runs_after = int(df_merged.drop_duplicates(subset=key_cols).shape[0]) if not df_merged.empty else 0

    duplicates_before = total_input_runs - unique_runs_before
    duplicates_removed = unique_runs_before - unique_runs_after

    summary = {
        "merged_at": datetime.utcnow().isoformat() + "Z",
        "dedup_strategy": dedup_strategy,
        "input_files": [
            {
                "path": s.path,
                "total_runs": s.total_runs,
                "models": s.models,
                "test_cases": s.test_cases,
            }
            for s in input_summaries
        ],
        "summary": {
            "total_input_runs": total_input_runs,
            "unique_runs_before_dedup": unique_runs_before,
            "unique_runs_after_dedup": unique_runs_after,
            "duplicate_runs_detected": duplicates_before,
            "duplicate_runs_removed": max(0, duplicates_removed),
        },
        "models_merged": sorted(df_merged["Model"].dropna().unique().tolist()) if not df_merged.empty else [],
    }

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple Merbench processed JSON result files.",
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        dest="input_files",
        required=True,
        help="Input JSON files produced by preprocess_merbench_data.py",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        required=True,
        help="Path to write merged JSON output",
    )

    parser.add_argument(
        "--dedup",
        dest="dedup_strategy",
        choices=["keep-all", "keep-first", "keep-last", "average"],
        default="keep-all",
        help=(
            "Deduplication strategy for duplicate (Model, Case) combinations: "
            "keep-all (default), keep-first, keep-last, average"
        ),
    )

    parser.add_argument(
        "--report",
        dest="report_file",
        help="Optional path to write a detailed merge report JSON",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    project_root = get_project_root()

    # Resolve file paths relative to project root if not absolute
    input_paths: List[Path] = []
    for p in args.input_files:
        path = Path(p)
        if not path.is_absolute():
            path = project_root / path
        input_paths.append(path)

    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    report_path: Optional[Path] = None
    if args.report_file:
        report_path = Path(args.report_file)
        if not report_path.is_absolute():
            report_path = project_root / report_path

    if args.verbose:
        print("Merging Merbench benchmark results...")
        print(f"- Deduplication strategy: {args.dedup_strategy}")
        print("- Input files:")
        for p in input_paths:
            print(f"  - {p}")
        print(f"- Output file: {output_path}")
        if report_path:
            print(f"- Report file: {report_path}")

    # Load all datasets
    datasets: List[Dict[str, Any]] = []
    input_summaries: List[InputFileSummary] = []

    for path in input_paths:
        data = load_json_file(path, verbose=args.verbose)
        datasets.append(data)
        input_summaries.append(summarise_input_file(path, data))

    # Choose a canonical config from the first dataset (if any)
    config: Dict[str, Any] = {}
    if datasets and isinstance(datasets[0].get("config"), dict):
        config = datasets[0]["config"]

    # Merge raw_data and build aggregates
    df_all = pd.DataFrame([row for data in datasets for row in data.get("raw_data", [])])
    df_merged = merge_raw_data(datasets, dedup_strategy=args.dedup_strategy, verbose=args.verbose)

    if args.verbose:
        print(f"Total input runs: {len(df_all)}")
        print(f"Merged runs after dedup ({args.dedup_strategy}): {len(df_merged)}")

    merged_data = build_aggregates(df_merged, config=config)

    # Write merged JSON output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)

    if args.verbose:
        print(f"Merged results written to {output_path}")

    # Optionally write a merge report
    if report_path is not None:
        report = build_merge_report(
            input_summaries=input_summaries,
            dedup_strategy=args.dedup_strategy,
            df_all=df_all,
            df_merged=df_merged,
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if args.verbose:
            print(f"Merge report written to {report_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
