#!/usr/bin/env python3
import pandas as pd
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from agents_mcp_usage.evaluations.mermaid_evals.dashboard_config import DEFAULT_CONFIG
from agents_mcp_usage.evaluations.mermaid_evals.schemas import DashboardConfig
from agents_mcp_usage.utils import get_project_root

def parse_metric_details(metric_details_str):
    """Safely parse JSON string from Metric_details column."""
    if pd.isna(metric_details_str) or not metric_details_str:
        return {}
    try:
        return json.loads(metric_details_str.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return {}

def calculate_failure_analysis_data(df):
    """Calculate failure counts by model and failure type."""
    failure_series = [
        {"name": "Invalid Diagram", "column": "Score_MermaidDiagramValid", "condition": "== 0"},
        {"name": "MCP Tool Failure", "column": "Score_UsedBothMCPTools", "condition": "< 1"},
        {"name": "Usage Limit Exceeded", "column": "Score_UsageLimitNotExceeded", "condition": "== 0"},
    ]
    
    models = sorted(df["Model"].unique())
    failure_data = []
    
    for model in models:
        model_data = df[df["Model"] == model]
        failure_counts = {"Model": model}
        
        for series in failure_series:
            condition_str = f"`{series['column']}` {series['condition']}"
            count = model_data.eval(condition_str).sum()
            failure_counts[series["name"]] = int(count)
        
        failure_data.append(failure_counts)
    
    return failure_data

def process_csv_for_static_site(csv_path):
    """Process CSV file and return data structure for static site."""
    # Load configuration
    config = DashboardConfig(**DEFAULT_CONFIG)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Replace NaN values with 0 for numeric columns
    numeric_columns = ['Metric_request_tokens', 'Metric_response_tokens', 'Metric_total_tokens']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Extract grouping column (test case types)
    df['test_group'] = df['Case'].apply(lambda x: x.split('_')[-1] if '_' in x else 'other')
    
    # Parse metric details to extract token information
    if "Metric_details" in df.columns:
        metric_details = df["Metric_details"].apply(parse_metric_details)
        df["thinking_tokens"] = metric_details.apply(lambda x: x.get("thoughts_tokens", 0))
        df["text_tokens"] = metric_details.apply(lambda x: x.get("text_prompt_tokens", 0))
    else:
        df["thinking_tokens"] = 0
        df["text_tokens"] = 0
    
    # Calculate total tokens
    df["total_tokens"] = df["Metric_total_tokens"].fillna(0)
    
    # Calculate success rate (primary metric)
    df["Success_Rate"] = df["Score_MermaidDiagramValid"] * 100
    
    # Extract provider from model name
    def extract_provider(model_name):
        if model_name.startswith("gemini-"):
            return "Google"
        elif "nova" in model_name.lower():
            return "Amazon"
        elif "claude" in model_name.lower():
            return "Anthropic"
        elif "gpt" in model_name.lower():
            return "OpenAI"
        elif model_name.startswith("o"):
            return "OpenAI"
        else:
            return "Other"
    
    df["provider"] = df["Model"].apply(extract_provider)
    
    # Create leaderboard data
    leaderboard = df.groupby("Model").agg({
        "Success_Rate": "mean",
        "Duration": "mean",
        "total_tokens": "mean",
        "Case": "count",  # Number of runs
        "provider": "first"
    }).reset_index()
    
    leaderboard.columns = ["Model", "Success_Rate", "Avg_Duration", "Avg_Tokens", "Runs", "Provider"]
    leaderboard = leaderboard.sort_values("Success_Rate", ascending=False)
    
    # Create data for Pareto frontier plot
    pareto_data = df.groupby("Model").agg({
        "Success_Rate": "mean",
        "Duration": "mean",
        "total_tokens": "mean",
        "Metric_request_tokens": lambda x: x[x > 0].mean() if any(x > 0) else 0,
        "Metric_response_tokens": lambda x: x[x > 0].mean() if any(x > 0) else 0
    }).reset_index()
    
    # Fill any remaining NaN values with 0
    pareto_data = pareto_data.fillna(0)
    
    # Create test group performance data
    test_groups_data = df.groupby(["Model", "test_group"]).agg({
        "Score_MermaidDiagramValid": "mean",
        "Score_UsageLimitNotExceeded": "mean",
        "Score_UsedBothMCPTools": "mean"
    }).reset_index()
    
    # Calculate failure analysis data
    failure_analysis_data = calculate_failure_analysis_data(df)
    
    # Calculate aggregate statistics
    stats = {
        "total_runs": len(df),
        "models_evaluated": df["Model"].nunique(),
        "test_cases": df["Case"].nunique(),
        "test_groups": sorted(df["test_group"].unique().tolist()),
        "providers": sorted(df["provider"].unique().tolist()),
        "models": sorted(df["Model"].unique().tolist())
    }
    
    # Create final data structure
    output_data = {
        "stats": stats,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "pareto_data": pareto_data.to_dict(orient="records"),
        "test_groups_data": test_groups_data.to_dict(orient="records"),
        "failure_analysis_data": failure_analysis_data,
        "raw_data": df[[
            "Model", "Case", "test_group", "Duration", 
            "Score_MermaidDiagramValid", "Score_UsageLimitNotExceeded", 
            "Score_UsedBothMCPTools", "total_tokens", "provider",
            "Metric_request_tokens", "Metric_response_tokens"
        ]].to_dict(orient="records"),
        "config": {
            "title": config.title,
            "description": config.description,
            "primary_metric": {
                "name": "Success_Rate",
                "label": "Success Rate (%)"
            }
        }
    }
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Process CSV evaluation results for static site")
    parser.add_argument("-i", "--input_csv", nargs="?", help="Path to input CSV file", default=None)
    parser.add_argument("-o", "--output_json", nargs="?", help="Path to output JSON file", default=None)
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    current_month = datetime.now().strftime("%b").lower()
    
    # Set default paths if not provided
    if args.input_csv:
        csv_path = Path(args.input_csv)
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
    else:
        csv_path = project_root / "mermaid_eval_results" / f"latest_combined_results.csv"
    
    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = project_root / "agents_mcp_usage" / "evaluations" / "mermaid_evals" / "results" / f"{current_month}_results_processed.json"
    
    print(f"Processing {csv_path}...")
    data = process_csv_for_static_site(csv_path)
    
    # Convert the data to JSON string, replacing NaN with null
    json_str = json.dumps(data, indent=2)
    # Replace NaN values with null for valid JSON
    json_str = json_str.replace(": NaN", ": null")
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(json_str)
    
    print(f"Data processed and saved to {output_path}")
    print(f"- Total runs: {data['stats']['total_runs']}")
    print(f"- Models evaluated: {data['stats']['models_evaluated']}")
    print(f"- Test cases: {data['stats']['test_cases']}")

if __name__ == "__main__":
    main()
