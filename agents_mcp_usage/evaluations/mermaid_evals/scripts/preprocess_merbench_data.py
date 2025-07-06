#!/usr/bin/env python3
import pandas as pd
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from agents_mcp_usage.evaluations.mermaid_evals.dashboard_config import DEFAULT_CONFIG
from agents_mcp_usage.evaluations.mermaid_evals.schemas import DashboardConfig
from agents_mcp_usage.utils import get_project_root

def load_model_costs(file_path: Path) -> Dict[str, Any]:
    """Load model costs from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert "inf" strings to actual infinity
    def convert_inf_strings(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_inf_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_inf_strings(v) for v in obj]
        elif obj == "inf":
            return float("inf")
        return obj
    
    # Extract model_costs from the loaded data
    model_costs = data.get('model_costs', {})
    return convert_inf_strings(model_costs)

def get_price_for_tokens(token_count: int, price_tiers: List[Dict]) -> float:
    """Find the correct price for a given number of tokens from a list of tiers."""
    for tier in price_tiers:
        if token_count <= tier["up_to"]:
            return tier["price"]
    return price_tiers[-1]["price"]  # Fallback to the highest tier price

def calculate_costs(df: pd.DataFrame, cost_config: Dict, config: DashboardConfig) -> pd.DataFrame:
    """Calculate input, output, and total costs for each run based on tiered pricing.
    
    Sets cost to 0 if Score_UsageLimitNotExceeded == 0.
    """
    df_with_costs = df.copy()
    
    # Get cost calculation config from dashboard config
    cost_calc_config = config.cost_calculation
    input_token_cols = cost_calc_config.input_token_cols
    output_token_cols = cost_calc_config.output_token_cols
    
    # Initialize cost columns
    df_with_costs["input_cost"] = 0.0
    df_with_costs["output_cost"] = 0.0
    df_with_costs["total_cost"] = 0.0
    
    for idx, row in df_with_costs.iterrows():
        # Check if usage limit was exceeded - if so, cost is 0
        if row.get("Score_UsageLimitNotExceeded", 1) == 0:
            continue
            
        model = row.get("Model")
        model_costs = cost_config.get(model)
        
        if not model_costs:
            continue
            
        try:
            # Calculate token counts
            input_tokens = sum(row.get(col, 0) or 0 for col in input_token_cols)
            output_tokens = sum(row.get(col, 0) or 0 for col in output_token_cols)
            thinking_tokens = row.get("thinking_tokens", 0) or 0
            non_thinking_output_tokens = output_tokens - thinking_tokens
            
            total_tokens = input_tokens + output_tokens
            
            # Calculate input cost
            input_price_tiers = model_costs.get("input", [])
            if input_price_tiers:
                input_price = get_price_for_tokens(total_tokens, input_price_tiers)
                input_cost = (input_tokens / 1_000_000) * input_price
            else:
                input_cost = 0.0
                
            # Calculate output cost
            output_cost = 0.0
            output_pricing = model_costs.get("output", {})
            
            if "thinking" in output_pricing and thinking_tokens > 0:
                thinking_price_tiers = output_pricing["thinking"]
                thinking_price = get_price_for_tokens(total_tokens, thinking_price_tiers)
                output_cost += (thinking_tokens / 1_000_000) * thinking_price
                
            if "non_thinking" in output_pricing and non_thinking_output_tokens > 0:
                non_thinking_price_tiers = output_pricing["non_thinking"]
                non_thinking_price = get_price_for_tokens(total_tokens, non_thinking_price_tiers)
                output_cost += (non_thinking_output_tokens / 1_000_000) * non_thinking_price
                
            elif "default" in output_pricing:
                default_price_tiers = output_pricing["default"]
                default_price = get_price_for_tokens(total_tokens, default_price_tiers)
                output_cost += (output_tokens / 1_000_000) * default_price
                
            df_with_costs.at[idx, "input_cost"] = input_cost
            df_with_costs.at[idx, "output_cost"] = output_cost
            df_with_costs.at[idx, "total_cost"] = input_cost + output_cost
            
        except (TypeError, KeyError, IndexError) as e:
            print(f"Cost calculation error for model {model} at row {idx}: {e}")
            
    return df_with_costs

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
    
    # Load cost configuration
    project_root = get_project_root()
    costs_json_path = project_root / "agents_mcp_usage" / "evaluations" / "mermaid_evals" / "costs.json"
    cost_config = load_model_costs(costs_json_path) if costs_json_path.exists() else {}
    
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
    
    # Calculate costs for each row
    df = calculate_costs(df, cost_config, config)
    
    # Create leaderboard data
    leaderboard = df.groupby("Model").agg({
        "Success_Rate": "mean",
        "Duration": "mean",
        "total_tokens": "mean",
        "total_cost": "mean",
        "input_cost": "mean",
        "output_cost": "mean",
        "Case": "count",  # Number of runs
        "provider": "first"
    }).reset_index()
    
    leaderboard.columns = ["Model", "Success_Rate", "Avg_Duration", "Avg_Tokens", "Avg_Cost", "Avg_Input_Cost", "Avg_Output_Cost", "Runs", "Provider"]
    leaderboard = leaderboard.sort_values("Success_Rate", ascending=False)
    
    # Create data for Pareto frontier plot
    pareto_data = df.groupby("Model").agg({
        "Success_Rate": "mean",
        "Duration": "mean",
        "total_tokens": "mean",
        "total_cost": "mean",
        "input_cost": "mean",
        "output_cost": "mean",
        "Metric_request_tokens": lambda x: x[x > 0].mean() if any(x > 0) else 0,
        "Metric_response_tokens": lambda x: x[x > 0].mean() if any(x > 0) else 0
    }).reset_index()
    
    # Fill any remaining NaN values with 0
    pareto_data = pareto_data.fillna(0)
    
    # Create test group performance data
    test_groups_data = df.groupby(["Model", "test_group"]).agg({
        "Score_MermaidDiagramValid": "mean",
        "Score_UsageLimitNotExceeded": "mean",
        "Score_UsedBothMCPTools": "mean",
        "total_cost": "mean",
        "input_cost": "mean",
        "output_cost": "mean",
        "total_tokens": "mean"
    }).reset_index()
    
    # Calculate failure analysis data
    failure_analysis_data = calculate_failure_analysis_data(df)
    
    # Calculate cost breakdown by model and test group
    cost_breakdown_data = df.groupby(["Model", "test_group"]).agg({
        "total_cost": ["mean", "sum", "count"],
        "input_cost": ["mean", "sum"],
        "output_cost": ["mean", "sum"]
    }).round(6)
    
    # Flatten the multi-level columns
    cost_breakdown_data.columns = ['_'.join(col).strip() for col in cost_breakdown_data.columns.values]
    cost_breakdown_data = cost_breakdown_data.reset_index()
    cost_breakdown_data = cost_breakdown_data.rename(columns={
        "total_cost_mean": "avg_total_cost",
        "total_cost_sum": "sum_total_cost", 
        "total_cost_count": "run_count",
        "input_cost_mean": "avg_input_cost",
        "input_cost_sum": "sum_input_cost",
        "output_cost_mean": "avg_output_cost",
        "output_cost_sum": "sum_output_cost"
    })
    
    # Calculate aggregate statistics
    stats = {
        "total_runs": len(df),
        "models_evaluated": df["Model"].nunique(),
        "test_cases": df["Case"].nunique(),
        "test_groups": sorted(df["test_group"].unique().tolist()),
        "providers": sorted(df["provider"].unique().tolist()),
        "models": sorted(df["Model"].unique().tolist()),
        "total_cost": df["total_cost"].sum(),
        "avg_cost_per_run": df["total_cost"].mean()
    }
    
    # Create final data structure
    output_data = {
        "stats": stats,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "pareto_data": pareto_data.to_dict(orient="records"),
        "test_groups_data": test_groups_data.to_dict(orient="records"),
        "failure_analysis_data": failure_analysis_data,
        "cost_breakdown_data": cost_breakdown_data.to_dict("records"),
        "raw_data": df[[
            "Model", "Case", "test_group", "Duration", 
            "Score_MermaidDiagramValid", "Score_UsageLimitNotExceeded", 
            "Score_UsedBothMCPTools", "total_tokens", "provider",
            "Metric_request_tokens", "Metric_response_tokens",
            "total_cost", "input_cost", "output_cost"
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
        csv_path = project_root / "mermaid_eval_results" / "latest_combined_results.csv"
    
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
