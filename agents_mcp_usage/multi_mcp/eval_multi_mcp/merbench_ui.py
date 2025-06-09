import os
import glob
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from typing import List, Dict
import json
import re
from pydantic import ValidationError
import csv
import io

from agents_mcp_usage.multi_mcp.eval_multi_mcp.dashboard_config import (
    DEFAULT_CONFIG,
)
from agents_mcp_usage.multi_mcp.eval_multi_mcp.schemas import DashboardConfig

# Load and validate the configuration
try:
    EVAL_CONFIG = DashboardConfig(**DEFAULT_CONFIG)
except ValidationError as e:
    st.error(f"Dashboard configuration error: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title=EVAL_CONFIG.title, page_icon=EVAL_CONFIG.icon, layout="wide"
)

# --- Cost Loading ---


def load_model_costs(file_path: str) -> Dict:
    """Loads model costs from a CSV file and returns a structured dictionary."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Read lines, skipping comments and empty lines
            lines = [
                line for line in f if not line.strip().startswith("#") and line.strip()
            ]

            # Find the start of the dictionary-like definition
            dict_str = "".join(lines)
            match = re.search(r"MODEL_COSTS\s*=\s*({.*})", dict_str, re.DOTALL)
            if not match:
                st.error(f"Could not find 'MODEL_COSTS' dictionary in {file_path}")
                return {}

            # Safely evaluate the dictionary string
            model_costs_raw = eval(match.group(1), {"float": float})

            return model_costs_raw

    except FileNotFoundError:
        st.warning(f"Cost file not found at {file_path}. Using empty cost config.")
        return {}
    except (SyntaxError, NameError, Exception) as e:
        st.error(f"Error parsing cost file {file_path}: {e}")
        return {}


# --- Data Loading and Processing ---


def find_all_combined_results_csvs(directory_path: str) -> list[str]:
    """Finds all '*_combined_results.csv' files, sorted by modification time."""
    if not os.path.isdir(directory_path):
        return []
    try:
        search_pattern = os.path.join(directory_path, "*_combined_results.csv")
        files = glob.glob(search_pattern)
        return sorted(files, key=os.path.getmtime, reverse=True)
    except Exception as e:
        st.error(f"Error finding CSV files in '{directory_path}': {e}")
        return []


def detect_csv_files(directory: str = None) -> List[str]:
    """Detect CSV result files in the specified directory."""
    if directory is None:
        # Go up three levels from the current script to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        directory = os.path.join(project_root, "mermaid_eval_results")
    return find_all_combined_results_csvs(directory)


def load_csv_data(file_paths: List[str]) -> pd.DataFrame:
    """Load and combine multiple CSV files into a single DataFrame."""
    if not file_paths:
        return pd.DataFrame()

    dataframes = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                df["source_file"] = os.path.basename(file_path)
                dataframes.append(df)
            else:
                st.warning(f"Empty file skipped: {os.path.basename(file_path)}")
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            st.error(f"Could not load {os.path.basename(file_path)}: {e}")
        except Exception as e:
            st.error(
                f"An unexpected error occurred while loading {os.path.basename(file_path)}: {e}"
            )

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def extract_grouping_column(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Extracts a grouping column based on regex from the config."""
    source_col = config["grouping"]["column"]
    target_col = config["grouping"]["target_column"]
    regex = config["grouping"]["extractor_regex"]

    def extract_group(value: str) -> str:
        match = re.match(regex, str(value))
        return match.group(1) if match else "other"

    df[target_col] = df[source_col].apply(extract_group)
    return df


def parse_metric_details(metric_details_str: str) -> Dict:
    """Safely parses a JSON string from the 'Metric_details' column."""
    if pd.isna(metric_details_str) or not metric_details_str:
        return {}
    try:
        # Attempt to fix common JSON issues like single quotes
        return json.loads(metric_details_str.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return {}


def get_price_for_tokens(token_count: int, price_tiers: List[Dict]) -> float:
    """Finds the correct price for a given number of tokens from a list of tiers."""
    for tier in price_tiers:
        if token_count <= tier["up_to"]:
            return tier["price"]
    return price_tiers[-1]["price"]  # Fallback to the highest tier price


def calculate_costs(
    df: pd.DataFrame, cost_config: Dict, eval_config: Dict
) -> pd.DataFrame:
    """Calculates input, output, and total costs for each run based on new tiered pricing."""
    df_with_costs = df.copy()
    cost_calc_config = eval_config.get("cost_calculation", {})
    input_token_cols = cost_calc_config.get("input_token_cols", [])
    output_token_cols = cost_calc_config.get("output_token_cols", [])

    df_with_costs["input_cost"] = 0.0
    df_with_costs["output_cost"] = 0.0
    df_with_costs["total_cost"] = 0.0

    for idx, row in df_with_costs.iterrows():
        model = row.get("Model")
        model_costs = cost_config.get(model)

        if not model_costs:
            continue

        try:
            input_tokens = sum(row.get(col, 0) or 0 for col in input_token_cols)
            output_tokens = sum(row.get(col, 0) or 0 for col in output_token_cols)
            thinking_tokens = row.get("thinking_tokens", 0) or 0
            non_thinking_output_tokens = output_tokens - thinking_tokens

            total_tokens = input_tokens + output_tokens

            # Determine input cost
            input_price_tiers = model_costs.get("input", [])
            input_price = get_price_for_tokens(total_tokens, input_price_tiers)
            input_cost = (input_tokens / 1_000_000) * input_price

            # Determine output cost
            output_cost = 0
            output_pricing = model_costs.get("output", {})

            if "thinking" in output_pricing and thinking_tokens > 0:
                thinking_price_tiers = output_pricing["thinking"]
                thinking_price = get_price_for_tokens(
                    total_tokens, thinking_price_tiers
                )
                output_cost += (thinking_tokens / 1_000_000) * thinking_price

            if "non_thinking" in output_pricing and non_thinking_output_tokens > 0:
                non_thinking_price_tiers = output_pricing["non_thinking"]
                non_thinking_price = get_price_for_tokens(
                    total_tokens, non_thinking_price_tiers
                )
                output_cost += (
                    non_thinking_output_tokens / 1_000_000
                ) * non_thinking_price

            elif "default" in output_pricing:
                default_price_tiers = output_pricing["default"]
                default_price = get_price_for_tokens(total_tokens, default_price_tiers)
                output_cost += (output_tokens / 1_000_000) * default_price

            df_with_costs.at[idx, "input_cost"] = input_cost
            df_with_costs.at[idx, "output_cost"] = output_cost
            df_with_costs.at[idx, "total_cost"] = input_cost + output_cost

        except (TypeError, KeyError, IndexError) as e:
            st.warning(f"Cost calculation error for model {model} at row {idx}: {e}")

    return df_with_costs


def process_data(
    df: pd.DataFrame, cost_config: Dict, eval_config: DashboardConfig
) -> pd.DataFrame:
    """Main data processing pipeline."""
    if df.empty:
        return df

    processed_df = df.copy()

    # Generic grouping
    processed_df = extract_grouping_column(processed_df, eval_config.model_dump())

    # Extract token counts from metric details (assuming 'Metric_details' exists)
    if "Metric_details" in processed_df.columns:
        metric_details = processed_df["Metric_details"].apply(parse_metric_details)
        processed_df["thinking_tokens"] = metric_details.apply(
            lambda x: x.get("thoughts_tokens", 0)
        )
        processed_df["text_tokens"] = metric_details.apply(
            lambda x: x.get("text_prompt_tokens", 0)
        )
    else:
        # Ensure these columns exist even if Metric_details is missing
        processed_df["thinking_tokens"] = 0
        processed_df["text_tokens"] = 0

    # Calculate total response tokens
    processed_df["total_response_tokens"] = (
        processed_df.get("Metric_response_tokens", 0) + processed_df["thinking_tokens"]
    )

    # Calculate total tokens for leaderboard
    cost_calc_config = eval_config.cost_calculation
    input_token_cols = cost_calc_config.input_token_cols
    output_token_cols = cost_calc_config.output_token_cols

    processed_df["total_tokens"] = 0
    for col in input_token_cols + output_token_cols:
        processed_df["total_tokens"] += processed_df.get(col, 0).fillna(0)

    # Standardize primary metric score
    primary_metric_config = eval_config.primary_metric
    if (
        primary_metric_config.name not in processed_df.columns
        and primary_metric_config.score_column
    ):
        if primary_metric_config.score_column in processed_df.columns:
            # Create the primary metric from the specified score column, defaulting to 0
            processed_df[primary_metric_config.name] = (
                processed_df.get(primary_metric_config.score_column, 0) * 100
            )
        else:
            st.warning(
                f"Specified score_column '{primary_metric_config.score_column}' not found. Primary metric will be 0."
            )
            processed_df[primary_metric_config.name] = 0
    elif primary_metric_config.name in processed_df.columns:
        # if the column already exists, make sure it's scaled to 100
        processed_df[primary_metric_config.name] = (
            processed_df[primary_metric_config.name] * 100
        )
    else:
        st.error(
            f"Primary metric column '{primary_metric_config.name}' not found and no 'score_column' provided in config."
        )
        st.stop()

    return calculate_costs(processed_df, cost_config, eval_config.model_dump())


# --- UI & Plotting ---


def create_leaderboard(
    df: pd.DataFrame, selected_groups: List[str], config: Dict
) -> pd.DataFrame:
    """Creates a leaderboard DataFrame with key performance indicators."""
    if df.empty or not selected_groups:
        return pd.DataFrame()

    grouping_col = config["grouping"]["target_column"]
    df_filtered = df[df[grouping_col].isin(selected_groups)]
    if df_filtered.empty:
        return pd.DataFrame()

    primary_metric_name = config["primary_metric"]["name"]
    sort_ascending = config["primary_metric"]["goal"] == "minimize"

    agg_config = {
        "Correct": (primary_metric_name, "mean"),
        "Cost": ("total_cost", "mean"),
        "Duration": ("Duration", "mean"),
        "Avg Total Tokens": ("total_tokens", "mean"),
        "Runs": ("Model", "size"),
    }

    leaderboard = df_filtered.groupby("Model").agg(**agg_config).reset_index()

    return leaderboard.sort_values("Correct", ascending=sort_ascending)


def create_pareto_frontier_plot(
    df: pd.DataFrame, selected_groups: List[str], x_axis_mode: str, config: Dict
) -> go.Figure:
    """Visualizes the trade-off between model performance and cost/token usage."""
    fig = go.Figure()
    plot_config = config["plots"]["pareto"]
    if df.empty or not selected_groups or not plot_config.get("enabled", False):
        return fig.update_layout(title="No data available for selected filters.")

    grouping_col = config["grouping"]["target_column"]
    df_filtered = df[df[grouping_col].isin(selected_groups)]

    primary_metric_name = config["primary_metric"]["name"]
    y_axis_label = config["primary_metric"]["label"]

    model_metrics = (
        df_filtered.groupby("Model")
        .agg(
            y_axis=(primary_metric_name, "mean"),
            total_cost=("total_cost", "mean"),
            total_response_tokens=("total_response_tokens", "mean"),
            color_axis=(plot_config["color_axis"], "mean"),
        )
        .reset_index()
    )

    x_axis_config = plot_config["x_axis_options"][x_axis_mode]
    x_data = model_metrics[x_axis_config["column"]]
    x_title = x_axis_config["label"]
    hover_label = x_axis_config["label"]
    hover_format = ":.4f" if x_axis_mode == "cost" else ":.0f"

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=model_metrics["y_axis"],
            mode="markers+text",
            marker=dict(
                size=18,
                color=model_metrics["color_axis"],
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title=f"Avg {plot_config['color_axis']} (s)"),
            ),
            text=model_metrics["Model"],
            textposition="top center",
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{y_axis_label}: %{{y:.1f}}%<br>"
                f"{hover_label}: %{{x{hover_format}}}<br>"
                f"Avg {plot_config['color_axis']}: %{{marker.color:.1f}}s<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=plot_config["title"].format(x_axis_label=x_title),
        xaxis_title=f"Average {x_title}",
        yaxis_title=y_axis_label,
        showlegend=False,
        height=600,
    )
    return fig


def create_success_rates_plot(
    df: pd.DataFrame, selected_groups: List[str], config: Dict
) -> go.Figure:
    """Compares models across different success metrics."""
    fig = go.Figure()
    plot_config = config["plots"]["success_rates"]
    if df.empty or not selected_groups or not plot_config.get("enabled", False):
        return fig.update_layout(title="No data available for selected filters.")

    grouping_col = config["grouping"]["target_column"]
    df_filtered = df[df[grouping_col].isin(selected_groups)]

    y_prefix = plot_config.get("y_prefix")
    y_columns = plot_config.get("y_columns", [])
    metric_cols = (
        y_columns
        if y_columns
        else [col for col in df_filtered.columns if col.startswith(y_prefix)]
    )

    if not metric_cols:
        return fig.update_layout(title=f"No columns with prefix '{y_prefix}' found.")

    models = sorted(df_filtered["Model"].unique())

    for metric_col in metric_cols:
        metric_name = metric_col.replace(y_prefix, "") if y_prefix else metric_col
        avg_scores = [
            df_filtered[df_filtered["Model"] == model][metric_col].mean() * 100
            for model in models
        ]
        fig.add_trace(go.Bar(name=metric_name, x=models, y=avg_scores))

    fig.update_layout(
        title=plot_config["title"],
        xaxis_title="Model",
        yaxis_title="Success Rate (%)",
        barmode="group",
        height=500,
        legend_title="Metric",
    )
    return fig


def create_failure_analysis_plot(
    df: pd.DataFrame, selected_groups: List[str], config: Dict
) -> go.Figure:
    """Shows common failure reasons by model."""
    fig = go.Figure()
    plot_config = config["plots"]["failure_analysis"]
    if df.empty or not selected_groups or not plot_config.get("enabled", False):
        return fig.update_layout(title="No data available for selected filters.")

    grouping_col = config["grouping"]["target_column"]
    df_filtered = df[df[grouping_col].isin(selected_groups)]
    models = sorted(df_filtered["Model"].unique())

    failure_series = plot_config["series"]
    failure_counts = {series["name"]: [] for series in failure_series}

    for model in models:
        model_data = df_filtered[df_filtered["Model"] == model]
        for series in failure_series:
            # Use pandas.eval to safely evaluate the condition string
            count = model_data.eval(f"`{series['column']}` {series['condition']}").sum()
            failure_counts[series["name"]].append(count)

    for reason, counts in failure_counts.items():
        fig.add_trace(go.Bar(name=reason, x=models, y=counts))

    fig.update_layout(
        title=plot_config["title"],
        xaxis_title="Model",
        yaxis_title="Number of Failures",
        barmode="stack",
        height=500,
        legend_title="Failure Reason",
    )
    return fig


def create_token_breakdown_plot(
    df: pd.DataFrame, selected_groups: List[str], config: Dict
) -> go.Figure:
    """Creates a stacked bar chart showing token breakdown."""
    fig = go.Figure()
    plot_config = config["plots"]["token_breakdown"]
    if df.empty or not selected_groups or not plot_config.get("enabled", False):
        return fig.update_layout(title="No data available for selected filters.")

    grouping_col = config["grouping"]["target_column"]
    df_filtered = df[df[grouping_col].isin(selected_groups)]

    series_config = plot_config["series"]
    agg_dict = {series["name"]: (series["column"], "mean") for series in series_config}
    token_data = df_filtered.groupby("Model").agg(**agg_dict).reset_index()

    for series in series_config:
        fig.add_trace(
            go.Bar(
                name=series["name"], x=token_data["Model"], y=token_data[series["name"]]
            )
        )

    fig.update_layout(
        title=plot_config["title"],
        xaxis_title="Model",
        yaxis_title="Average Tokens",
        barmode="stack",
        height=500,
        legend_title="Token Type",
    )
    return fig


def create_cost_breakdown_plot(
    df: pd.DataFrame, selected_groups: List[str], config: Dict
) -> go.Figure:
    """Creates a stacked bar chart for cost breakdown."""
    fig = go.Figure()
    plot_config = config["plots"]["cost_breakdown"]
    if df.empty or not selected_groups or not plot_config.get("enabled", False):
        return fig.update_layout(title="No data available for selected filters.")

    grouping_col = config["grouping"]["target_column"]
    df_filtered = df[df[grouping_col].isin(selected_groups)]

    series_config = plot_config["series"]
    agg_dict = {series["name"]: (series["column"], "mean") for series in series_config}
    cost_data = df_filtered.groupby("Model").agg(**agg_dict).reset_index()

    for series in series_config:
        fig.add_trace(
            go.Bar(
                name=series["name"], x=cost_data["Model"], y=cost_data[series["name"]]
            )
        )

    fig.update_layout(
        title=plot_config["title"],
        xaxis_title="Model",
        yaxis_title="Average Cost ($)",
        barmode="stack",
        height=500,
        legend_title="Cost Type",
    )
    return fig


def main():
    """Main Streamlit application entrypoint."""
    eval_config = EVAL_CONFIG  # Use the validated config

    st.title(eval_config.title)
    st.subheader("LLM Evaluation Benchmark Dashboard")

    # --- Sidebar Setup ---
    st.sidebar.header("⚙️ Data Configuration")

    # File selection
    default_dir_path = (
        os.path.dirname(detect_csv_files()[0]) if detect_csv_files() else ""
    )
    custom_dir = st.sidebar.text_input("Results Directory:", value=default_dir_path)

    if st.sidebar.button("🔄 Refresh Files"):
        st.rerun()

    csv_files = detect_csv_files(custom_dir)
    if not csv_files:
        st.warning(f"No `*_combined_results.csv` files found in `{custom_dir}`.")
        return

    selected_files = st.sidebar.multiselect(
        "Select result files:",
        options=[os.path.basename(f) for f in csv_files],
        default=[os.path.basename(f) for f in csv_files],
    )

    if not selected_files:
        st.info("Select one or more result files to begin analysis.")
        return

    # --- Data Loading and Filtering ---
    full_file_paths = [os.path.join(custom_dir, f) for f in selected_files]
    df_initial = load_csv_data(full_file_paths)

    if df_initial.empty:
        st.error("No data loaded. Please check the selected files.")
        return

    # Grouping filter
    grouping_config = eval_config.grouping
    st.sidebar.subheader(f"🎯 {grouping_config.label} Filter")

    # Ensure the target column exists before trying to access it
    if grouping_config.target_column not in df_initial.columns:
        df_initial = extract_grouping_column(df_initial, eval_config.model_dump())

    available_groups = sorted(df_initial[grouping_config.target_column].unique())
    selected_groups = st.sidebar.multiselect(
        f"Filter by {grouping_config.label.lower()}:",
        options=available_groups,
        default=available_groups,
    )

    # Cost configuration in sidebar
    st.sidebar.subheader("💰 Cost Configuration")
    cost_file_path = os.path.join(os.path.dirname(__file__), "costs.csv")
    model_costs = load_model_costs(cost_file_path)
    available_models = sorted(df_initial["Model"].unique())

    cost_config = {}
    user_cost_override = {}

    with st.sidebar.expander("Edit Model Costs (per 1M tokens)", expanded=False):
        for model in available_models:
            if model in model_costs:
                cost_config[model] = model_costs[model]
            else:
                st.warning(f"No cost data found for model: {model}. Using zeros.")
                cost_config[model] = {
                    "input": [{"up_to": float("inf"), "price": 0.0}],
                    "output": {"default": [{"up_to": float("inf"), "price": 0.0}]},
                }

        st.markdown("---")
        st.markdown("Override costs below (optional, simplified):")

        for model in available_models:
            cols = st.columns(2)
            default_input = (
                cost_config.get(model, {}).get("input", [{}])[0].get("price", 0.0)
            )
            output_pricing = cost_config.get(model, {}).get("output", {})
            if "default" in output_pricing:
                default_output = output_pricing["default"][0].get("price", 0.0)
            elif "non_thinking" in output_pricing:
                default_output = output_pricing["non_thinking"][0].get("price", 0.0)
            else:
                default_output = 0.0

            input_cost = cols[0].number_input(
                f"{model} Input",
                value=float(default_input),
                step=0.01,
                format="%.4f",
                key=f"{model}_input_cost",
            )
            output_cost = cols[1].number_input(
                f"{model} Output",
                value=float(default_output),
                step=0.01,
                format="%.4f",
                key=f"{model}_output_cost",
            )

            if input_cost != default_input or output_cost != default_output:
                user_cost_override[model] = {
                    "input": [{"up_to": float("inf"), "price": input_cost}],
                    "output": {
                        "default": [{"up_to": float("inf"), "price": output_cost}]
                    },
                }

    # Apply overrides
    final_cost_config = cost_config.copy()
    final_cost_config.update(user_cost_override)

    df = process_data(df_initial, final_cost_config, eval_config)

    # --- Main Panel ---
    st.header("📊 Overview")

    # Key metrics
    cols = st.columns(4)
    cols[0].metric("Evaluation Runs", len(df))
    cols[1].metric("Models Evaluated", df["Model"].nunique())
    cols[2].metric("Test Cases", df[grouping_config.column].nunique())
    cols[3].metric("Files Loaded", len(selected_files))

    st.info(
        f"**Showing results for {grouping_config.label.lower()}:** {', '.join(selected_groups) if selected_groups else 'None'}"
    )

    # --- Leaderboard & Pareto ---
    st.header("🏅 Leaderboard")
    leaderboard_df = create_leaderboard(df, selected_groups, eval_config.model_dump())

    if not leaderboard_df.empty:
        primary_metric_label = eval_config.primary_metric.label
        st.dataframe(
            leaderboard_df,
            column_config={
                "Correct": st.column_config.ProgressColumn(
                    primary_metric_label,
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Cost": st.column_config.ProgressColumn(
                    "Avg Cost ($)",
                    format="$%.4f",
                    min_value=0,
                    max_value=leaderboard_df["Cost"].max(),
                ),
                "Duration": st.column_config.NumberColumn(
                    "Avg Duration (s)", format="%.2fs"
                ),
                "Avg Total Tokens": st.column_config.NumberColumn(
                    "Avg Total Tokens", format="%.0f"
                ),
            },
            use_container_width=True,
        )
    else:
        st.warning("No data available for the current filter selection.")

    st.header("📈 Pareto Frontier Analysis")
    pareto_config = eval_config.plots.pareto
    x_axis_mode = st.radio(
        "Compare performance against:",
        list(pareto_config.x_axis_options.keys()),
        format_func=lambda x: x.capitalize(),
        horizontal=True,
    )
    st.plotly_chart(
        create_pareto_frontier_plot(
            df, selected_groups, x_axis_mode, eval_config.model_dump()
        ),
        use_container_width=True,
    )

    # --- Deep Dive Analysis ---
    with st.expander("🔍 Deep Dive Analysis", expanded=False):
        plot_configs = eval_config.plots

        # Build a dictionary of active plot configs
        active_plots = {
            name: plot
            for name, plot in plot_configs.model_dump().items()
            if plot.get("enabled")
        }

        tabs_to_create = {}
        if "success_rates" in active_plots:
            tabs_to_create["Success Rates"] = True
        if "failure_analysis" in active_plots:
            tabs_to_create["Failure Analysis"] = True
        if "token_breakdown" in active_plots or "cost_breakdown" in active_plots:
            tabs_to_create["Resource Usage"] = True
        tabs_to_create["Raw Data"] = True

        if tabs_to_create:
            tabs = st.tabs(list(tabs_to_create.keys()))
            tab_map = dict(zip(tabs_to_create.keys(), tabs))

            if "Success Rates" in tab_map:
                with tab_map["Success Rates"]:
                    st.plotly_chart(
                        create_success_rates_plot(
                            df, selected_groups, eval_config.model_dump()
                        ),
                        use_container_width=True,
                    )
            if "Failure Analysis" in tab_map:
                with tab_map["Failure Analysis"]:
                    st.plotly_chart(
                        create_failure_analysis_plot(
                            df, selected_groups, eval_config.model_dump()
                        ),
                        use_container_width=True,
                    )
            if "Resource Usage" in tab_map:
                with tab_map["Resource Usage"]:
                    if "token_breakdown" in active_plots:
                        st.plotly_chart(
                            create_token_breakdown_plot(
                                df, selected_groups, eval_config.model_dump()
                            ),
                            use_container_width=True,
                        )
                    if "cost_breakdown" in active_plots:
                        st.plotly_chart(
                            create_cost_breakdown_plot(
                                df, selected_groups, eval_config.model_dump()
                            ),
                            use_container_width=True,
                        )
            if "Raw Data" in tab_map:
                with tab_map["Raw Data"]:
                    st.dataframe(
                        df[
                            df[eval_config.grouping.target_column].isin(selected_groups)
                        ],
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
