import os
import glob
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from typing import List, Dict
import json

st.set_page_config(
    page_title="Merbench - LLM Evaluation Benchmark", page_icon="🏆", layout="wide"
)

DEFAULT_COSTS = {
    "gemini-2.5-pro-preview-06-05": {"input": 3.50, "output": 10.50},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-preview-04-17": {"input": 0.075, "output": 0.30},
    "openai:o4-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4.1-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4.1": {"input": 2.50, "output": 10.00},
}


def find_all_combined_results_csvs(directory_path: str) -> list[str]:
    """
    Finds all '*_combined_results.csv' files in the given directory,
    sorted by modification time (newest first).
    """
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
        script_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        directory = os.path.join(script_dir, "mermaid_eval_results")

    return find_all_combined_results_csvs(directory)


def load_csv_data(file_paths: List[str]) -> pd.DataFrame:
    """Load and combine multiple CSV files."""
    if not file_paths:
        return pd.DataFrame()

    dataframes = []
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
                continue

            df = pd.read_csv(file_path)
            if df.empty:
                st.warning(f"Empty file: {os.path.basename(file_path)}")
                continue

            df["source_file"] = os.path.basename(file_path)
            dataframes.append(df)
        except pd.errors.EmptyDataError:
            st.error(f"Empty or invalid CSV file: {os.path.basename(file_path)}")
        except pd.errors.ParserError as e:
            st.error(f"CSV parsing error in {os.path.basename(file_path)}: {e}")
        except Exception as e:
            st.error(f"Unexpected error loading {os.path.basename(file_path)}: {e}")

    if not dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def extract_case_difficulty(case_name: str) -> str:
    """Extract difficulty level from case name."""
    case_lower = case_name.lower()
    if "easy" in case_lower:
        return "easy"
    elif "medium" in case_lower:
        return "medium"
    elif "hard" in case_lower:
        return "hard"
    else:
        return "unknown"


def parse_metric_details(metric_details_str: str) -> Dict:
    """Parse the metric details JSON string."""
    if pd.isna(metric_details_str) or metric_details_str == "":
        return {}
    try:
        return json.loads(metric_details_str.replace("'", '"'))
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        st.warning(f"Failed to parse metric details: {e}")
        return {}


def calculate_costs(df: pd.DataFrame, cost_config: Dict) -> pd.DataFrame:
    """Calculate costs based on token usage and cost configuration."""
    df_with_costs = df.copy()

    df_with_costs["input_cost"] = 0.0
    df_with_costs["output_cost"] = 0.0
    df_with_costs["total_cost"] = 0.0

    for idx, row in df_with_costs.iterrows():
        model = row.get("Model", "")
        if model in cost_config:
            try:
                input_tokens = max(0, row.get("Metric_request_tokens", 0) or 0)
                input_cost_per_1m = cost_config[model]["input"]
                input_cost = (input_tokens / 1_000_000) * input_cost_per_1m

                response_tokens = max(0, row.get("Metric_response_tokens", 0) or 0)
                thinking_tokens = max(0, row.get("thinking_tokens", 0) or 0)
                total_output_tokens = response_tokens + thinking_tokens
                output_cost_per_1m = cost_config[model]["output"]
                output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m

                df_with_costs.at[idx, "input_cost"] = input_cost
                df_with_costs.at[idx, "output_cost"] = output_cost
                df_with_costs.at[idx, "total_cost"] = input_cost + output_cost
            except (TypeError, ValueError, KeyError) as e:
                st.warning(
                    f"Error calculating costs for model {model} at row {idx}: {e}"
                )
                continue

    return df_with_costs


def process_data(df: pd.DataFrame, cost_config: Dict) -> pd.DataFrame:
    """Process and clean the data for visualization."""
    if df.empty:
        return df

    processed_df = df.copy()

    processed_df["difficulty"] = processed_df["Case"].apply(extract_case_difficulty)

    processed_df["thinking_tokens"] = 0
    processed_df["text_tokens"] = 0

    if "Metric_details" in processed_df.columns:
        for idx, row in processed_df.iterrows():
            details = parse_metric_details(row["Metric_details"])
            processed_df.at[idx, "thinking_tokens"] = details.get("thoughts_tokens", 0)
            processed_df.at[idx, "text_tokens"] = details.get("text_prompt_tokens", 0)

    if "Metric_response_tokens" in processed_df.columns:
        processed_df["total_response_tokens"] = (
            processed_df["Metric_response_tokens"] + processed_df["thinking_tokens"]
        )
    else:
        processed_df["total_response_tokens"] = processed_df["thinking_tokens"]

    score_columns = [col for col in processed_df.columns if col.startswith("Score_")]
    if score_columns:
        processed_df["overall_score"] = processed_df[score_columns].mean(axis=1) * 100
    else:
        processed_df["overall_score"] = 50  # Default neutral score

    processed_df = calculate_costs(processed_df, cost_config)

    return processed_df


def create_leaderboard(df: pd.DataFrame, selected_cases: List[str]) -> pd.DataFrame:
    """Create a leaderboard table with model performance metrics."""
    if df.empty:
        return pd.DataFrame()

    if selected_cases and "all" not in selected_cases:
        df_filtered = df[df["difficulty"].isin(selected_cases)]
    else:
        df_filtered = df

    leaderboard_data = []

    for model in df_filtered["Model"].unique():
        model_data = df_filtered[df_filtered["Model"] == model]

        for difficulty in ["easy", "medium", "hard"]:
            diff_data = model_data[model_data["difficulty"] == difficulty]

            if not diff_data.empty:
                avg_score = diff_data["overall_score"].mean()
                avg_duration = diff_data["Duration"].mean()
                avg_tokens = diff_data["total_response_tokens"].mean()
                avg_cost = diff_data["total_cost"].mean()
                run_count = len(diff_data)

                leaderboard_data.append(
                    {
                        "Model": model,
                        "Difficulty": difficulty,
                        "Avg_Score": avg_score,
                        "Avg_Duration": avg_duration,
                        "Avg_Tokens": avg_tokens,
                        "Avg_Cost": avg_cost,
                        "Run_Count": run_count,
                        "Efficiency": avg_tokens / avg_duration
                        if avg_duration > 0
                        else 0,
                        "Cost_Efficiency": avg_score / avg_cost if avg_cost > 0 else 0,
                    }
                )

    leaderboard_df = pd.DataFrame(leaderboard_data)

    if not leaderboard_df.empty:
        leaderboard_df = leaderboard_df.sort_values("Avg_Score", ascending=False)

    return leaderboard_df


def create_pareto_frontier_plot(
    df: pd.DataFrame, selected_cases: List[str], x_axis_mode: str = "tokens"
) -> go.Figure:
    """Create a Pareto frontier plot showing model performance vs efficiency."""
    if df.empty:
        return go.Figure()

    if selected_cases and "all" not in selected_cases:
        df_filtered = df[df["difficulty"].isin(selected_cases)]
    else:
        df_filtered = df

    model_metrics = (
        df_filtered.groupby("Model")
        .agg(
            {
                "overall_score": "mean",
                "Duration": "mean",
                "total_response_tokens": "mean",
                "thinking_tokens": "mean",
                "total_cost": "mean",
            }
        )
        .reset_index()
    )

    model_metrics["efficiency"] = np.where(
        model_metrics["Duration"] > 0,
        model_metrics["total_response_tokens"] / model_metrics["Duration"],
        0,
    )

    if x_axis_mode == "cost":
        x_data = model_metrics["total_cost"]
        x_title = "Average Total Cost ($)"
        hover_x_label = "Avg Cost"
        hover_x_format = ":.4f"
    else:  # tokens
        x_data = model_metrics["total_response_tokens"]
        x_title = "Average Total Response Tokens"
        hover_x_label = "Avg Tokens"
        hover_x_format = ":.0f"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=model_metrics["overall_score"],
            mode="markers+text",
            marker=dict(
                size=15,
                color=model_metrics["Duration"],
                colorscale="RdYlGn_r",  # Red for slow, green for fast
                showscale=True,
                colorbar=dict(title="Avg Duration (sec)"),
            ),
            text=model_metrics["Model"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>"
            + "Avg Score: %{y:.1f}%<br>"
            + f"{hover_x_label}: %{{x{hover_x_format}}}<br>"
            + "Avg Duration: %{marker.color:.1f}s<br>"
            + "<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Pareto Frontier: Model Performance vs {'Cost' if x_axis_mode == 'cost' else 'Token Usage'}",
        xaxis_title=x_title,
        yaxis_title="Average Overall Score (%)",
        showlegend=False,
        height=600,
    )

    return fig


def create_cost_configuration_ui(models: List[str]) -> Dict:
    """Create UI for configuring model costs."""
    st.sidebar.subheader("💰 Cost Configuration")
    st.sidebar.write("Set costs per 1M tokens:")

    cost_config = {}

    with st.sidebar.expander("Model Costs", expanded=False):
        for model in models:
            st.write(f"**{model}**")

            default_input = DEFAULT_COSTS.get(model, {"input": 1.0, "output": 3.0})[
                "input"
            ]
            default_output = DEFAULT_COSTS.get(model, {"input": 1.0, "output": 3.0})[
                "output"
            ]

            col1, col2 = st.columns(2)
            with col1:
                input_cost = st.number_input(
                    "Input",
                    min_value=0.0,
                    value=default_input,
                    step=0.01,
                    format="%.3f",
                    key=f"input_{model}",
                )
            with col2:
                output_cost = st.number_input(
                    "Output",
                    min_value=0.0,
                    value=default_output,
                    step=0.01,
                    format="%.3f",
                    key=f"output_{model}",
                )

            cost_config[model] = {"input": input_cost, "output": output_cost}

    return cost_config


def create_token_breakdown_plot(
    df: pd.DataFrame, selected_cases: List[str]
) -> go.Figure:
    """Create a stacked bar chart showing token breakdown by model."""
    if df.empty:
        return go.Figure()

    if selected_cases and "all" not in selected_cases:
        df_filtered = df[df["difficulty"].isin(selected_cases)]
    else:
        df_filtered = df

    token_data = (
        df_filtered.groupby("Model")
        .agg(
            {
                "Metric_request_tokens": "mean",
                "Metric_response_tokens": "mean",
                "thinking_tokens": "mean",
            }
        )
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Request Tokens",
            x=token_data["Model"],
            y=token_data["Metric_request_tokens"],
            marker_color="lightblue",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Response Tokens",
            x=token_data["Model"],
            y=token_data["Metric_response_tokens"],
            marker_color="orange",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Thinking Tokens",
            x=token_data["Model"],
            y=token_data["thinking_tokens"],
            marker_color="lightgreen",
        )
    )

    fig.update_layout(
        title="Token Usage Breakdown by Model",
        xaxis_title="Model",
        yaxis_title="Average Tokens",
        barmode="stack",
        height=500,
    )

    return fig


def create_success_rates_plot(df: pd.DataFrame, selected_cases: List[str]) -> go.Figure:
    """Create a bar chart showing overall average success rates by metric."""
    if df.empty:
        return go.Figure()

    if selected_cases and "all" not in selected_cases:
        df_filtered = df[df["difficulty"].isin(selected_cases)]
    else:
        df_filtered = df

    metric_columns = [col for col in df_filtered.columns if col.startswith("Score_")]

    if not metric_columns:
        return go.Figure()

    models = sorted(df_filtered["Model"].unique())

    metric_data = []
    for metric_col in metric_columns:
        metric_name = metric_col.replace("Score_", "")
        model_scores = []
        for model in models:
            model_data = df_filtered[df_filtered["Model"] == model]
            if not model_data.empty:
                avg_score = model_data[metric_col].mean() * 100
                model_scores.append(avg_score)
            else:
                model_scores.append(0)

        metric_data.append(
            {"metric": metric_name, "models": models, "scores": model_scores}
        )

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, data in enumerate(metric_data):
        fig.add_trace(
            go.Bar(
                name=data["metric"],
                x=data["models"],
                y=data["scores"],
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        title="Overall Average Success Rates by Metric",
        xaxis_title="Model",
        yaxis_title="Success Rate (%)",
        barmode="group",
        height=500,
        legend=dict(title="Metric"),
    )

    return fig


def main():
    """Main Streamlit application."""

    st.title("🏆 Merbench")
    st.subheader("LLM Evaluation Benchmark Dashboard")

    st.sidebar.header("Configuration")

    st.sidebar.subheader("Detected CSV Files")

    script_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    default_dir = os.path.join(script_dir, "mermaid_eval_results")

    custom_dir = st.sidebar.text_input(
        "Directory path:",
        value=default_dir,
        help="Path to directory containing *_combined_results.csv files",
    )

    if "refresh_files" not in st.session_state:
        st.session_state.refresh_files = False

    if st.sidebar.button("🔄 Refresh Detected Files"):
        st.session_state.refresh_files = True
        st.rerun()

    csv_files = detect_csv_files(custom_dir)

    if not csv_files:
        st.warning(f"No *_combined_results.csv files found in {custom_dir}")
        st.info("Looking for files matching pattern: *_combined_results.csv")
        return

    st.sidebar.write(f"Found {len(csv_files)} files (sorted by newest first):")

    selected_files = []
    for file in csv_files:
        file_name = os.path.basename(file)
        if st.sidebar.checkbox(file_name, value=True, key=f"file_{file_name}"):
            selected_files.append(file)

    if not selected_files:
        st.warning("Please select at least one CSV file to visualize.")
        return

    with st.spinner("Loading data..."):
        df_initial = load_csv_data(selected_files)
        if df_initial.empty:
            st.error("No data could be loaded from the selected files.")
            return

    available_models = sorted(df_initial["Model"].unique())
    cost_config = create_cost_configuration_ui(available_models)

    df = process_data(df_initial, cost_config)

    available_cases = sorted(df["difficulty"].unique())
    st.sidebar.subheader("Case Selection")

    case_options = ["all"] + available_cases
    selected_cases = st.sidebar.multiselect(
        "Select difficulty levels", options=case_options, default=["all"]
    )

    st.subheader("📊 Data Overview")

    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        st.metric("Evaluation Runs", len(df))
        st.metric("Files Loaded", len(selected_files))

    with overview_col2:
        st.metric("Models", len(df["Model"].unique()))
        st.metric("Cases", len(df["Case"].unique()))

    st.write(f"**Models:** {', '.join(sorted(df['Model'].unique()))}")
    st.write(f"**Cases:** {', '.join(sorted(df['Case'].unique()))}")

    if "all" not in selected_cases:
        st.info(f"🎯 **Filtered by difficulty:** {', '.join(selected_cases)}")
    else:
        st.info("🎯 **Showing all difficulty levels**")

    st.header("🏅 Leaderboard")
    leaderboard_df = create_leaderboard(df, selected_cases)

    if not leaderboard_df.empty:
        display_df = leaderboard_df.copy()
        display_df["Avg_Score"] = display_df["Avg_Score"].round(1)
        display_df["Avg_Duration"] = display_df["Avg_Duration"].round(2)
        display_df["Avg_Tokens"] = display_df["Avg_Tokens"].round(0)
        display_df["Avg_Cost"] = display_df["Avg_Cost"].round(4)
        display_df["Efficiency"] = display_df["Efficiency"].round(1)
        display_df["Cost_Efficiency"] = display_df["Cost_Efficiency"].round(1)

        st.dataframe(
            display_df,
            column_config={
                "Avg_Score": st.column_config.NumberColumn(
                    "Avg Score (%)", format="%.1f"
                ),
                "Avg_Duration": st.column_config.NumberColumn(
                    "Avg Duration (s)", format="%.2f"
                ),
                "Avg_Tokens": st.column_config.NumberColumn(
                    "Avg Tokens", format="%.0f"
                ),
                "Avg_Cost": st.column_config.NumberColumn(
                    "Avg Cost ($)", format="%.4f"
                ),
                "Efficiency": st.column_config.NumberColumn(
                    "Tokens/sec", format="%.1f"
                ),
                "Cost_Efficiency": st.column_config.NumberColumn(
                    "Score/Cost", format="%.1f"
                ),
                "Run_Count": st.column_config.NumberColumn("Runs", format="%d"),
            },
            use_container_width=True,
        )
    else:
        st.warning("No data available for the selected filters.")

    st.header("📈 Pareto Frontier Analysis")

    x_axis_mode = st.radio(
        "X-Axis Mode:",
        options=["tokens", "cost"],
        format_func=lambda x: "Token Count" if x == "tokens" else "Cost ($)",
        horizontal=True,
    )

    pareto_fig = create_pareto_frontier_plot(df, selected_cases, x_axis_mode)
    st.plotly_chart(pareto_fig, use_container_width=True)

    st.header("📊 Overall Average Success Rates by Metric")
    success_fig = create_success_rates_plot(df, selected_cases)
    st.plotly_chart(success_fig, use_container_width=True)

    st.header("🔢 Token Usage Analysis")
    token_fig = create_token_breakdown_plot(df, selected_cases)
    st.plotly_chart(token_fig, use_container_width=True)

    with st.expander("📋 Raw Data"):
        if selected_cases and "all" not in selected_cases:
            filtered_df = df[df["difficulty"].isin(selected_cases)]
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True)


if __name__ == "__main__":
    main()
