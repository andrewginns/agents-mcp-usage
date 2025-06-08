import os
import glob
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from typing import List, Dict
import json

# Page configuration
st.set_page_config(
    page_title="Merbench - LLM Evaluation Benchmark", page_icon="🏆", layout="wide"
)

# Default model costs (per 1M tokens)
DEFAULT_COSTS = {
    "gemini-2.5-pro-preview-06-05": {"input": 3.50, "output": 10.50},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-preview-04-17": {"input": 0.075, "output": 0.30},
    "openai:o4-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4.1-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4.1": {"input": 2.50, "output": 10.00},
}

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
            st.error(f"An unexpected error occurred while loading {os.path.basename(file_path)}: {e}")

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def extract_case_difficulty(case_name: str) -> str:
    """Extracts difficulty ('easy', 'medium', 'hard') from a case name."""
    case_lower = str(case_name).lower()
    if "easy" in case_lower:
        return "easy"
    if "medium" in case_lower:
        return "medium"
    if "hard" in case_lower:
        return "hard"
    return "unknown"


def parse_metric_details(metric_details_str: str) -> Dict:
    """Safely parses a JSON string from the 'Metric_details' column."""
    if pd.isna(metric_details_str) or not metric_details_str:
        return {}
    try:
        # Attempt to fix common JSON issues like single quotes
        return json.loads(metric_details_str.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return {}


def calculate_costs(df: pd.DataFrame, cost_config: Dict) -> pd.DataFrame:
    """Calculates input, output, and total costs for each run."""
    df_with_costs = df.copy()
    df_with_costs["input_cost"] = 0.0
    df_with_costs["output_cost"] = 0.0
    df_with_costs["total_cost"] = 0.0

    for idx, row in df_with_costs.iterrows():
        model = row.get("Model")
        if model in cost_config:
            try:
                input_tokens = row.get("Metric_request_tokens", 0) or 0
                output_tokens = row.get("Metric_response_tokens", 0) or 0
                thinking_tokens = row.get("thinking_tokens", 0) or 0

                input_cost = (input_tokens / 1_000_000) * cost_config[model]["input"]
                output_cost = ((output_tokens + thinking_tokens) / 1_000_000) * cost_config[model]["output"]
                
                df_with_costs.at[idx, "input_cost"] = input_cost
                df_with_costs.at[idx, "output_cost"] = output_cost
                df_with_costs.at[idx, "total_cost"] = input_cost + output_cost
            except (TypeError, KeyError) as e:
                st.warning(f"Cost calculation error for model {model} at row {idx}: {e}")
    return df_with_costs


def process_data(df: pd.DataFrame, cost_config: Dict) -> pd.DataFrame:
    """Main data processing pipeline."""
    if df.empty:
        return df

    processed_df = df.copy()
    processed_df["difficulty"] = processed_df["Case"].apply(extract_case_difficulty)
    
    # Extract token counts from metric details
    metric_details = processed_df["Metric_details"].apply(parse_metric_details)
    processed_df["thinking_tokens"] = metric_details.apply(lambda x: x.get("thoughts_tokens", 0))
    processed_df["text_tokens"] = metric_details.apply(lambda x: x.get("text_prompt_tokens", 0))

    # Calculate total response tokens
    processed_df["total_response_tokens"] = (
        processed_df.get("Metric_response_tokens", 0) + processed_df["thinking_tokens"]
    )
    
    # Standardize correctness score
    processed_df["correctness_score"] = processed_df.get("Score_MermaidDiagramValid", 0.5) * 100

    return calculate_costs(processed_df, cost_config)


# --- UI & Plotting ---

def create_leaderboard(df: pd.DataFrame, selected_cases: List[str]) -> pd.DataFrame:
    """Creates a leaderboard DataFrame with key performance indicators."""
    if df.empty or not selected_cases:
        return pd.DataFrame()

    df_filtered = df[df["difficulty"].isin(selected_cases)]
    if df_filtered.empty:
        return pd.DataFrame()

    leaderboard = df_filtered.groupby("Model").agg(
        Correct=("correctness_score", "mean"),
        Cost=("total_cost", "mean"),
        Duration=("Duration", "mean"),
        Tokens=("total_response_tokens", "mean"),
        Runs=("Model", "size"),
    ).reset_index()

    return leaderboard.sort_values("Correct", ascending=False)



def create_pareto_frontier_plot(df: pd.DataFrame, selected_cases: List[str], x_axis_mode: str) -> go.Figure:
    """Visualizes the trade-off between model performance and cost/token usage."""
    fig = go.Figure()
    if df.empty or not selected_cases:
        return fig.update_layout(title="No data available for selected filters.")

    df_filtered = df[df["difficulty"].isin(selected_cases)]
    model_metrics = df_filtered.groupby("Model").agg(
        correctness_score=("correctness_score", "mean"),
        total_cost=("total_cost", "mean"),
        total_response_tokens=("total_response_tokens", "mean"),
        Duration=("Duration", "mean"),
    ).reset_index()

    x_data, x_title, hover_label, hover_format = (
        (model_metrics["total_cost"], "Average Total Cost ($)", "Avg Cost", ":.4f")
        if x_axis_mode == "cost"
        else (model_metrics["total_response_tokens"], "Average Total Response Tokens", "Avg Tokens", ":.0f")
    )

    fig.add_trace(go.Scatter(
        x=x_data,
        y=model_metrics["correctness_score"],
        mode="markers+text",
        marker=dict(
            size=18,
            color=model_metrics["Duration"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Avg Duration (s)"),
        ),
        text=model_metrics["Model"],
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Avg Score: %{y:.1f}%<br>"
            f"{hover_label}: %{{x{hover_format}}}<br>"
            "Avg Duration: %{marker.color:.1f}s<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title=f"Performance vs. {'Cost' if x_axis_mode == 'cost' else 'Tokens'}",
        xaxis_title=x_title,
        yaxis_title="Average Success Rate (%)",
        showlegend=False,
        height=600,
    )
    return fig


def create_success_rates_plot(df: pd.DataFrame, selected_cases: List[str]) -> go.Figure:
    """Compares models across different success metrics."""
    fig = go.Figure()
    if df.empty or not selected_cases:
        return fig.update_layout(title="No data available for selected filters.")

    df_filtered = df[df["difficulty"].isin(selected_cases)]
    metric_cols = [col for col in df_filtered.columns if col.startswith("Score_")]
    if not metric_cols:
        return fig.update_layout(title="No 'Score_' columns found.")

    models = sorted(df_filtered["Model"].unique())
    
    for metric_col in metric_cols:
        metric_name = metric_col.replace("Score_", "")
        avg_scores = [
            df_filtered[df_filtered["Model"] == model][metric_col].mean() * 100
            for model in models
        ]
        fig.add_trace(go.Bar(name=metric_name, x=models, y=avg_scores))

    fig.update_layout(
        title="Success Rate by Metric",
        xaxis_title="Model",
        yaxis_title="Success Rate (%)",
        barmode="group",
        height=500,
        legend_title="Metric",
    )
    return fig


def create_failure_analysis_plot(df: pd.DataFrame, selected_cases: List[str]) -> go.Figure:
    """Shows common failure reasons by model."""
    fig = go.Figure()
    if df.empty or not selected_cases:
        return fig.update_layout(title="No data available for selected filters.")

    df_filtered = df[df["difficulty"].isin(selected_cases)]
    models = sorted(df_filtered["Model"].unique())
    
    failure_counts = {
        "Invalid Diagram": [],
        "MCP Tool Failure": [],
        "Usage Limit Exceeded": [],
    }

    for model in models:
        model_data = df_filtered[df_filtered["Model"] == model]
        failure_counts["Invalid Diagram"].append((model_data["Score_MermaidDiagramValid"] == 0).sum())
        failure_counts["MCP Tool Failure"].append((model_data["Score_UsedBothMCPTools"] < 1).sum())
        failure_counts["Usage Limit Exceeded"].append((model_data["Score_UsageLimitNotExceeded"] == 0).sum())

    for reason, counts in failure_counts.items():
        fig.add_trace(go.Bar(name=reason, x=models, y=counts))

    fig.update_layout(
        title="Failure Analysis by Reason",
        xaxis_title="Model",
        yaxis_title="Number of Failures",
        barmode="stack",
        height=500,
        legend_title="Failure Reason",
    )
    return fig


def create_token_breakdown_plot(df: pd.DataFrame, selected_cases: List[str]) -> go.Figure:
    """Creates a stacked bar chart showing token breakdown."""
    fig = go.Figure()
    if df.empty or not selected_cases:
        return fig.update_layout(title="No data available for selected filters.")

    df_filtered = df[df["difficulty"].isin(selected_cases)]
    token_data = df_filtered.groupby("Model").agg(
        Request=("Metric_request_tokens", "mean"),
        Response=("Metric_response_tokens", "mean"),
        Thinking=("thinking_tokens", "mean"),
    ).reset_index()

    fig.add_trace(go.Bar(name="Request", x=token_data["Model"], y=token_data["Request"]))
    fig.add_trace(go.Bar(name="Response", x=token_data["Model"], y=token_data["Response"]))
    fig.add_trace(go.Bar(name="Thinking", x=token_data["Model"], y=token_data["Thinking"]))

    fig.update_layout(
        title="Average Token Usage by Type",
        xaxis_title="Model",
        yaxis_title="Average Tokens",
        barmode="stack",
        height=500,
        legend_title="Token Type",
    )
    return fig


def create_cost_breakdown_plot(df: pd.DataFrame, selected_cases: List[str]) -> go.Figure:
    """Creates a stacked bar chart for cost breakdown."""
    fig = go.Figure()
    if df.empty or not selected_cases:
        return fig.update_layout(title="No data available for selected filters.")

    df_filtered = df[df["difficulty"].isin(selected_cases)]
    cost_data = df_filtered.groupby("Model").agg(
        Input=("input_cost", "mean"),
        Output=("output_cost", "mean"),
    ).reset_index()

    fig.add_trace(go.Bar(name="Input Cost", x=cost_data["Model"], y=cost_data["Input"]))
    fig.add_trace(go.Bar(name="Output Cost", x=cost_data["Model"], y=cost_data["Output"]))

    fig.update_layout(
        title="Average Cost Breakdown by Type",
        xaxis_title="Model",
        yaxis_title="Average Cost ($)",
        barmode="stack",
        height=500,
        legend_title="Cost Type",
    )
    return fig


def main():
    """Main Streamlit application entrypoint."""
    st.title("🏆 Merbench")
    st.subheader("LLM Evaluation Benchmark Dashboard")

    # --- Sidebar Setup ---
    st.sidebar.header("⚙️ Configuration")
    
    # File selection
    default_dir_path = os.path.dirname(detect_csv_files()[0]) if detect_csv_files() else ""
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

    available_models = sorted(df_initial["Model"].unique())
    
    # Cost configuration in sidebar
    st.sidebar.subheader("💰 Cost Configuration")
    cost_config = {}
    with st.sidebar.expander("Edit Model Costs (per 1M tokens)", expanded=False):
        for model in available_models:
            cols = st.columns(2)
            default = DEFAULT_COSTS.get(model, {"input": 0.0, "output": 0.0})
            input_cost = cols[0].number_input(f"{model} Input", value=float(default["input"]), step=0.01, format="%.2f")
            output_cost = cols[1].number_input(f"{model} Output", value=float(default["output"]), step=0.01, format="%.2f")
            cost_config[model] = {"input": input_cost, "output": output_cost}

    df = process_data(df_initial, cost_config)

    # Difficulty filter
    st.sidebar.subheader("🎯 Difficulty Filter")
    available_difficulties = sorted(df["difficulty"].unique())
    selected_difficulties = st.sidebar.multiselect(
        "Filter by difficulty:",
        options=available_difficulties,
        default=available_difficulties,
    )

    # --- Main Panel ---
    st.header("📊 Overview")
    
    # Key metrics
    cols = st.columns(4)
    cols[0].metric("Evaluation Runs", len(df))
    cols[1].metric("Models Evaluated", df["Model"].nunique())
    cols[2].metric("Test Cases", df["Case"].nunique())
    cols[3].metric("Files Loaded", len(selected_files))

    st.info(f"**Showing results for difficulties:** {', '.join(selected_difficulties) if selected_difficulties else 'None'}")

    # --- Leaderboard & Pareto ---
    st.header("🏅 Leaderboard")
    leaderboard_df = create_leaderboard(df, selected_difficulties)
    
    if not leaderboard_df.empty:
        st.markdown("""
        <style>
            /* Target the 'Correct' column's progress bar (2nd column) */
            .stDataFrame [data-testid="stDataFrameApp"] div:nth-child(2) [data-testid="stProgress"] > div > div > div > div {
                background-color: #28a745; /* Green */
            }
            /* Target the 'Cost' column's progress bar (3rd column) */
            .stDataFrame [data-testid="stDataFrameApp"] div:nth-child(3) [data-testid="stProgress"] > div > div > div > div {
                background-color: #007bff; /* Blue */
            }
        </style>
        """, unsafe_allow_html=True)
        st.dataframe(
            leaderboard_df,
            column_config={
                "Correct": st.column_config.ProgressColumn(
                    "Avg. Success",
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
                "Duration": st.column_config.NumberColumn("Avg Duration (s)", format="%.2fs"),
                "Tokens": st.column_config.NumberColumn("Avg Tokens", format="%.0f"),
            },
            column_order=["Model", "Correct", "Cost", "Duration", "Tokens", "Runs"],
            use_container_width=True,
        )
    else:
        st.warning("No data available for the current filter selection.")

    st.header("📈 Pareto Frontier Analysis")
    x_axis_mode = st.radio("Compare performance against:", ["cost", "tokens"], format_func=lambda x: x.capitalize(), horizontal=True)
    st.plotly_chart(create_pareto_frontier_plot(df, selected_difficulties, x_axis_mode), use_container_width=True)

    # --- Deep Dive Analysis ---
    with st.expander("🔍 Deep Dive Analysis", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Success Rates", "📉 Failure Analysis", "⚙️ Resource Usage", "📋 Raw Data"])

        with tab1:
            st.plotly_chart(create_success_rates_plot(df, selected_difficulties), use_container_width=True)
        with tab2:
            st.plotly_chart(create_failure_analysis_plot(df, selected_difficulties), use_container_width=True)
        with tab3:
            st.plotly_chart(create_token_breakdown_plot(df, selected_difficulties), use_container_width=True)
            st.plotly_chart(create_cost_breakdown_plot(df, selected_difficulties), use_container_width=True)
        with tab4:
            st.dataframe(df[df["difficulty"].isin(selected_difficulties)], use_container_width=True)


if __name__ == "__main__":
    main()
