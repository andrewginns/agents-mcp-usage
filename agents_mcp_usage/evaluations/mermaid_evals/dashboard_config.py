"""
Dashboard Configuration for the Generic Evaluation UI

This file defines the "personality" of the Streamlit dashboard. By modifying this
configuration, you can adapt the UI to display results from any evaluation
that produces a CSV file with a compatible format.

Each section of the configuration is documented to explain its purpose and
the available options.
"""

# ==============================================================================
# MERBENCH DASHBOARD CONFIGURATION
# ==============================================================================
# This is an example configuration for the "Merbench" evaluation.
# You can duplicate and modify this structure to create configurations
# for other evaluations.
# ==============================================================================

MERBENCH_CONFIG = {
    # --- General Dashboard Settings ---
    "title": "🧜‍♀️ Merbench - LLM Evaluation ",
    "description": (
        "Getting LLMs to consistently nail the Mermaid diagram syntax can be... an adventure. "
        "\n\nMerbench evaluates an LLM's ability to autonomously write and debug Mermaid syntax. The agent can access "
        "an MCP server that validates its code and provides error feedback, guiding it towards a correct solution."
        "\n\nEach model is tested across three difficulty levels, with a limited number of five attempts per test case. "
        "Performance is measured by the final success rate, averaged over complete runs, **reflecting both an understanding of Mermaid syntax and effective tool usage.**"
    ),
    "icon": "🧜‍♀️",  # Emoji for the browser tab
    # --- Primary Metric Configuration ---
    # The primary metric is the main score used for the leaderboard and
    # the y-axis of the Pareto frontier plot.
    "primary_metric": {
        "name": "correctness_score",  # The column name in the CSV
        "label": "Avg. Success Rate",  # How the metric is displayed in the UI
        "goal": "maximize",  # 'maximize' or 'minimize'
        "score_column": "Score_MermaidDiagramValid",
    },
    # --- Grouping and Filtering ---
    # Defines how to group test cases (e.g., by difficulty).
    "grouping": {
        "column": "Case",  # The column containing the case names
        "label": "Difficulty",  # The label for the filter in the sidebar
        # A regex to extract group names from the 'column'. The first
        # capture group will be used as the group name.
        "extractor_regex": r".*_(easy|medium|hard)",
        "target_column": "difficulty",
    },
    # --- Plot and Analysis Tab Configuration ---
    # This section defines which plots are displayed in the UI.
    "plots": {
        "pareto": {
            "enabled": True,
            "title": "Performance vs. {x_axis_label}",
            "y_axis": "primary_metric",  # Uses the primary_metric defined above
            "x_axis_options": {
                "cost": {"column": "total_cost", "label": "Cost"},
                "tokens": {"column": "total_response_tokens", "label": "Tokens"},
                "duration": {"column": "Duration", "label": "Duration"},
            },
            "color_axis": "Duration",  # Column to use for the color scale
        },
        "success_rates": {
            "enabled": True,
            "title": "Success Rate by Metric",
            "type": "grouped_bar",
            # Finds all columns starting with this prefix to create a bar for each.
            "y_prefix": "Score_",
        },
        "failure_analysis": {
            "enabled": True,
            "title": "Failure Analysis by Reason",
            "type": "stacked_bar",
            # Defines the series for the stacked bar chart.
            # Each item represents a condition that counts as a "failure".
            "series": [
                {
                    "name": "Invalid Diagram",
                    "column": "Score_MermaidDiagramValid",
                    "condition": "== 0",
                },
                {
                    "name": "MCP Tool Failure",
                    "column": "Score_UsedBothMCPTools",
                    "condition": "< 1",
                },
                {
                    "name": "Usage Limit Exceeded",
                    "column": "Score_UsageLimitNotExceeded",
                    "condition": "== 0",
                },
            ],
        },
        "token_breakdown": {
            "enabled": True,
            "title": "Average Token Usage by Type",
            "type": "stacked_bar",
            "series": [
                {"name": "Request", "column": "Metric_request_tokens"},
                {"name": "Response", "column": "Metric_response_tokens"},
                {"name": "Thinking", "column": "thinking_tokens"},
            ],
        },
        "cost_breakdown": {
            "enabled": True,
            "title": "Average Cost Breakdown by Type",
            "type": "stacked_bar",
            "series": [
                {"name": "Input Cost", "column": "input_cost"},
                {"name": "Output Cost", "column": "output_cost"},
            ],
        },
    },
    # --- Cost Calculation ---
    # Defines which columns are used to calculate the total cost.
    "cost_calculation": {
        "input_token_cols": ["Metric_request_tokens"],
        "output_token_cols": ["Metric_response_tokens", "thinking_tokens"],
    },
}

# --- Add other configurations for different evaluations below ---
# EXAMPLE_OTHER_EVAL_CONFIG = { ... }

# The default configuration to use when the dashboard starts.
# You can change this to point to a different configuration.
DEFAULT_CONFIG = MERBENCH_CONFIG