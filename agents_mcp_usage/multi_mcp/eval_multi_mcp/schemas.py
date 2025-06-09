"""
Pydantic Schemas for Dashboard Configuration Validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any


class PrimaryMetricConfig(BaseModel):
    name: str = Field(..., description="The column name for the primary metric.")
    label: str = Field(..., description="How the metric is displayed in the UI.")
    goal: str = Field(..., description="'maximize' or 'minimize'")
    score_column: Optional[str] = Field(
        None,
        description="Optional source column to calculate the primary metric from if it doesn't exist.",
    )

    @validator("goal")
    def goal_must_be_max_or_min(cls, v: str) -> str:
        """Validates that the goal is either 'maximize' or 'minimize'."""
        if v not in ["maximize", "minimize"]:
            raise ValueError("goal must be 'maximize' or 'minimize'")
        return v


class GroupingConfig(BaseModel):
    column: str = Field(..., description="The column containing the case names.")
    label: str = Field(..., description="The label for the filter in the sidebar.")
    extractor_regex: str = Field(
        ...,
        description="A regex to extract group names. The first capture group will be used.",
    )
    target_column: str = Field(
        ..., description="The name of the new column to store the extracted group."
    )


class ParetoAxisOption(BaseModel):
    column: str
    label: str


class ParetoPlotConfig(BaseModel):
    enabled: bool
    title: str
    y_axis: str
    x_axis_options: Dict[str, ParetoAxisOption]
    color_axis: str


class BarPlotSeries(BaseModel):
    name: str
    column: str


class StackedBarPlotSeries(BarPlotSeries):
    condition: Optional[str] = None


class BarPlotConfig(BaseModel):
    enabled: bool
    title: str
    type: str
    y_prefix: Optional[str] = None
    y_columns: Optional[List[str]] = None
    series: Optional[List[StackedBarPlotSeries]] = None

    @validator("y_columns", always=True)
    def check_prefix_or_columns(
        cls, v: Optional[List[str]], values: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Validates that either 'y_prefix' or 'y_columns' is provided for grouped_bar plots."""
        if not values.get("y_prefix") and not v:
            if values.get("type") == "grouped_bar":
                raise ValueError(
                    "Either 'y_prefix' or 'y_columns' must be provided for grouped_bar plots."
                )
        return v


class PlotConfig(BaseModel):
    pareto: ParetoPlotConfig
    success_rates: BarPlotConfig
    failure_analysis: BarPlotConfig
    token_breakdown: BarPlotConfig
    cost_breakdown: BarPlotConfig


class CostCalculationConfig(BaseModel):
    input_token_cols: List[str]
    output_token_cols: List[str]


class DashboardConfig(BaseModel):
    title: str
    icon: str
    primary_metric: PrimaryMetricConfig
    grouping: GroupingConfig
    plots: PlotConfig
    cost_calculation: CostCalculationConfig
