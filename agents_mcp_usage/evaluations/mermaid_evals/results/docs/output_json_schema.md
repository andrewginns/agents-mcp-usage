# Merbench Output JSON Schema Documentation

## Overview

The preprocessed JSON output from `preprocess_merbench_data.py` contains structured evaluation data for LLM models tested on Mermaid diagram generation tasks. This document describes the schema and how to build visualizations with filtering capabilities.

## Filtering System

The visualization system supports three filter dimensions that work as logical AND conditions:

1. **Test Group Filter**: `easy`, `medium`, `hard`
2. **Provider Filter**: `Google`, `Amazon`, `OpenAI`, `Other`
3. **Model Filter**: Individual model selection (e.g., `gemini-2.5-pro`, `bedrock:us.amazon.nova-premier-v1:0`)

When filters are applied, only data matching ALL selected criteria is included in visualizations.

## JSON Schema Structure

### 1. Stats Object
```json
{
  "stats": {
    "total_runs": 180,
    "models_evaluated": 12,
    "test_cases": 3,
    "test_groups": ["easy", "medium", "hard"],
    "providers": ["Google", "Amazon", "OpenAI"],
    "models": ["model1", "model2", ...],
    "total_cost": 11.899786,
    "avg_cost_per_run": 0.066110
  }
}
```

**Usage**: Display overall evaluation metrics. Update these counts when filters are applied by recalculating from filtered raw_data.

### 2. Leaderboard Array
```json
{
  "leaderboard": [
    {
      "Model": "gemini-2.5-pro-preview-06-05",
      "Success_Rate": 40.0,
      "Avg_Duration": 46.887859,
      "Avg_Tokens": 8693.733333,
      "Avg_Cost": 0.045514,
      "Avg_Input_Cost": 0.005918,
      "Avg_Output_Cost": 0.039596,
      "Runs": 15,
      "Provider": "Google"
    }
  ]
}
```

**Usage**: Main leaderboard visualization. When filters are applied:
1. Filter `raw_data` based on selected criteria
2. Recalculate aggregates (group by Model)
3. Sort by Success_Rate descending

**Visualization**: Table or bar chart showing model rankings.

### 3. Pareto Data Array
```json
{
  "pareto_data": [
    {
      "Model": "bedrock:us.amazon.nova-lite-v1:0",
      "Success_Rate": 4.0,
      "Duration": 14.524447,
      "total_tokens": 2350.2,
      "total_cost": 0.000080,
      "input_cost": 0.000034,
      "output_cost": 0.000046,
      "Metric_request_tokens": 1337.8,
      "Metric_response_tokens": 1012.4
    }
  ]
}
```

**Usage**: Create Pareto frontier plots showing trade-offs between:
- Success Rate vs Cost
- Success Rate vs Duration
- Success Rate vs Tokens

**Filtering**: Recalculate from filtered raw_data, then plot.

### 4. Test Groups Data Array
```json
{
  "test_groups_data": [
    {
      "Model": "bedrock:us.amazon.nova-lite-v1:0",
      "test_group": "easy",
      "Score_MermaidDiagramValid": 0.2,
      "Score_UsageLimitNotExceeded": 1.0,
      "Score_UsedBothMCPTools": 1.0,
      "total_cost": 0.000114,
      "input_cost": 0.000049,
      "output_cost": 0.000066,
      "total_tokens": 3377.2
    }
  ]
}
```

**Usage**: Create grouped bar charts showing performance by difficulty level.

**Filtering**: 
- Filter by Model and Provider
- Group remaining data by test_group
- Show comparison across difficulty levels

### 5. Failure Analysis Data Array
```json
{
  "failure_analysis_data": [
    {
      "Model": "bedrock:us.amazon.nova-lite-v1:0",
      "Invalid Diagram": 13,
      "MCP Tool Failure": 0,
      "Usage Limit Exceeded": 0
    }
  ]
}
```

**Usage**: Stacked bar chart showing failure reasons by model.

**Filtering**: When filters are applied, recalculate failure counts from filtered raw_data.

### 6. Cost Breakdown Data Array
```json
{
  "cost_breakdown_data": [
    {
      "Model": "bedrock:us.amazon.nova-lite-v1:0",
      "test_group": "easy",
      "avg_total_cost": 0.000114,
      "sum_total_cost": 0.00057,
      "run_count": 5,
      "avg_input_cost": 0.000049,
      "sum_input_cost": 0.000243,
      "avg_output_cost": 0.000066,
      "sum_output_cost": 0.000328
    }
  ]
}
```

**Usage**: Detailed cost analysis by model and test difficulty.

**Visualizations**:
- Stacked bar chart of input vs output costs
- Heatmap of costs by model × difficulty
- Cost comparison across test groups

### 7. Raw Data Array
```json
{
  "raw_data": [
    {
      "Model": "gemini-2.5-pro-preview-05-06",
      "Case": "simple_easy",
      "test_group": "easy",
      "Duration": 18.090633,
      "Score_MermaidDiagramValid": 0,
      "Score_UsageLimitNotExceeded": 0,
      "Score_UsedBothMCPTools": 1,
      "total_tokens": 127419,
      "provider": "Google",
      "Metric_request_tokens": 25483,
      "Metric_response_tokens": 101936,
      "total_cost": 0.0,
      "input_cost": 0.0,
      "output_cost": 0.0
    }
  ]
}
```

**Usage**: Source data for all filtering and aggregation operations.

**Key fields for filtering**:
- `Model`: For model filter
- `provider`: For provider filter  
- `test_group`: For difficulty filter
- `Score_UsageLimitNotExceeded`: When 0, cost is 0 (failed tests)

## Implementing Filters

### Filter Logic (Pseudo-code)
```javascript
function applyFilters(rawData, filters) {
  return rawData.filter(row => {
    // All conditions must be true (AND logic)
    const modelMatch = !filters.models.length || 
                      filters.models.includes(row.Model);
    const providerMatch = !filters.providers.length || 
                         filters.providers.includes(row.provider);
    const testGroupMatch = !filters.testGroups.length || 
                          filters.testGroups.includes(row.test_group);
    
    return modelMatch && providerMatch && testGroupMatch;
  });
}
```

### Updating Visualizations After Filtering

1. **Filter raw_data** using the AND logic
2. **Recalculate aggregates** from filtered data:
   - Leaderboard: Group by Model, calculate means
   - Test groups: Group by Model + test_group
   - Failure analysis: Count failures per type
   - Cost breakdown: Sum and average costs
3. **Update visualizations** with new data

## Visualization Examples

### 1. Cost vs Performance Scatter Plot
- X-axis: Average cost (from pareto_data)
- Y-axis: Success rate (from pareto_data)
- Color: Provider
- Size: Number of runs
- Filters affect which models appear

### 2. Performance by Difficulty Grouped Bar Chart
- X-axis: Models
- Y-axis: Success rate
- Groups: easy, medium, hard (different bars)
- Filters reduce which models/groups are shown

### 3. Cost Breakdown Stacked Bar
- X-axis: Models
- Y-axis: Cost
- Stack: Input cost vs Output cost
- Facet by: test_group (optional)

### 4. Failure Analysis Heatmap
- Rows: Models
- Columns: Failure types
- Values: Failure counts
- Color intensity: Number of failures

## Configuration Object
```json
{
  "config": {
    "title": "🧜‍♀️ Merbench - LLM Evaluation",
    "description": "...",
    "primary_metric": {
      "name": "Success_Rate",
      "label": "Success Rate (%)"
    }
  }
}
```

**Usage**: UI configuration and metric definitions for consistent labeling.

## Best Practices

1. **Always filter on raw_data first**, then recalculate aggregates
2. **Cache filtered results** to avoid recalculation on every interaction
3. **Show filter status** in UI (e.g., "Showing 5 of 12 models")
4. **Handle empty results** gracefully when filters exclude all data
5. **Cost considerations**: Remember that failed tests (Score_UsageLimitNotExceeded = 0) have $0 cost

## Example Filter Combinations

1. **"Show only Amazon models on hard tests"**
   - Provider filter: ["Amazon"]
   - Test group filter: ["hard"]
   - Result: Only Amazon model performance on hard difficulty

2. **"Compare Google models across all difficulties"**
   - Provider filter: ["Google"]
   - Test group filter: [] (empty = all)
   - Result: All Google models, all difficulties

3. **"Show specific model performance"**
   - Model filter: ["gemini-2.5-pro"]
   - Result: Single model data across all test groups 