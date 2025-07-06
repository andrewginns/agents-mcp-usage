# Merbench Visualization Documentation

This directory contains comprehensive documentation for understanding and using the Merbench evaluation output data to build visualizations.

## Documentation Files

### 1. [output_json_schema.md](./output_json_schema.md)
**Complete schema reference for the preprocessed JSON output**
- Detailed description of all data sections
- Filter system explanation (Model, Provider, Test Group)
- Implementation guidelines for AND logic filtering
- Visualization examples and best practices

### 2. [visualization_example.ts](./visualization_example.ts)
**Practical TypeScript implementation example**
- Complete working example of a cost vs performance scatter plot
- Filter application with real code
- Statistics update implementation
- Event handler setup for interactive filtering

### 3. [data_relationships_quickref.md](./data_relationships_quickref.md)
**Quick reference for data relationships and common operations**
- Key relationships between data sections
- Common query patterns
- Performance optimization tips
- Data validation checks

## Key Concepts

### Data Flow
```
CSV Input → preprocess_merbench_data.py → JSON Output → Visualizations
                           ↓
                    Cost Calculations
                    (from costs.json)
```

### Filter Types (AND Logic)
1. **Model Filter**: Select specific models
2. **Provider Filter**: Google, Amazon, OpenAI, Other
3. **Test Group Filter**: easy, medium, hard

### Primary Data Sections
- **raw_data**: Source for all filtering and aggregation
- **leaderboard**: Pre-aggregated model rankings
- **pareto_data**: Performance vs efficiency metrics
- **test_groups_data**: Performance by difficulty
- **cost_breakdown_data**: Detailed cost analysis
- **failure_analysis_data**: Failure reason counts

## Quick Start

1. **Load the JSON data**
   ```javascript
   const data = await fetch('processed_results.json').then(r => r.json());
   ```

2. **Apply filters to raw_data**
   ```javascript
   const filtered = data.raw_data.filter(row => 
     row.provider === "Google" && row.test_group === "easy"
   );
   ```

3. **Recalculate aggregates**
   ```javascript
   const modelStats = {};
   filtered.forEach(row => {
     if (!modelStats[row.Model]) {
       modelStats[row.Model] = {runs: 0, success: 0, cost: 0};
     }
     modelStats[row.Model].runs++;
     modelStats[row.Model].success += row.Score_MermaidDiagramValid;
     modelStats[row.Model].cost += row.total_cost;
   });
   ```

4. **Create visualizations**
   - Use pre-aggregated data for initial views
   - Recalculate from filtered raw_data when filters change
   - Update all related visualizations together

## Cost Calculation Notes

- Costs are calculated per token using tiered pricing from `costs.json`
- Failed tests (Score_UsageLimitNotExceeded = 0) have $0 cost
- Input and output costs are tracked separately
- Thinking tokens may have different pricing than regular output tokens

## Visualization Types

1. **Leaderboard Table**: Model rankings by success rate
2. **Pareto Scatter Plot**: Performance vs cost/duration/tokens
3. **Grouped Bar Charts**: Performance by test difficulty
4. **Stacked Bar Charts**: Failure reasons, cost breakdown
5. **Heatmaps**: Model × difficulty performance matrix

## Tips for Developers

- Always start filtering from `raw_data`
- Cache filter results for performance
- Use the `provider` field for color coding
- Show active filters in the UI
- Handle empty filter results gracefully
- Consider log scale for cost axes due to wide ranges 