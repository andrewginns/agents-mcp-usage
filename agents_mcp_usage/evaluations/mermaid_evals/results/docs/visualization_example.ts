/**
 * Example: Creating a Cost vs Performance Scatter Plot with Filtering (TypeScript)
 * This example demonstrates how to use the JSON output data with filters
 */

// Type definitions for the data structures

interface RawDataRow {
  Model: string;
  Case: string;
  test_group: 'easy' | 'medium' | 'hard';
  Duration: number;
  Score_MermaidDiagramValid: 0 | 1;
  Score_UsageLimitNotExceeded: 0 | 1;
  Score_UsedBothMCPTools: 0 | 1;
  total_tokens: number;
  provider: 'Google' | 'Amazon' | 'OpenAI' | 'Other';
  Metric_request_tokens: number;
  Metric_response_tokens: number;
  total_cost: number;
  input_cost: number;
  output_cost: number;
}

interface LeaderboardEntry {
  Model: string;
  Success_Rate: number;
  Avg_Duration: number;
  Avg_Tokens: number;
  Avg_Cost: number;
  Avg_Input_Cost: number;
  Avg_Output_Cost: number;
  Runs: number;
  Provider: string;
}

interface ParetoDataEntry {
  Model: string;
  Success_Rate: number;
  Duration: number;
  total_tokens: number;
  total_cost: number;
  input_cost: number;
  output_cost: number;
  Metric_request_tokens: number;
  Metric_response_tokens: number;
}

interface TestGroupDataEntry {
  Model: string;
  test_group: 'easy' | 'medium' | 'hard';
  Score_MermaidDiagramValid: number;
  Score_UsageLimitNotExceeded: number;
  Score_UsedBothMCPTools: number;
  total_cost: number;
  input_cost: number;
  output_cost: number;
  total_tokens: number;
}

interface FailureAnalysisEntry {
  Model: string;
  'Invalid Diagram': number;
  'MCP Tool Failure': number;
  'Usage Limit Exceeded': number;
}

interface CostBreakdownEntry {
  Model: string;
  test_group: 'easy' | 'medium' | 'hard';
  avg_total_cost: number;
  sum_total_cost: number;
  run_count: number;
  avg_input_cost: number;
  sum_input_cost: number;
  avg_output_cost: number;
  sum_output_cost: number;
}

interface Stats {
  total_runs: number;
  models_evaluated: number;
  test_cases: number;
  test_groups: Array<'easy' | 'medium' | 'hard'>;
  providers: Array<'Google' | 'Amazon' | 'OpenAI' | 'Other'>;
  models: string[];
  total_cost: number;
  avg_cost_per_run: number;
}

interface Config {
  title: string;
  description: string;
  primary_metric: {
    name: string;
    label: string;
  };
}

interface JsonData {
  stats: Stats;
  leaderboard: LeaderboardEntry[];
  pareto_data: ParetoDataEntry[];
  test_groups_data: TestGroupDataEntry[];
  failure_analysis_data: FailureAnalysisEntry[];
  cost_breakdown_data: CostBreakdownEntry[];
  raw_data: RawDataRow[];
  config: Config;
}

interface Filters {
  models: string[];
  providers: Array<'Google' | 'Amazon' | 'OpenAI' | 'Other'>;
  testGroups: Array<'easy' | 'medium' | 'hard'>;
}

interface AggregatedModel {
  model: string;
  provider: string;
  avgCost: number;
  avgTokens: number;
  successRate: number;
  avgDuration: number;
  runCount: number;
}

interface ModelGroup {
  model: string;
  provider: string;
  runs: number;
  totalCost: number;
  totalTokens: number;
  successCount: number;
  totalDuration: number;
}

interface PlotlyTrace {
  x: number[];
  y: number[];
  text: string[];
  marker: {
    size: number[];
  };
  mode: string;
  type: string;
  name: string;
  textposition: string;
}

interface PlotlyLayout {
  title: string;
  xaxis: {
    title: string;
    type?: string;
    tickformat?: string;
  };
  yaxis: {
    title: string;
    range?: number[];
  };
  hovermode: string;
  showlegend: boolean;
  annotations?: Array<{
    x: number;
    y: number;
    xref: string;
    yref: string;
    text: string;
    showarrow: boolean;
    font: {
      size: number;
      color: string;
    };
  }>;
}

interface StatsDisplay {
  totalRuns: number;
  modelsShown: number;
  totalCost: number;
  avgCostPerRun: number;
}

// Sample filter state
const filters: Filters = {
  models: [],  // Empty array means "all models"
  providers: ["Google", "Amazon"],  // Only Google and Amazon models
  testGroups: ["easy", "medium"]    // Exclude "hard" tests
};

// Function to apply filters to raw data
function applyFilters(rawData: RawDataRow[], filters: Filters): RawDataRow[] {
  return rawData.filter((row: RawDataRow): boolean => {
    // Check each filter condition (AND logic)
    const modelMatch: boolean = filters.models.length === 0 || 
                               filters.models.includes(row.Model);
    
    const providerMatch: boolean = filters.providers.length === 0 || 
                                  filters.providers.includes(row.provider);
    
    const testGroupMatch: boolean = filters.testGroups.length === 0 || 
                                   filters.testGroups.includes(row.test_group);
    
    return modelMatch && providerMatch && testGroupMatch;
  });
}

// Function to calculate aggregated data from filtered raw data
function calculateAggregates(filteredData: RawDataRow[]): AggregatedModel[] {
  // Group by model
  const modelGroups: Record<string, ModelGroup> = {};
  
  filteredData.forEach((row: RawDataRow): void => {
    if (!modelGroups[row.Model]) {
      modelGroups[row.Model] = {
        model: row.Model,
        provider: row.provider,
        runs: 0,
        totalCost: 0,
        totalTokens: 0,
        successCount: 0,
        totalDuration: 0
      };
    }
    
    const group: ModelGroup = modelGroups[row.Model];
    group.runs++;
    group.totalCost += row.total_cost;
    group.totalTokens += row.total_tokens;
    group.successCount += row.Score_MermaidDiagramValid;
    group.totalDuration += row.Duration;
  });
  
  // Calculate averages
  return Object.values(modelGroups).map((group: ModelGroup): AggregatedModel => ({
    model: group.model,
    provider: group.provider,
    avgCost: group.totalCost / group.runs,
    avgTokens: group.totalTokens / group.runs,
    successRate: (group.successCount / group.runs) * 100,
    avgDuration: group.totalDuration / group.runs,
    runCount: group.runs
  }));
}

// Declare Plotly for global access
declare const Plotly: {
  newPlot(elementId: string, traces: PlotlyTrace[], layout: PlotlyLayout): void;
};

// Function to create Plotly scatter plot
function createCostVsPerformancePlot(jsonData: JsonData, filters: Filters): void {
  // Step 1: Apply filters to raw data
  const filteredRawData: RawDataRow[] = applyFilters(jsonData.raw_data, filters);
  
  // Step 2: Calculate aggregates
  const aggregatedData: AggregatedModel[] = calculateAggregates(filteredRawData);
  
  // Step 3: Prepare data for Plotly
  const traces: Record<string, PlotlyTrace> = {};
  
  // Group by provider for different colors
  aggregatedData.forEach((model: AggregatedModel): void => {
    if (!traces[model.provider]) {
      traces[model.provider] = {
        x: [],
        y: [],
        text: [],
        marker: { size: [] },
        mode: 'markers+text',
        type: 'scatter',
        name: model.provider,
        textposition: 'top center'
      };
    }
    
    traces[model.provider].x.push(model.avgCost);
    traces[model.provider].y.push(model.successRate);
    traces[model.provider].text.push(model.model);
    traces[model.provider].marker.size.push(Math.sqrt(model.runCount) * 10);
  });
  
  // Step 4: Create the plot
  const layout: PlotlyLayout = {
    title: 'Model Performance vs Cost (Filtered)',
    xaxis: {
      title: 'Average Cost ($)',
      type: 'log',  // Log scale for better cost distribution
      tickformat: '$.4f'
    },
    yaxis: {
      title: 'Success Rate (%)',
      range: [0, 105]
    },
    hovermode: 'closest',
    showlegend: true,
    annotations: [{
      x: 0.5,
      y: -0.15,
      xref: 'paper',
      yref: 'paper',
      text: `Filters: ${filters.providers.join(', ')} providers | ${filters.testGroups.join(', ')} tests`,
      showarrow: false,
      font: { size: 12, color: 'gray' }
    }]
  };
  
  Plotly.newPlot('cost-performance-plot', Object.values(traces), layout);
}

// Function to update summary statistics
function updateStats(jsonData: JsonData, filteredRawData: RawDataRow[]): void {
  const stats: StatsDisplay = {
    totalRuns: filteredRawData.length,
    modelsShown: new Set(filteredRawData.map((r: RawDataRow) => r.Model)).size,
    totalCost: filteredRawData.reduce((sum: number, r: RawDataRow) => sum + r.total_cost, 0),
    avgCostPerRun: 0
  };
  
  stats.avgCostPerRun = stats.totalCost / stats.totalRuns || 0;
  
  // Update UI elements (with null checks)
  const totalRunsElement = document.getElementById('stat-total-runs');
  const modelsElement = document.getElementById('stat-models');
  const totalCostElement = document.getElementById('stat-total-cost');
  const avgCostElement = document.getElementById('stat-avg-cost');
  
  if (totalRunsElement) {
    totalRunsElement.textContent = stats.totalRuns.toString();
  }
  
  if (modelsElement) {
    modelsElement.textContent = `${stats.modelsShown} of ${jsonData.stats.models_evaluated}`;
  }
  
  if (totalCostElement) {
    totalCostElement.textContent = `$${stats.totalCost.toFixed(2)}`;
  }
  
  if (avgCostElement) {
    avgCostElement.textContent = `$${stats.avgCostPerRun.toFixed(4)}`;
  }
}

// Helper function to get selected values from multi-select element
function getSelectedValues(element: HTMLSelectElement): string[] {
  return Array.from(element.selectedOptions).map((option: HTMLOptionElement) => option.value);
}

// Main function to initialize visualization
async function initializeVisualization(): Promise<void> {
  try {
    // Load the JSON data
    const response: Response = await fetch('path/to/processed_results.json');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const jsonData: JsonData = await response.json();
    
    // Apply initial filters and create plot
    createCostVsPerformancePlot(jsonData, filters);
    
    // Update statistics
    const filteredData: RawDataRow[] = applyFilters(jsonData.raw_data, filters);
    updateStats(jsonData, filteredData);
    
    // Set up filter change handlers
    const providerFilterElement = document.getElementById('provider-filter') as HTMLSelectElement;
    const testGroupFilterElement = document.getElementById('test-group-filter') as HTMLSelectElement;
    
    if (providerFilterElement) {
      providerFilterElement.addEventListener('change', (e: Event): void => {
        const target = e.target as HTMLSelectElement;
        filters.providers = getSelectedValues(target) as Array<'Google' | 'Amazon' | 'OpenAI' | 'Other'>;
        createCostVsPerformancePlot(jsonData, filters);
        updateStats(jsonData, applyFilters(jsonData.raw_data, filters));
      });
    }
    
    if (testGroupFilterElement) {
      testGroupFilterElement.addEventListener('change', (e: Event): void => {
        const target = e.target as HTMLSelectElement;
        filters.testGroups = getSelectedValues(target) as Array<'easy' | 'medium' | 'hard'>;
        createCostVsPerformancePlot(jsonData, filters);
        updateStats(jsonData, applyFilters(jsonData.raw_data, filters));
      });
    }
    
    // Set up model filter if it exists
    const modelFilterElement = document.getElementById('model-filter') as HTMLSelectElement;
    if (modelFilterElement) {
      modelFilterElement.addEventListener('change', (e: Event): void => {
        const target = e.target as HTMLSelectElement;
        filters.models = getSelectedValues(target);
        createCostVsPerformancePlot(jsonData, filters);
        updateStats(jsonData, applyFilters(jsonData.raw_data, filters));
      });
    }
    
  } catch (error) {
    console.error('Error initializing visualization:', error);
    
    // Show error message to user
    const errorElement = document.getElementById('error-message');
    if (errorElement) {
      errorElement.textContent = `Failed to load data: ${error instanceof Error ? error.message : 'Unknown error'}`;
      errorElement.style.display = 'block';
    }
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initializeVisualization);

// Export types for use in other modules
export type {
  RawDataRow,
  JsonData,
  Filters,
  AggregatedModel,
  LeaderboardEntry,
  ParetoDataEntry,
  TestGroupDataEntry,
  FailureAnalysisEntry,
  CostBreakdownEntry,
  Stats,
  Config
}; 