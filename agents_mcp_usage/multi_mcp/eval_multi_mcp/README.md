# Multi-MCP Mermaid Diagram Evaluation System

This directory contains evaluation modules for testing LLM agents on mermaid diagram fixing tasks using multiple MCP (Model Context Protocol) servers. The system evaluates how well language models can fix invalid mermaid diagrams while utilizing multiple external tools.

## Overview

The evaluation system consists of two main components:

1. **`evals_pydantic_mcp.py`** - Core evaluation module for single-model testing
2. **`run_multi_evals.py`** - Multi-model evaluation runner with parallel execution

## Evaluation Task

The system tests LLM agents on their ability to:
- Fix syntactically invalid mermaid diagrams
- Use both MCP servers (example server for time, mermaid validator for validation)
- Handle errors gracefully with proper categorization
- Provide meaningful failure reasons for debugging

### Test Cases

The evaluation includes three test cases of increasing difficulty:
1. **Easy** - Simple syntax errors in mermaid diagrams
2. **Medium** - More complex structural issues
3. **Hard** - Advanced mermaid syntax problems

## Output Schema

### MermaidOutput

The main output schema captures comprehensive information about each evaluation:

```python
class MermaidOutput(BaseModel):
    fixed_diagram: str           # The corrected mermaid diagram
    failure_reason: str = ""     # Why the case failed (if applicable)
    metrics: Dict[str, Any] = {} # Usage metrics from the LLM
    tools_used: List[str] = []   # Which MCP tools were called
```

### Metrics Captured

The system automatically captures detailed usage metrics:

- **`requests`** - Number of API requests made
- **`request_tokens`** - Total tokens in requests
- **`response_tokens`** - Total tokens in responses
- **`total_tokens`** - Sum of request and response tokens
- **`details`** - Additional model-specific usage details

## Failure Reasons

The system provides meaningful failure categorization for debugging and analysis:

### Agent-Level Failures (from `fix_mermaid_diagram`)

- **`usage_limit_exceeded`** - Agent hit configured usage limits
- **`response_validation_failed`** - Agent response failed Pydantic validation
- **`agent_timeout`** - Agent operation timed out
- **`http_error_{status_code}`** - HTTP errors (e.g., `http_error_502`, `http_error_503`)
- **`timeout_error`** - General timeout errors
- **`connection_error`** - Network/connection issues
- **`rate_limit_error`** - API rate limiting or quota exceeded
- **`error_{ExceptionType}`** - Other specific exceptions (fallback)

### Evaluation-Level Failures (from `run_multi_evals`)

- **`evaluation_timeout`** - Entire evaluation run timed out
- **`evaluation_validation_failed`** - Evaluation framework validation error
- **`model_api_error`** - Model API-specific errors
- **`network_error`** - Network-related evaluation failures
- **`evaluation_error_{ExceptionType}`** - Other evaluation framework errors

## Evaluators

The system uses five different evaluators to assess performance:

### 1. UsedBothMCPTools
- **Score**: 0.0, 0.5, or 1.0
- **Purpose**: Checks if the agent used tools from both MCP servers
- **Scoring**:
  - 1.0: Used both example server and mermaid validator
  - 0.5: Used only one MCP server
  - 0.0: Used no MCP tools or only non-MCP tools

### 2. UsageLimitNotExceeded
- **Score**: 0.0 or 1.0
- **Purpose**: Detects if the case failed due to usage limits
- **Scoring**:
  - 1.0: No usage limit failure
  - 0.0: Failed due to `usage_limit_exceeded`

### 3. MermaidDiagramValid
- **Score**: 0.0 or 1.0
- **Purpose**: Validates the fixed diagram using the mermaid validator MCP server
- **Features**:
  - Skips validation if there was a prior failure
  - Strips markdown formatting and backticks
  - Uses retry logic for transient validation errors
- **Scoring**:
  - 1.0: Diagram passes mermaid syntax validation
  - 0.0: Diagram is invalid or validation failed

### 4. LLMJudge (Format Check)
- **Score**: 0.0 to 1.0 (continuous)
- **Purpose**: Evaluates if response contains only a mermaid diagram
- **Rubric**: "The response only contains a mermaid diagram inside the fixed_diagram field, no other text"

### 5. LLMJudge (Structure Check)  
- **Score**: 0.0 to 1.0 (continuous)
- **Purpose**: Evaluates if the fixed diagram maintains original structure and intent
- **Rubric**: "The fixed_diagram field should maintain the same overall structure and intent as the expected output diagram while fixing any syntax errors"

## Retry Logic

The system includes robust retry logic for handling transient API failures:

### Retryable Errors
- HTTP status codes: 429, 500, 502, 503, 504
- Connection errors and network issues
- General `OSError` exceptions

### Retry Configuration
- **Max attempts**: 3
- **Base delay**: 1 second
- **Exponential backoff**: 1s → 2s → 4s
- **Max delay**: 30 seconds
- **Jitter**: ±50% randomization to prevent thundering herd

### Non-Retryable Errors
- HTTP 4xx errors (except 429)
- Validation errors
- Authentication errors

## CSV Output Format

Results are exported to CSV files with the following columns:

### Basic Information
- **Model** - LLM model used
- **Run** - Run number (for multi-run evaluations)
- **Case** - Test case name (easy/medium/hard)
- **Duration** - Task execution time in seconds
- **Fixed_Diagram_Length** - Length of the output diagram
- **Failure_Reason** - Categorized failure reason (if any)
- **Tools_Used** - Pipe-separated list of MCP tools used

### Evaluator Scores
- **Score_UsedBothMCPTools** - MCP tool usage score
- **Score_UsageLimitNotExceeded** - Usage limit check score
- **Score_MermaidDiagramValid** - Diagram validity score
- **Score_LLMJudge** - Format evaluation scores (2 columns)

### Metrics
- **Metric_requests** - Number of API requests
- **Metric_request_tokens** - Input token count
- **Metric_response_tokens** - Output token count
- **Metric_total_tokens** - Total token usage
- **Metric_details** - Additional usage details

## Usage

### Single Model Evaluation

```bash
# Run evaluation with default model
uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/evals_pydantic_mcp.py

# Customize model and judge
AGENT_MODEL="gemini-2.5-pro" JUDGE_MODEL="gemini-2.0-flash" \
uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/evals_pydantic_mcp.py
```

### Multi-Model Evaluation

```bash
# Run evaluation across multiple models
uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/run_multi_evals.py \
  --models "gemini-2.5-pro,gemini-2.0-flash" \
  --runs 5 \
  --parallel \
  --timeout 600 \
  --output-dir ./results

# Sequential execution with custom judge
uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/run_multi_evals.py \
  --models "gemini-2.5-pro,claude-3-opus" \
  --runs 3 \
  --sequential \
  --judge-model "gemini-2.5-pro" \
  --output-dir ./eval_results
```

### Available Options

- **`--models`** - Comma-separated list of models to evaluate
- **`--runs`** - Number of evaluation runs per model (default: 3)
- **`--judge-model`** - Model for LLM judging (default: gemini-2.5-pro-preview-06-05)
- **`--parallel`** - Run evaluations in parallel (default: true)
- **`--sequential`** - Force sequential execution
- **`--timeout`** - Timeout in seconds per evaluation run (default: 600)
- **`--output-dir`** - Directory to save results (default: ./mermaid_eval_results)

## MCP Servers

The evaluation uses two MCP servers:

1. **Example Server** (`mcp_servers/example_server.py`)
   - Provides time-related tools
   - Used to add timestamps to diagrams

2. **Mermaid Validator** (`mcp_servers/mermaid_validator.py`)
   - Validates mermaid diagram syntax
   - Returns validation results with error details

## Output Files

### Single Model
- `YYYY-MM-DD_HH-MM-SS_mermaid_results_{model}.csv`

### Multi-Model
- `YYYY-MM-DD_HH-MM-SS_individual_{model}.csv` - Per-model results
- `YYYY-MM-DD_HH-MM-SS_combined_results.csv` - All models combined

## Logging and Monitoring

The system integrates with Logfire for comprehensive monitoring:

- **Agent operations** - MCP server interactions, tool usage
- **Retry attempts** - Failure reasons, backoff delays
- **Evaluation progress** - Success rates, timing metrics
- **Error categorization** - Detailed failure analysis

## Error Handling Best Practices

The system implements robust error handling:

1. **Graceful degradation** - Partial results rather than complete failure
2. **Meaningful categorization** - Specific failure reasons for debugging
3. **Retry logic** - Automatic recovery from transient issues
4. **Comprehensive logging** - Full context for error analysis
5. **Resource cleanup** - Proper MCP server lifecycle management

## Dependencies

- **pydantic-ai** - Core agent framework with MCP support
- **pydantic-evals** - Evaluation framework and metrics
- **logfire** - Logging and monitoring
- **rich** - Console output and progress bars
- **asyncio** - Asynchronous evaluation execution 