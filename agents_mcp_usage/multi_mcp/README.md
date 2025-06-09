# Multi-MCP Usage & Evaluation Suite

This directory contains examples demonstrating the integration of tools from multiple Model Context Protocol (MCP) servers with various LLM agent frameworks, along with a comprehensive evaluation and benchmarking system.

Agents utilising multiple MCP servers can be dramatically more complex than an Agent using a single server. This is because as the number of servers grow the number of tools that the Agent must reason on when and how to use increases. As a result this component not only demonstrates an Agent's use of multiple MCP servers, but also includes a production-ready evaluation suite to validate performance, analyse costs, and compare models across multiple difficulty levels.



## Quickstart

1. Configure `.env` and API keys following instructions in the [README.md](README.md)

3. Ensure the Python MCP servers can be used:
   - The Python MCP servers (example_server.py and mermaid_validator.py) are included in the repository

4. Run an example script:
   ```bash
   # Run the Pydantic-AI multi-MCP example
   uv run agents_mcp_usage/multi_mcp/multi_mcp_use/pydantic_mcp.py
   
   # Run the Google ADK multi-MCP example
   uv run agents_mcp_usage/multi_mcp/multi_mcp_use/adk_mcp.py
   
   # Run the multi-MCP evaluation
   uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/evals_pydantic_mcp.py
   
   # Run multi-model benchmarking
   uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/run_multi_evals.py --models "gemini-2.5-pro,gemini-2.0-flash" --runs 5 --parallel
   
   # Launch the evaluation dashboard
   uv run streamlit run agents_mcp_usage/multi_mcp/eval_multi_mcp/merbench_ui.py
   ```

5. Check the console output, Logfire, or dashboard for results.


### Multi-MCP Architecture

```mermaid
graph LR
    User((User)) --> |"Run script<br>(e.g., pydantic_mcp.py)"| Agent

    subgraph "Agent Framework"
        Agent["Pydantic-AI/ADK Agent<br>(pydantic_mcp.py/adk_mcp.py)"]
    end

    subgraph "MCP Servers"
        PythonMCP["Python MCP Server<br>(mcp_servers/example_server.py)"]
        MermaidMCP["Python Mermaid MCP Server<br>(mcp_servers/mermaid_validator.py)"]
        
        Tools["Tools<br>- add(a, b)<br>- get_current_time()"]
        Resources["Resources<br>- greeting://{name}"]
        MermaidValidator["Mermaid Diagram<br>Validation Tools"]
        
        PythonMCP --- Tools
        PythonMCP --- Resources
        MermaidMCP --- MermaidValidator
    end

    subgraph "LLM Providers"
        LLMs["PydanticAI/Gemini LLM call"]
    end
    
    Logfire[("Logfire<br>Tracing")]
    
    Agent --> PythonMCP
    Agent --> MermaidMCP
    
    PythonMCP --> LLMs
    MermaidMCP --> LLMs
    
    Agent --> Logfire
    
    LLM_Response[("Response")] --> User
    LLMs --> LLM_Response

    %% Node styling for better dark/light mode readability
    classDef userNode fill:#b3e0ff,stroke:#0066cc,stroke-width:2px,color:#000000;
    classDef agentNode fill:#d1c4e9,stroke:#673ab7,stroke-width:2px,color:#000000;
    classDef mcpNode fill:#ffccbc,stroke:#ff5722,stroke-width:2px,color:#000000;
    classDef pythonMcpNode fill:#ffccbc,stroke:#ff5722,stroke-width:2px,color:#000000;
    classDef nodeMcpNode fill:#ffe0b2,stroke:#ff9800,stroke-width:2px,color:#000000;
    classDef toolNode fill:#dcedc8,stroke:#8bc34a,stroke-width:1px,color:#000000;
    classDef llmNode fill:#c8e6c9,stroke:#4caf50,stroke-width:1px,color:#000000;
    classDef outputNode fill:#ffcdd2,stroke:#e53935,stroke-width:2px,color:#000000;
    classDef logNode fill:#e1bee7,stroke:#8e24aa,stroke-width:2px,color:#000000;

    %% Apply styles to nodes
    class User userNode;
    class Agent agentNode;
    class PythonMCP pythonMcpNode;
    class MermaidMCP nodeMcpNode;
    class Tools,Resources,MermaidValidator toolNode;
    class LLMs llmNode;
    class LLM_Response outputNode;
    class Logfire logNode;
```

This diagram illustrates how an agent can leverage multiple specialised MCP servers simultaneously, each providing distinct tools and resources.

### Multi-MCP Sequence Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as Pydantic-AI/ADK Agent
    participant PyMCP as Python MCP Server
    participant MermaidMCP as Python Mermaid MCP Server
    participant LLM as LLM Provider
    participant PyTools as Python Tools
    participant MermaidTools as Mermaid Validator
    participant Logfire as Logfire Tracing
    
    Note over User,Logfire: Multi-MCP Interaction Flow
    
    User->>Agent: Run script with query
    
    activate Agent
    Agent->>Logfire: Start tracing session
    
    par Connect to multiple MCP servers
        Agent->>PyMCP: Initialise connection
        activate PyMCP
        PyMCP-->>Agent: Connection established
        
        Agent->>MermaidMCP: Initialise connection
        activate MermaidMCP
        MermaidMCP-->>Agent: Connection established
    end
    
    Agent->>LLM: Process user query
    activate LLM
    
    loop Tool Selection & Execution
        alt Python MCP Tools Needed
            LLM-->>Agent: Need Python tool
            Agent->>PyMCP: Execute tool (e.g., add, get_time)
            PyMCP->>PyTools: Call tool function
            activate PyTools
            PyTools-->>PyMCP: Return result
            deactivate PyTools
            PyMCP-->>Agent: Tool result
            Agent->>LLM: Continue with tool result
        else Mermaid MCP Tools Needed
            LLM-->>Agent: Need Mermaid validation
            Agent->>MermaidMCP: Validate Mermaid diagram
            MermaidMCP->>MermaidTools: Process diagram
            activate MermaidTools
            MermaidTools-->>MermaidMCP: Validation result
            deactivate MermaidTools
            MermaidMCP-->>Agent: Tool result
            Agent->>LLM: Continue with tool result
        end
    end
    
    LLM-->>Agent: Final response
    deactivate LLM
    
    Agent->>Logfire: Log completion
    
    par Close MCP connections
        Agent->>PyMCP: Close connection
        deactivate PyMCP
        Agent->>MermaidMCP: Close connection
        deactivate MermaidMCP
    end
    
    Agent->>User: Display final answer
    deactivate Agent
    
    Note over User,Logfire: End of interaction
```

The sequence diagram shows how the agent coordinates between multiple specialised MCP servers. It highlights the parallel connection establishment, selective tool usage based on need, and proper connection management.

## Agent Evaluations

Research in LLM agent development has identified tool overload as a significant challenge for agent performance. When faced with too many tools, agents often struggle with:

1. **Tool Selection Complexity**: Determining which tool from which server is most appropriate for a given subtask becomes exponentially more difficult as the number of available tools increases.

2. **Context Management**: Maintaining awareness of which server provides which capabilities and how to properly format requests for each server adds cognitive load to the agent.

3. **Error Recovery**: When tool usage fails, diagnosing whether the issue stems from incorrect tool selection, improper input formatting, or server-specific limitations becomes more challenging with multiple servers.

4. **Reasoning Overhead**: The agent must dedicate more of its context window and reasoning capacity to tool management rather than task completion.

The evaluation framework included in this component is essential for validating that agents can effectively navigate the increased complexity of multiple MCP servers. By measuring success against specific evaluation criteria, developers can ensure that the benefits of tool specialisation outweigh the potential pitfalls of tool overload.

## Comprehensive Evaluation Suite

The `eval_multi_mcp/` directory contains a production-ready evaluation system for benchmarking LLM agent performance across multiple frameworks and models. The suite tests agents on mermaid diagram correction tasks using multiple MCP servers, providing rich metrics and analysis capabilities.

### Evaluation Components

#### Core Modules
- **`evals_pydantic_mcp.py`** - Single-model evaluation with comprehensive metrics collection
- **`run_multi_evals.py`** - Multi-model parallel/sequential benchmarking with CSV export
- **`merbench_ui.py`** - Interactive Streamlit dashboard for visualisation and analysis
- **`dashboard_config.py`** - Configuration-driven UI setup for flexible dashboard customisation
- **`costs.csv`** - Pricing integration for cost analysis and budget planning

#### Test Difficulty Levels
The evaluation includes three test cases of increasing complexity:
1. **Easy** - Simple syntax errors in mermaid diagrams
2. **Medium** - More complex structural issues requiring deeper reasoning
3. **Hard** - Advanced mermaid syntax problems testing sophisticated tool usage

#### Evaluation Metrics
The system captures five key performance indicators:
- **UsedBothMCPTools** - Validates proper coordination between multiple MCP servers
- **UsageLimitNotExceeded** - Monitors resource consumption and efficiency
- **MermaidDiagramValid** - Assesses technical correctness of outputs
- **LLMJudge (Format)** - Evaluates response formatting and structure
- **LLMJudge (Structure)** - Measures preservation of original diagram intent

## Interactive Dashboard & Visualisation

The Streamlit-based dashboard (`merbench_ui.py`) provides comprehensive analysis and comparison capabilities:

### Dashboard Features
- **Model Leaderboards** - Performance rankings by accuracy, cost efficiency, and execution speed
- **Cost Analysis** - Detailed cost breakdowns with cost-per-success metrics and budget projections
- **Failure Analysis** - Categorised failure reasons with debugging insights and error patterns
- **Performance Trends** - Visualisation of model behaviour across difficulty levels and test iterations
- **Resource Usage** - Token consumption patterns and API call efficiency metrics
- **Comparative Analysis** - Side-by-side model performance comparison with statistical significance

### Dashboard Quick Launch
```bash
# Launch the interactive evaluation dashboard
uv run streamlit run agents_mcp_usage/multi_mcp/eval_multi_mcp/merbench_ui.py
```

The dashboard automatically loads evaluation results from the `mermaid_eval_results/` directory, providing immediate insights into model performance and cost efficiency.

## Multi-Model Benchmarking

The `run_multi_evals.py` script enables systematic comparison across multiple LLM models with flexible execution options:

### Benchmarking Features
- **Parallel Execution** - Simultaneous evaluation across models for faster results
- **Sequential Mode** - Conservative execution for resource-constrained environments
- **Configurable Runs** - Multiple iterations per model for statistical reliability
- **Comprehensive Error Handling** - Robust retry logic with exponential backoff
- **CSV Export** - Structured results for downstream analysis and reporting

### Example Benchmarking Commands

```bash
# Parallel benchmarking across multiple models
uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/run_multi_evals.py \
  --models "gemini-2.5-pro,gemini-2.0-flash,gemini-2.5-flash-preview-04-17" \
  --runs 5 \
  --parallel \
  --timeout 600 \
  --output-dir ./benchmark_results

# Sequential execution with custom judge model
uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/run_multi_evals.py \
  --models "gemini-2.5-pro,claude-3-opus" \
  --runs 3 \
  --sequential \
  --judge-model "gemini-2.5-pro" \
  --output-dir ./comparative_analysis
```

### Output Structure
Results are organised with timestamped files:
- **Individual model results** - `YYYY-MM-DD_HH-MM-SS_individual_{model}.csv`
- **Combined analysis** - `YYYY-MM-DD_HH-MM-SS_combined_results.csv`
- **Dashboard integration** - Automatic loading into visualisation interface

The evaluation system enables robust, repeatable benchmarking across LLM models and agent frameworks, supporting both research investigations and production model selection decisions.

## Example Files

### Pydantic-AI Multi-MCP

**File:** `multi_mcp_use/pydantic_mcp.py`

This example demonstrates how to use multiple MCP servers with Pydantic-AI agents.

```bash
uv run agents_mcp_usage/multi_mcp/multi_mcp_use/pydantic_mcp.py
```

Key features:
- Connects to multiple specialised MCP servers simultaneously
- Organises tools and resources by domain
- Shows how to coordinate between different MCP servers
- Includes Logfire instrumentation for comprehensive tracing

### Google ADK Multi-MCP

**File:** `multi_mcp_use/adk_mcp.py`

This example demonstrates how to use multiple MCP servers with Google's Agent Development Kit (ADK).

```bash
uv run agents_mcp_usage/multi_mcp/multi_mcp_use/adk_mcp.py
```

Key features:
- Uses Google's ADK framework with Gemini model
- Connects to both Python MCP server and Python Mermaid validator
- Demonstrates proper connection management with contextlib.AsyncExitStack
- Shows how to handle asynchronous MCP tool integration
- Uses a simple test case that utilizes both MCP servers in a single query

### Multi-MCP Evaluation

**File:** `eval_multi_mcp/evals_pydantic_mcp.py`

This example demonstrates how to evaluate the effectiveness of using multiple MCP servers.

```bash
uv run agents_mcp_usage/multi_mcp/eval_multi_mcp/evals_pydantic_mcp.py
```

Key features:
- Evaluates agent performance when using multiple specialised MCP servers
- Uses PydanticAI Agent evaluation to measure success of outcomes
- Generates performance metrics viewable in Logfire

## Benefits of Multi-MCP Architecture

Using multiple specialised MCP servers offers several advantages:

1. **Domain Separation**: Each MCP server can focus on a specific domain or set of capabilities.
2. **Modularity**: Add, remove, or update capabilities without disrupting the entire system.
3. **Scalability**: Distribute load across multiple servers for better performance.
4. **Specialisation**: Optimise each MCP server for its specific use case.
5. **Security**: Control access to sensitive tools or data through separate servers.

This approach provides a more flexible and maintainable architecture for complex agent systems.
