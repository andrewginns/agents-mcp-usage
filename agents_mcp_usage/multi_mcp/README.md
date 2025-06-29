# Multi-MCP Usage

This directory contains examples demonstrating the integration of tools from multiple Model Context Protocol (MCP) servers with various LLM agent frameworks.

Agents utilising multiple MCP servers can be dramatically more complex than an Agent using a single server. This is because as the number of servers grow the number of tools that the Agent must reason on when and how to use increases. For evaluating and benchmarking these agents, please see the [evaluation suite](../evaluations/mermaid_evals/README.md).



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
   
   ## Launch ADK web UI for visual interaction
   make adk_multi_ui
   
   # Run the evaluation suite
   uv run agents_mcp_usage/evaluations/mermaid_evals/evals_pydantic_mcp.py
   
   # Run multi-model benchmarking
   uv run agents_mcp_usage/evaluations/mermaid_evals/run_multi_evals.py --models "gemini-2.5-pro-preview-06-05,gemini-2.0-flash" --runs 5 --parallel
   
   # Launch the evaluation dashboard
   uv run streamlit run agents_mcp_usage/evaluations/mermaid_evals/merbench_ui.py
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

## Evaluations

Located at [agents_mcp_usage/evaluations/mermaid_evals](../evaluations/mermaid_evals/README.md). The evaluation suite provides comprehensive benchmarking for LLM agents using multiple MCP servers.

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

This example demonstrates how to use multiple MCP servers with Google's Agent Development Kit (ADK 1.3.0).

```bash
uv run agents_mcp_usage/multi_mcp/multi_mcp_use/adk_mcp.py
```

Key features:
- Uses Google's ADK framework with Gemini model
- Connects to both Python MCP server and Python Mermaid validator
- Uses new ADK 1.3.0 patterns for toolset management
- Implements proper resource tracking for MCP connections
- Shows how to handle asynchronous MCP tool integration
- Supports ADK web UI through module exports and callback-based tool attachment
- Uses a simple test case that utilizes both MCP servers in a single query

To run with the ADK web UI:
```bash
make adk_multi_ui
```

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
