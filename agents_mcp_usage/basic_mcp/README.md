# Basic MCP Usage Examples

This directory contains examples of integrating Model Context Protocol (MCP) with various LLM agent frameworks.

Each script demonstrates how to connect to a single local MCP server and use it with a different agent framework.

## Quickstart

1. Configure `.env` and API keys following instructions in the [README.md](README.md)

2. Run any example script:
   ```bash
   # Run the Pydantic-AI example
   uv run agents_mcp_usage/basic_mcp/basic_mcp_use/pydantic_mcp.py
   
   # Run the OpenAI Agents example
   uv run agents_mcp_usage/basic_mcp/basic_mcp_use/oai-agent_mcp.py
   
   # Run the LangGraph example
   uv run agents_mcp_usage/basic_mcp/basic_mcp_use/langgraph_mcp.py
   
   # Run the Google ADK example
   uv run agents_mcp_usage/basic_mcp/basic_mcp_use/adk_mcp.py
   ```

4. Check the console output or Logfire for results.

### Basic MCP Architecture

```mermaid
graph LR
    User((User)) --> |"Run script<br>(e.g., pydantic_mcp.py)"| Agent

    subgraph "Agent Frameworks"
        Agent[Agent]
        ADK["Google ADK<br>(adk_mcp.py)"]
        LG["LangGraph<br>(langgraph_mcp.py)"]
        OAI["OpenAI Agents<br>(oai-agent_mcp.py)"]
        PYD["Pydantic-AI<br>(pydantic_mcp.py)"]
        
        Agent --> ADK
        Agent --> LG
        Agent --> OAI
        Agent --> PYD
    end

    subgraph "Python MCP Server"
        MCP["Model Context Protocol Server<br>(run_server.py)"]
        Tools["Tools<br>- add(a, b)<br>- get_current_time()"]
        Resources["Resources<br>- greeting://{name}"]
        MCP --- Tools
        MCP --- Resources
    end

    subgraph "LLM Providers"
        OAI_LLM["OpenAI Models"]
        GEM["Google Gemini Models"]
        OTHER["Other LLM Providers..."]
    end
    
    Logfire[("Logfire<br>Tracing")]
    
    ADK --> MCP
    LG --> MCP
    OAI --> MCP
    PYD --> MCP
    
    MCP --> OAI_LLM
    MCP --> GEM
    MCP --> OTHER
    
    ADK --> Logfire
    LG --> Logfire
    OAI --> Logfire
    PYD --> Logfire
    
    LLM_Response[("Response")] --> User
    OAI_LLM --> LLM_Response
    GEM --> LLM_Response
    OTHER --> LLM_Response

    %% Node styling for better dark/light mode readability
    classDef userNode fill:#b3e0ff,stroke:#0066cc,stroke-width:2px,color:#000000;
    classDef agentNode fill:#d1c4e9,stroke:#673ab7,stroke-width:1px,color:#000000;
    classDef mcpNode fill:#ffccbc,stroke:#ff5722,stroke-width:2px,color:#000000;
    classDef toolNode fill:#ffe0b2,stroke:#ff9800,stroke-width:1px,color:#000000;
    classDef llmNode fill:#c8e6c9,stroke:#4caf50,stroke-width:1px,color:#000000;
    classDef outputNode fill:#ffcdd2,stroke:#e53935,stroke-width:2px,color:#000000;
    classDef logNode fill:#e1bee7,stroke:#8e24aa,stroke-width:2px,color:#000000;

    %% Apply styles to nodes
    class User,LLM_Response userNode;
    class Agent,ADK,LG,OAI,PYD agentNode;
    class MCP mcpNode;
    class Tools,Resources toolNode;
    class OAI_LLM,GEM,OTHER llmNode;
    class LLM_Response outputNode;
    class Logfire logNode;
```

The diagram illustrates how MCP serves as a standardised interface between different agent frameworks and LLM providers.The flow shows how users interact with the system by running a specific agent script, which then leverages MCP to communicate with LLM providers, while Logfire provides tracing and observability.

### Basic MCP Sequence Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as Agent Framework
    participant MCP as Python MCP Server
    participant LLM as LLM Provider
    participant Tools as MCP Tools
    participant Logfire as Logfire Tracing
    
    Note over User,Logfire: Basic MCP Interaction Flow
    
    User->>Agent: Run agent script with query
    
    activate Agent
    Agent->>Logfire: Start tracing
    
    Agent->>MCP: Initialise connection
    activate MCP
    MCP-->>Agent: Connection established
    
    Agent->>MCP: Send user query
    
    MCP->>LLM: Generate response/action
    activate LLM
    
    loop Tool Use Cycle
        LLM-->>MCP: Request tool execution
        MCP->>Tools: Execute tool (e.g., add, get_time)
        activate Tools
        Tools-->>MCP: Return tool result
        deactivate Tools
        MCP->>LLM: Continue with tool result
    end
    
    LLM-->>MCP: Final response
    deactivate LLM
    
    MCP-->>Agent: Return response
    deactivate MCP
    
    Agent->>Logfire: Log completion
    Agent->>User: Display final answer
    deactivate Agent
    
    Note over User,Logfire: End of interaction
```

The sequence diagram illustrates the temporal flow of interactions between the user, agent framework, MCP server, LLM provider, and tools. It highlights how the tool execution cycle operates within the MCP architecture.

### Google Agent Development Kit (ADK)

**File:** `adk_mcp.py`

This example demonstrates how to use MCP with Google's Agent Development Kit (ADK).

```bash
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/adk_mcp.py
```

Key features:
- Uses `MCPToolset` for connecting to the MCP server
- Configures a Gemini model using ADK's `LlmAgent`
- Sets up session handling and runner for agent execution
- Includes Logfire instrumentation for tracing

### LangGraph

**File:** `langgraph_mcp.py`

This example demonstrates how to use MCP with LangGraph agents.

```bash
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/langgraph_mcp.py
```

Key features:
- Uses LangChain MCP adapters to load tools
- Creates a ReAct agent with LangGraph
- Demonstrates stdio-based client connection to MCP server
- Uses Gemini model for agent reasoning

### OpenAI Agents

**File:** `oai-agent_mcp.py`

This example demonstrates how to use MCP with OpenAI's Agents package.

```bash
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/oai-agent_mcp.py
```

Key features:
- Uses OpenAI's Agent and Runner classes
- Connects to MCP server through MCPServerStdio
- Uses OpenAI's o4-mini model
- Includes Logfire instrumentation for both MCP and OpenAI Agents

### Pydantic-AI

**File:** `pydantic_mcp.py`

This example demonstrates how to use MCP with the Pydantic-AI agent framework.

```bash
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/pydantic_mcp.py
```

Key features:
- Uses the simplified Pydantic-AI Agent interface
- Configures MCPServerStdio for MCP communication
- Employs context manager for server lifecycle management
- Includes comprehensive instrumentation for both MCP and Pydantic-AI


## Understanding the Examples

Each example follows a similar pattern:

1. **Environment Setup**: Loading environment variables and configuring logging
2. **Server Connection**: Establishing a connection to the local MCP server
3. **Agent Configuration**: Setting up an agent with the appropriate model
4. **Execution**: Running the agent with a query and handling the response

The examples are designed to be as similar as possible, allowing you to compare how different frameworks approach MCP integration.

## MCP Server

All examples connect to the same MCP server defined in `run_server.py` at the project root. This server provides:

- An addition tool (`add(a, b)`)
- A time tool (`get_current_time()`) 
- A dynamic greeting resource (`greeting://{name}`)

You can modify the MCP server to add your own tools and resources for experimentation. 