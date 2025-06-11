invalid_mermaid_diagram_hard = """
```mermaid
graph LR
    User((User)) --> |"Run script<br>(e.g., pydantic_mcp.py)"| Agent

    # Agent Frameworks
    subgraph "Agent"
        direction TD
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

    # MCP Server
    subgraph "MCP"
        direction TD
        MCP["Model Context Protocol Server<br>(mcp_servers/example_server.py)"]
        Tools["Tools<br>- add(a, b)<br>- get_current_time() {current_time}"]
        Resources["Resources<br>- greeting://{{name}}"]
        MCP --- Tools
        MCP --- Resources
    end

    # LLM Providers
    subgraph "LLM Providers"
        direction TD
        OAI_LLM["OpenAI Models"]
        GEM["Google Gemini Models"]
        OTHER["Other LLM Providers..."]
    end

    Logfire[("Logfire<br>Tracing")]

    ADK --> MCP
    LG -- > MCP
    OAI --> MCP
    PYD --> MCP

    MCP --> OAI_LLM
    MCP --> GEM
    MCP --> OTHER

    ADK --> Logfire
    LG -- > Logfire
    OAI --> Logfire
    PYD --> Logfire

    LLM_Response[("Response")] --> User
    OAI_LLM --> LLM_Response
    GEM --> LLM_Response
    OTHER --> LLM_Response

    style MCP fill:#f9f,stroke:#333,stroke-width:2px
    style User fill:#bbf,stroke:#338,stroke-width:2px
    style Logfire fill:#bfb,stroke:#383,stroke-width:2px
    style LLM_Response fill:#fbb,stroke:#833,stroke-width:2px
```
"""

# 7 syntax errors
invalid_mermaid_diagram_medium = """
```mermaid
graph LR
    User((User)) --> |"Run script<br>(e.g., pydantic_mcp.py)"| Agent

    # Agent Frameworks
    subgraph "Agent"
        direction TB
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

    # MCP Server
    subgraph "MCP"
        direction TB
        MCP["Model Context Protocol Server<br>(mcp_servers/example_server.py)"]
        Tools["Tools<br>- add(a, b)<br>- get_current_time() {current_time}"]
        Resources["Resources<br>- greeting://{{name}}"]
        MCP --- Tools
        MCP --- Resources
    end

    # LLM Providers
    subgraph "LLM Providers"
        direction TB
        OAI_LLM["OpenAI Models"]
        GEM["Google Gemini Models"]
        OTHER["Other LLM Providers..."]
    end

    Logfire[("Logfire<br>Tracing")]

    ADK --> MCP
    LG -- > MCP
    OAI --> MCP
    PYD --> MCP

    MCP --> OAI_LLM
    MCP --> GEM
    MCP --> OTHER

    ADK --> Logfire
    LG -- > Logfire
    OAI --> Logfire
    PYD --> Logfire

    LLM_Response[("Response")] --> User
    OAI_LLM --> LLM_Response
    GEM --> LLM_Response
    OTHER --> LLM_Response

    style MCP fill:#f9f,stroke:#333,stroke-width:2px
    style User fill:#bbf,stroke:#338,stroke-width:2px
    style Logfire fill:#bfb,stroke:#383,stroke-width:2px
    style LLM_Response fill:#fbb,stroke:#833,stroke-width:2px
```
"""

# 2 syntax errors
invalid_mermaid_diagram_easy = """
```mermaid
graph LR
    User((User)) --> |"Run script<br>(e.g., pydantic_mcp.py)"| Agent

    %% Agent Frameworks
    subgraph "Agent Frameworks"
        direction TB
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

    %% MCP Server
    subgraph "MCP"
        direction TB
        MCP["Model Context Protocol Server<br>(mcp_servers/example_server.py)"]
        Tools["Tools<br>- add(a, b)<br>- get_current_time() {current_time}"]
        Resources["Resources<br>- greeting://{{name}}"]
        MCP --- Tools
        MCP --- Resources
    end

    %% LLM Providers
    subgraph "LLM Providers"
        direction TB
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
    MCP --> GEMINI
    MCP --> OTHER

    ADK --> Logfire
    LG --> Logfire
    OAI --> Logfire
    PYD --> Logfire

    LLM_Response[("Response")] --> User
    OAI_LLM --> LLM_Response
    GEM --> LLM_Response
    OTHER --> LLM_Response

    style MCP fill:#f9f,stroke:#333,stroke-width:2px
    style User fill:#bbf,stroke:#338,stroke-width:2px
    style Logfire fill:#bfb,stroke:#383,stroke-width:2px
    style LLM_Response fill:#fbb,stroke:#833,stroke-width:2px
```
"""

valid_mermaid_diagram = """`
```mermaid
graph LR
    User((User)) --> |"Run script<br>(e.g., pydantic_mcp.py)"| Agent

    %% Agent Frameworks
    subgraph "Agent Frameworks"
        direction TB
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

    %% MCP Server
    subgraph "MCP Server"
        direction TB
        MCP["Model Context Protocol Server<br>(mcp_servers/example_server.py)"]
        Tools["Tools<br>- add(a, b)<br>- get_current_time() {current_time}"]
        Resources["Resources<br>- greeting://{{name}}"]
        MCP --- Tools
        MCP --- Resources
    end

    %% LLM Providers
    subgraph "LLM Providers"
        direction TB
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

    style MCP fill:#f9f,stroke:#333,stroke-width:2px
    style User fill:#bbf,stroke:#338,stroke-width:2px
    style Logfire fill:#bfb,stroke:#383,stroke-width:2px
    style LLM_Response fill:#fbb,stroke:#833,stroke-width:2px
```
"""
