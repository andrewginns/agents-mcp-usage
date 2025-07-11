# Model Context Protocol (MCP) Agent Frameworks Demo & Benchmarking Platform

This repository demonstrates LLM Agents using tools from Model Context Protocol (MCP) servers with several frameworks:
- Google Agent Development Kit (ADK)
- LangGraph Agents
- OpenAI Agents
- Pydantic-AI Agents

## Repository Structure

- [Agent with a single MCP server](agents_mcp_usage/basic_mcp/README.md) - Learning examples and basic patterns
- [Agent with multiple MCP servers](agents_mcp_usage/multi_mcp/README.md) - Advanced usage with MCP server coordination
- [Evaluation suite](agents_mcp_usage/evaluations/mermaid_evals/README.md) - Comprehensive benchmarking tools
  - **Evaluation Dashboard**: Interactive Streamlit UI for model comparison
  - **Multi-Model Benchmarking**: Parallel/sequential evaluation across multiple LLMs
  - **Rich Metrics**: Usage analysis, cost comparison, and performance leaderboards

The repo also includes Python MCP Servers:
- [`example_server.py`](mcp_servers/example_server.py) based on [MCP Python SDK Quickstart](https://github.com/modelcontextprotocol/python-sdk/blob/b4c7db6a50a5c88bae1db5c1f7fba44d16eebc6e/README.md?plain=1#L104) - Modified to include a datetime tool and run as a server invoked by Agents
- [`mermaid_validator.py`](mcp_servers/mermaid_validator.py) - Mermaid diagram validation server using mermaid-cli

Tracing is done through Pydantic Logfire.

![MCP Concept](docs/images/mcp_concept.png)

# Quickstart

`cp .env.example .env`
- Add `GEMINI_API_KEY` and/or `OPENAI_API_KEY`
  - Individual scripts can be adjusted to use models from any provider supported by the specific framework
    - By default only [basic_mcp_use/oai-agent_mcp.py](agents_mcp_usage/basic_mcp/basic_mcp_use/oai-agent_mcp.py) requires `OPENAI_API_KEY`
    - All other scripts require `GEMINI_API_KEY` (Free tier key can be created at https://aistudio.google.com/apikey)
- [Optional] Add `LOGFIRE_TOKEN` to visualise evaluations in Logfire web ui

Run an Agent framework script e.g.:
- `uv run agents_mcp_usage/basic_mcp/basic_mcp_use/pydantic_mcp.py`
  - Requires `GEMINI_API_KEY` by default

- `uv run agents_mcp_usage/basic_mcp/basic_mcp_use/oai-agent_mcp.py`
  - Requires `OPENAI_API_KEY` by default

- Launch the ADK web UI for visual interaction with the agents:
  - `make adk_basic_ui`
  
Check console, Logfire, or the ADK web UI for output

## Project Overview

This project aims to teach:
1. How to use MCP with multiple LLM Agent frameworks
    - Agent using a single MCP server ([basic_mcp](#basic-mcp-single-server-usage))
    - Agent using multiple MCP servers ([multi_mcp](#multi-mcp-advanced-usage))
2. How to see traces LLM Agents with Logfire
3. How to evaluate LLMs with PydanticAI evals

![Logfire UI](docs/images/logfire_ui.png)

## Repository Structure

- **[agents_mcp_usage/basic_mcp/](agents_mcp_usage/basic_mcp/)** - Single MCP server integration examples
  - **basic_mcp_use/** - Contains basic examples of single MCP usage:
    - `adk_mcp.py` - Example of using MCP with Google's Agent Development Kit (ADK 1.3.0)
    - `langgraph_mcp.py` - Example of using MCP with LangGraph
    - `oai-agent_mcp.py` - Example of using MCP with OpenAI Agents
    - `pydantic_mcp.py` - Example of using MCP with Pydantic-AI


- **[agents_mcp_usage/multi_mcp/](agents_mcp_usage/multi_mcp/)** - Advanced multi-MCP server integration examples
  - **multi_mcp_use/** - Contains examples of using multiple MCP servers simultaneously:
    - `pydantic_mcp.py` - Example of using multiple MCP servers with Pydantic-AI Agent

- **[agents_mcp_usage/evaluations/](agents_mcp_usage/evaluations/)** - Evaluation modules for benchmarking
  - **mermaid_evals/** - Comprehensive evaluation suite for mermaid diagram fixing tasks
    - `evals_pydantic_mcp.py` - Core evaluation module for single-model testing
    - `run_multi_evals.py` - Multi-model benchmarking with parallel execution
    - `merbench_ui.py` - Interactive dashboard for result visualization

- **Demo Python MCP Servers**
  - `mcp_servers/example_server.py` - Simple MCP server that runs locally, implemented in Python
  - `mcp_servers/mermaid_validator.py` - Mermaid diagram validation MCP server, implemented in Python

## Basic MCP: Single Server Usage

The `basic_mcp` directory demonstrates how to integrate a single MCP server with different agent frameworks. Each example follows a similar pattern:

1. **Environment Setup**: Loading environment variables and configuring logging
2. **Server Connection**: Establishing a connection to the local MCP server
3. **Agent Configuration**: Setting up an agent with the appropriate model
4. **Execution**: Running the agent with a query and handling the response

The MCP server in these examples provides:
- An addition tool (`add(a, b)`)
- A time tool (`get_current_time()`) 
- A dynamic greeting resource (`greeting://{name}`)

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
        MCP["Model Context Protocol Server<br>(mcp_servers/example_server.py)"]
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
```

#### Try the Basic MCP Examples:

```bash
# Google ADK example
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/adk_mcp.py

# LangGraph example
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/langgraph_mcp.py

# OpenAI Agents example
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/oai-agent_mcp.py

# Pydantic-AI example
uv run agents_mcp_usage/basic_mcp/basic_mcp_use/pydantic_mcp.py

# Launch ADK web UI for visual interaction
make adk_basic_ui
```

More details on basic MCP implementation can be found in the [basic_mcp README](agents_mcp_usage/basic_mcp/README.md).

## Multi-MCP: Advanced Usage

The `multi_mcp` directory demonstrates advanced techniques for connecting to and coordinating between multiple specialised MCP servers simultaneously. This approach offers several advantages:

1. **Domain Separation**: Each MCP server can focus on a specific domain or set of capabilities
2. **Modularity**: Add, remove, or update capabilities without disrupting the entire system
3. **Scalability**: Distribute load across multiple servers for better performance
4. **Specialisation**: Optimise each MCP server for its specific use case

### Multi-MCP Architecture

```mermaid
graph LR
    User((User)) --> |"Run script<br>(e.g., pydantic_mcp.py)"| Agent

    subgraph "Agent Framework"
        Agent["Pydantic-AI Agent<br>(pydantic_mcp.py)"]
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
        LLMs["PydanticAI LLM call"]
    end
    
    Logfire[("Logfire<br>Tracing")]
    
    Agent --> PythonMCP
    Agent --> MermaidMCP
    
    PythonMCP --> LLMs
    MermaidMCP --> LLMs
    
    Agent --> Logfire
    
    LLM_Response[("Response")] --> User
    LLMs --> LLM_Response
```

#### Try the Multi-MCP Examples:

```bash
# Run the Pydantic-AI multi-MCP example
uv run agents_mcp_usage/multi_mcp/multi_mcp_use/pydantic_mcp.py

# Run the multi-MCP evaluation
uv run agents_mcp_usage/evaluations/mermaid_evals/evals_pydantic_mcp.py

# Run multi-model benchmarking
uv run agents_mcp_usage/evaluations/mermaid_evals/run_multi_evals.py --models "gemini-2.5-pro-preview-06-05,gemini-2.0-flash" --runs 5 --parallel

# Launch the evaluation dashboard
uv run streamlit run agents_mcp_usage/evaluations/mermaid_evals/merbench_ui.py
```

More details on multi-MCP implementation can be found in the [multi_mcp README](agents_mcp_usage/multi_mcp/README.md).

## Evaluation Suite & Benchmarking Dashboard

This repository includes a comprehensive evaluation system for benchmarking LLM agent performance across multiple frameworks and models. The evaluation suite tests agents on mermaid diagram correction tasks using multiple MCP servers, providing rich metrics and analysis capabilities.

### Key Evaluation Features

- **Multi-Level Difficulty**: Easy, medium, and hard test cases for comprehensive assessment
- **Multi-Model Benchmarking**: Parallel or sequential evaluation across multiple LLM models
- **Interactive Dashboard**: Streamlit-based UI for visualising results, cost analysis, and model comparison
- **Rich Metrics Collection**: Token usage, cost analysis, success rates, and failure categorisation
- **Robust Error Handling**: Comprehensive retry logic and detailed failure analysis
- **Export Capabilities**: CSV results for downstream analysis and reporting

### Dashboard Features

The included Streamlit dashboard (`merbench_ui.py`) provides:

- **Model Leaderboards**: Performance rankings by accuracy, cost efficiency, and speed
- **Cost Analysis**: Detailed cost breakdowns and cost-per-success metrics
- **Failure Analysis**: Categorised failure reasons with debugging insights
- **Performance Trends**: Visualisation of model behaviour across difficulty levels
- **Resource Usage**: Token consumption and API call patterns
- **Comparative Analysis**: Side-by-side model performance comparison

### Quick Evaluation Commands

```bash
# Single model evaluation
uv run agents_mcp_usage/evaluations/mermaid_evals/evals_pydantic_mcp.py

# Multi-model parallel benchmarking
uv run agents_mcp_usage/evaluations/mermaid_evals/run_multi_evals.py \
  --models "gemini-2.5-pro-preview-06-05,gemini-2.0-flash,gemini-2.5-flash" \
  --runs 5 \
  --parallel \
  --output-dir ./results

# Launch interactive dashboard
uv run streamlit run agents_mcp_usage/evaluations/mermaid_evals/merbench_ui.py
```

The evaluation system enables robust, repeatable benchmarking across LLM models and agent frameworks, supporting both research and production model selection decisions.

## What is MCP?

The Model Context Protocol allows applications to provide context for LLMs in a standardised way, separating the concerns of providing context from the actual LLM interaction.

Learn more: https://modelcontextprotocol.io/introduction

## Why MCP

By defining clear specifications for components like resources (data exposure), prompts (reusable templates), tools (actions), and sampling (completions), MCP simplifies the development process and fosters consistency.

A key advantage highlighted is flexibility; MCP allows developers to more easily switch between different LLM providers without needing to completely overhaul their tool and data integrations. It provides a structured approach, potentially reducing the complexity often associated with custom tool implementations for different models. While frameworks like Google Agent Development Kit, LangGraph, OpenAI Agents, or libraries like PydanticAI facilitate agent building, MCP focuses specifically on standardising the interface between the agent's reasoning (the LLM) and its capabilities (tools and data), aiming to create a more interoperable ecosystem.

## Setup Instructions

1. Clone this repository
2. Install required packages:
   ```bash
   make install
   ```

   To use the ADK web UI, run:
   ```bash
   make adk_basic_ui
   ```
3. Set up your environment variables in a `.env` file:
   ```
   LOGFIRE_TOKEN=your_logfire_token
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run any of the sample scripts as shown in the examples above

## About Logfire

[Logfire](https://github.com/pydantic/logfire) is an observability platform from the team behind Pydantic that makes monitoring AI applications straightforward. Features include:

- Simple yet powerful dashboard
- Python-centric insights, including rich display of Python objects
- SQL-based querying of your application data
- OpenTelemetry support for leveraging existing tooling
- Pydantic integration for analytics on validations

Logfire gives you visibility into how your code is running, which is especially valuable for LLM applications where understanding model behaviour is critical.
