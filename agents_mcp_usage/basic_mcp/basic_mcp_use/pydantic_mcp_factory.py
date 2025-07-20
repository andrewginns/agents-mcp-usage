"""
Pydantic MCP Example using Model Factory Pattern

This example demonstrates how to use the model factory to create agents
with different providers while maintaining the same MCP server setup.

Supported model formats:
- Standard: "gemini-2.5-pro", "gpt-4", "claude-3-opus"
- Provider-prefixed: "openai:gpt-4", "deepseek:deepseek-chat", "bedrock:us.amazon.nova-pro-v1:0"
- Local models: "ollama:llama3.2", "ollama:qwen2.5-coder:7b"
"""

import asyncio
import argparse
import os
from typing import Optional

import logfire
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio

from agents_mcp_usage.utils import get_mcp_server_path
from agents_mcp_usage.factory.model_factory import create_agent

load_dotenv()

# Configure logging to logfire if LOGFIRE_TOKEN is set in environment
logfire.configure(send_to_logfire="if-token-present", service_name="pydantic-basic-mcp-factory")
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

# Preset model aliases demonstrating different provider formats
MODEL_ALIASES = {
    # Standard providers (auto-detected)
    "gemini": "gemini-2.5-pro-preview-06-05",
    "gemini-flash": "gemini-2.5-flash",
    "gpt4": "gpt-4",
    "claude": "claude-3-opus-20240229",
    
    # Explicit provider prefix
    "openai-o4": "openai:o4-mini",
    "deepseek": "deepseek:deepseek-chat",
    "deepseek-reasoner": "deepseek:deepseek-reasoner",
    
    # AWS Bedrock models
    "bedrock-nova": "bedrock:us.amazon.nova-pro-v1:0",
    "bedrock-claude": "bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    
    # GitHub Models (free tier)
    "github-grok": "github:xai/grok-3-mini",
    
    # Local models via Ollama
    "ollama-llama": "ollama:llama3.2",
    "ollama-qwen": "ollama:qwen2.5-coder:7b",
    
    # Other providers
    "perplexity": "perplexity:sonar-pro",
    "grok": "grok:grok-2-1212",
}


def get_mcp_server() -> MCPServerStdio:
    """Get the MCP server configuration."""
    return MCPServerStdio(
        command="uv",
        args=[
            "run",
            str(get_mcp_server_path("example_server.py")),
            "stdio",
        ],
    )


async def main(
    model: str = "gemini-2.5-pro-preview-06-05",
    query: str = "Greet Andrew and give him the current time",
    provider_kwargs: Optional[dict] = None
) -> None:
    """Runs the Pydantic agent with a given query using the model factory.

    This function demonstrates how to use the model factory to create agents
    with different providers while maintaining the same MCP functionality.

    Args:
        model: The model identifier (e.g., "gemini-2.5-pro", "deepseek:deepseek-chat")
        query: The query to run the agent with
        provider_kwargs: Optional provider-specific configuration (e.g., for Azure)
    """
    # Create MCP server
    server = get_mcp_server()
    
    # Create agent using the factory
    print(f"Creating agent with model: {model}")
    agent = create_agent(
        model=model,
        mcp_servers=[server],
        provider_kwargs=provider_kwargs
    )
    
    # Run the agent
    try:
        async with agent.run_mcp_servers():
            result = await agent.run(query)
        print(f"\nResult from {model}:")
        print(result.output)
        
        # Print usage information if available
        usage = result.usage()
        if usage:
            print(f"\nToken usage: {usage.total_tokens} total "
                  f"({usage.request_tokens} in, {usage.response_tokens} out)")
    
    except Exception as e:
        print(f"Error running agent: {e}")
        print("\nMake sure you have the required API keys set:")
        print("- GEMINI_API_KEY for Gemini models")
        print("- OPENAI_API_KEY for OpenAI models")
        print("- DEEPSEEK_API_KEY for DeepSeek models")
        print("- AWS credentials for Bedrock models")
        print("- Other provider-specific keys as needed")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Pydantic MCP agent with different model providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Gemini model
  python pydantic_mcp_factory.py
  
  # Use a specific model by alias
  python pydantic_mcp_factory.py --model deepseek
  
  # Use a model by full name
  python pydantic_mcp_factory.py --model "openai:gpt-4"
  
  # Use a custom query
  python pydantic_mcp_factory.py --query "What's the weather like?"
  
  # List available model aliases
  python pydantic_mcp_factory.py --list-models
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gemini",
        help="Model to use (alias or full model string)"
    )
    
    parser.add_argument(
        "--query", "-q",
        default="Greet Andrew and give him the current time",
        help="Query to send to the agent"
    )
    
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available model aliases and exit"
    )
    
    # Provider-specific options
    parser.add_argument(
        "--azure-endpoint",
        help="Azure endpoint for Azure AI Foundry models"
    )
    
    parser.add_argument(
        "--azure-api-version",
        help="Azure API version for Azure AI Foundry models"
    )
    
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434/v1",
        help="Base URL for Ollama server"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # List models and exit if requested
    if args.list_models:
        print("Available model aliases:")
        print("-" * 50)
        for alias, model_string in MODEL_ALIASES.items():
            print(f"{alias:<20} -> {model_string}")
        print("\nYou can also use any model string directly, e.g.:")
        print("  openrouter:google/gemini-2.5-pro-preview")
        print("  fireworks:accounts/fireworks/models/qwq-32b")
        print("  together:meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
        exit(0)
    
    # Resolve model alias if applicable
    model = MODEL_ALIASES.get(args.model, args.model)
    
    # Build provider kwargs if Azure options are provided
    provider_kwargs = {}
    if args.azure_endpoint:
        provider_kwargs["azure_endpoint"] = args.azure_endpoint
    if args.azure_api_version:
        provider_kwargs["api_version"] = args.azure_api_version
    if args.ollama_base_url and model.startswith("ollama:"):
        provider_kwargs["base_url"] = args.ollama_base_url
    
    # Run the main function
    asyncio.run(main(
        model=model,
        query=args.query,
        provider_kwargs=provider_kwargs if provider_kwargs else None
    ))
