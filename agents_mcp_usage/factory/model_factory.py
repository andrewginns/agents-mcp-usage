"""
Model Factory for PydanticAI

This module provides a centralized way to create models across different providers,
leveraging PydanticAI's built-in provider support and handling special cases.
"""

import os
import re
from typing import Any, Dict, Optional, Union
from pydantic_ai import Agent, InlineDefsJsonSchemaTransformer
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider


# Providers that PydanticAI handles natively with provider:model syntax
NATIVE_PROVIDERS = {
    "deepseek", "openrouter", "github", "grok", "azure",
    "fireworks", "together", "heroku"
}

# Special cases that need custom handling
PROVIDER_CONFIGS = {
    "bedrock": {
        # Bedrock uses a different model class
        "handler": "bedrock_handler",
        "env_vars": {
            "region": "AWS_REGION",
            "profile": "AWS_PROFILE"
        }
    },
    "ollama": {
        # Ollama is OpenAI-compatible but needs custom base URL
        "handler": "openai_compatible_handler",
        "base_url": "http://localhost:11434/v1",
        "api_key": "not-needed",
        "profile": {
            "json_schema_transformer": InlineDefsJsonSchemaTransformer,
            "openai_supports_strict_tool_definition": False
        }
    },
    "perplexity": {
        # Perplexity is OpenAI-compatible with custom base URL
        "handler": "openai_compatible_handler",
        "base_url": "https://api.perplexity.ai",
        "env_var": "PERPLEXITY_API_KEY"
    }
}


def ensure_openai_responses_model(model_string: str) -> str:
    """Ensure OpenAI models use the Responses API.

    This helper normalises any OpenAI model identifier to the
    `openai-responses:` provider namespace so that PydanticAI
    will use the OpenAI Responses API rather than Chat Completions.

    Examples:
        "gpt-5" -> "openai-responses:gpt-5"
        "openai:gpt-5-mini" -> "openai-responses:gpt-5-mini"
        "openai-responses:gpt-5" -> "openai-responses:gpt-5" (unchanged)
    """
    # Avoid double-prefixing if already using Responses API directly
    if model_string.startswith("openai-responses:"):
        return model_string

    provider, model_name = parse_model_string(model_string)
    if provider == "openai":
        return f"openai-responses:{model_name}"

    return model_string


REASONING_SUFFIX_RE = re.compile(r"\s*\((low|medium|high)\)\s*$", re.IGNORECASE)


def extract_reasoning_effort(model_string: str) -> tuple[str, Optional[str]]:
    """Strip trailing reasoning hint like "(medium)" from model string.

    Returns the base model string and the effort level (lowercased) if present.
    """

    match = REASONING_SUFFIX_RE.search(model_string)
    if not match:
        return model_string.strip(), None

    effort = match.group(1).lower()
    base = REASONING_SUFFIX_RE.sub("", model_string).strip()
    return base, effort


def parse_model_string(model_string: str) -> tuple[Optional[str], str]:
    """
    Parse a model string to extract provider and model name.
    
    Args:
        model_string: Model identifier (e.g., "openai:gpt-4", "deepseek:deepseek-chat", "gemini-2.5-pro")
    
    Returns:
        Tuple of (provider, model_name)
    """
    if ":" in model_string:
        provider, model_name = model_string.split(":", 1)
        return provider, model_name
    
    # Check for known prefixes without colon
    if model_string.startswith("gemini-"):
        return "google", model_string
    elif model_string.startswith("gpt-") or model_string.startswith("o1-") or model_string.startswith("o3-") or model_string.startswith("o4-"):
        return "openai", model_string
    elif model_string.startswith("claude-"):
        return "anthropic", model_string
    
    # Default to no specific provider
    return None, model_string


def handle_bedrock_model(
    model_name: str,
    provider_kwargs: Optional[Dict[str, Any]] = None
) -> Model:
    """Handle AWS Bedrock models which require special treatment."""
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider
    
    config = PROVIDER_CONFIGS["bedrock"]
    provider_kwargs = provider_kwargs or {}
    
    region = provider_kwargs.get("region_name") or os.getenv(config["env_vars"]["region"], "us-east-1")
    profile = provider_kwargs.get("profile_name") or os.getenv(config["env_vars"]["profile"], "default")
    
    provider_instance = BedrockProvider(
        region_name=region,
        profile_name=profile
    )
    
    return BedrockConverseModel(
        model_name,
        provider=provider_instance
    )


def handle_openai_compatible(
    provider: str,
    model_name: str,
    provider_kwargs: Optional[Dict[str, Any]] = None
) -> Model:
    """Handle OpenAI-compatible providers that need custom configuration."""
    config = PROVIDER_CONFIGS[provider]
    provider_kwargs = provider_kwargs or {}
    
    # Build OpenAI provider kwargs
    openai_kwargs = {}
    
    # Set base URL
    if "base_url" in config:
        openai_kwargs["base_url"] = provider_kwargs.get("base_url", config["base_url"])
    
    # Handle API key
    if "api_key" in config:
        openai_kwargs["api_key"] = config["api_key"]
    elif "env_var" in config:
        api_key = provider_kwargs.get("api_key") or os.getenv(config["env_var"])
        if api_key:
            openai_kwargs["api_key"] = api_key
    
    # Create OpenAI provider
    provider_instance = OpenAIProvider(**openai_kwargs)
    
    # Create model with profile if specified
    if config.get("profile"):
        profile = OpenAIModelProfile(**config["profile"])
        return OpenAIResponsesModel(
            model_name,
            provider=provider_instance,
            profile=profile,
        )
    else:
        return OpenAIResponsesModel(
            model_name,
            provider=provider_instance,
        )


def create_model(
    model_string: str,
    model_settings: Optional[Dict[str, Any]] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None
) -> Union[Model, str]:
    """
    Create a model instance from a model string.
    
    This function:
    1. Returns the string directly for providers PydanticAI handles natively
    2. Handles special cases like Bedrock and OpenAI-compatible providers
    3. Falls back to returning the string for unknown providers
    
    Args:
        model_string: Model identifier
        model_settings: Optional model-specific settings
        provider_kwargs: Optional provider-specific kwargs (e.g., azure_endpoint)
    
    Returns:
        Either a Model instance or the original string for PydanticAI to handle
    """
    # First, normalise any OpenAI models to use the Responses API
    model_string = ensure_openai_responses_model(model_string)

    provider, model_name = parse_model_string(model_string)
    
    # No provider prefix - let PydanticAI handle it
    if provider is None:
        return model_string
    
    # Standard providers that PydanticAI auto-detects
    # Note: OpenAI models have been converted to "openai-responses:*" above
    if provider in ["google", "anthropic"]:
        return model_string
    
    # Providers with native PydanticAI support
    if provider in NATIVE_PROVIDERS:
        # Special validation for Azure
        if provider == "azure" and provider_kwargs:
            required = ["azure_endpoint", "api_version"]
            missing = [k for k in required if k not in provider_kwargs]
            if missing:
                raise ValueError(
                    f"Azure provider requires {missing} in provider_kwargs. "
                    f"Alternatively, let PydanticAI handle it by using the string directly."
                )
        # Return the string - PydanticAI will handle it
        return model_string
    
    # Handle special cases
    if provider in PROVIDER_CONFIGS:
        config = PROVIDER_CONFIGS[provider]
        handler = config.get("handler")
        
        if handler == "bedrock_handler":
            return handle_bedrock_model(model_name, provider_kwargs)
        elif handler == "openai_compatible_handler":
            return handle_openai_compatible(provider, model_name, provider_kwargs)
    
    # Unknown provider - return string and let PydanticAI try
    return model_string


def create_agent(
    model: Union[str, Model],
    mcp_servers=None,
    model_settings: Optional[Dict[str, Any]] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    **agent_kwargs
) -> Agent:
    """
    Create an Agent with proper model handling.
    
    This is a convenience function that:
    1. Converts model strings to appropriate Model instances when needed
    2. Passes through Model instances unchanged
    3. Leverages PydanticAI's native provider support
    
    Args:
        model: Model string or Model instance
        mcp_servers: Optional MCP servers
        model_settings: Optional model settings
        provider_kwargs: Optional provider-specific kwargs
        **agent_kwargs: Additional Agent kwargs
    
    Returns:
        Configured Agent instance
    """
    if isinstance(model, str):
        base_model, reasoning_effort = extract_reasoning_effort(model)

        # Ensure we have a dict to attach model-level settings
        if model_settings is None:
            model_settings = {}

        # Only apply reasoning hint to OpenAI Responses-capable models
        provider, _ = parse_model_string(base_model)
        if (
            reasoning_effort
            and isinstance(model_settings, dict)
            and provider in {"openai", "openai-responses"}
            and "openai_reasoning_effort" not in model_settings
        ):
            model_settings["openai_reasoning_effort"] = reasoning_effort

        model = create_model(base_model, model_settings, provider_kwargs)

    return Agent(
        model,
        mcp_servers=mcp_servers,
        model_settings=model_settings if isinstance(model, str) else None,
        **agent_kwargs
    )
