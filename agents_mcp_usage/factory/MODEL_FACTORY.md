# Model Factory Documentation

The model factory pattern provides a centralized way to create models across different providers in PydanticAI, leveraging the framework's built-in provider support while handling provider-specific configurations.

## Overview

The model factory (`agents_mcp_usage/factory/model_factory.py`) replaces manual string parsing with a configuration-driven approach that:

- Supports 13+ providers out of the box
- Leverages PydanticAI's native provider detection
- Handles provider-specific quirks via model profiles
- Provides an extensible design for adding new providers

## Usage

### Basic Usage

```python
from agents_mcp_usage.factory.model_factory import create_agent

# Using provider:model format
agent = create_agent(
    model="deepseek:deepseek-chat",
    mcp_servers=get_mcp_servers(),
    model_settings={"temperature": 0.7}
)

# Using native model names (automatically detected)
agent = create_agent(
    model="gemini-2.5-pro",  # Google provider auto-detected
    mcp_servers=get_mcp_servers()
)
```

### Model String Formats

The factory supports multiple model string formats:

1. **Provider Prefix**: `provider:model`
   - `deepseek:deepseek-chat`
   - `openrouter:google/gemini-2.5-pro`
   - `github:xai/grok-3-mini`

2. **Native Models**: Direct names (provider auto-detected)
   - `gemini-2.5-pro` → Google
   - `gpt-4` → OpenAI
   - `claude-3-opus` → Anthropic

3. **Special Formats**:
   - Bedrock: `bedrock:us.amazon.nova-pro-v1:0`
   - Ollama: `ollama:llama3.2`

## Supported Providers

### Native PydanticAI Providers
These providers are handled directly by PydanticAI without special configuration:
- **Google**: Gemini models
- **OpenAI**: GPT models
- **Anthropic**: Claude models

### OpenAI-Compatible Providers
These providers use the OpenAI API format with custom endpoints:

| Provider | Env Variable | Example Model |
|----------|-------------|---------------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek:deepseek-chat` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter:anthropic/claude-3.5-sonnet` |
| Perplexity | `PERPLEXITY_API_KEY` | `perplexity:sonar-pro` |
| Ollama | None (local) | `ollama:llama3.2` |

### OpenRouter (explicit provider handling)

This repo intentionally constructs an explicit OpenRouter provider/model for `openrouter:*` model strings, rather than relying on PydanticAI's implicit provider detection.

Implementation details:
- `openrouter:<provider>/<model>` is parsed by [`parse_model_string()`](agents_mcp_usage/factory/model_factory.py:110) and handled in [`create_model()`](agents_mcp_usage/factory/model_factory.py:258) via the OpenRouter handler.
- The handler builds an OpenAI-compatible client pointed at `https://openrouter.ai/api/v1` and wraps it in PydanticAI's [`OpenRouterProvider`](agents_mcp_usage/factory/model_factory.py:222).

#### App attribution (optional)

OpenRouter supports attribution via headers.

Set these environment variables (optional):

```bash
export OPENROUTER_APP_URL="https://your-app.com"   # sent as HTTP-Referer
export OPENROUTER_APP_TITLE="Your App"            # sent as X-Title
```

Or pass overrides via `provider_kwargs`:

```python
agent = create_agent(
    model="openrouter:anthropic/claude-3.5-sonnet",
    provider_kwargs={
        "api_key": "...",
        "app_url": "https://your-app.com",
        "app_title": "Your App",
    },
)
```

### Cloud Providers

| Provider | Configuration | Example Model |
|----------|--------------|---------------|
| AWS Bedrock | AWS credentials | `bedrock:us.amazon.nova-pro-v1:0` |
| Azure AI | Endpoint + API key | `azure:gpt-4` |
| Heroku AI | `HEROKU_INFERENCE_KEY` | `heroku:claude-3-sonnet` |

### Specialized Providers

| Provider | Env Variable | Example Model |
|----------|-------------|---------------|
| GitHub Models | `GITHUB_API_KEY` | `github:xai/grok-3-mini` |
| Grok (xAI) | `XAI_API_KEY` | `grok:grok-2-1212` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks:accounts/fireworks/models/qwq-32b` |
| Together | `TOGETHER_API_KEY` | `together:meta-llama/Llama-3.3-70B` |

## Model Profiles

Model profiles handle provider-specific behaviors and limitations:

```python
profile = OpenAIModelProfile(
    # Some providers don't support JSON schema references
    json_schema_transformer=InlineDefsJsonSchemaTransformer,
    
    # Some providers don't support strict tool definitions
    openai_supports_strict_tool_definition=False
)
```

### Providers Requiring Profiles
- DeepSeek: Requires inline JSON schemas
- Ollama: Limited tool definition support

## Adding New Providers

To add a new provider, update the `PROVIDER_CONFIGS` dictionary:

```python
PROVIDER_CONFIGS = {
    "new_provider": {
        "provider_class": "pydantic_ai.providers.new_provider.NewProvider",
        "env_var": "NEW_PROVIDER_API_KEY",
        "base_url": "https://api.newprovider.com",  # Optional
        "profile": {  # Optional
            "json_schema_transformer": InlineDefsJsonSchemaTransformer,
            "openai_supports_strict_tool_definition": False
        }
    }
}
```

## Architecture

### Components

1. **parse_model_string()**: Extracts provider and model name from string
2. **create_model()**: Creates appropriate model instance or returns string for PydanticAI
3. **create_agent()**: Convenience function to create agents with proper model handling
4. **PROVIDER_CONFIGS**: Central configuration for all providers

### Flow

1. Model string is parsed to identify provider
2. If native provider (Google/OpenAI/Anthropic), return string for PydanticAI
3. If configured provider, create appropriate model instance
4. If unknown provider, return string and let PydanticAI handle it
5. Apply model profiles if specified in configuration

## Benefits

1. **Centralized Configuration**: All provider settings in one place
2. **Extensibility**: Easy to add new providers
3. **Consistency**: Same interface regardless of provider
4. **Flexibility**: Supports both native and custom providers
5. **Type Safety**: Leverages PydanticAI's type system

## Example: Multi-Provider Comparison

```python
from agents_mcp_usage.factory.model_factory import create_agent

models = [
    "gemini-2.5-pro",
    "openai:gpt-4",
    "deepseek:deepseek-chat",
    "ollama:llama3.2"
]

for model in models:
    agent = create_agent(
        model=model,
        mcp_servers=get_mcp_servers()
    )
    result = await agent.run("Solve this problem...")
    print(f"{model}: {result}")
```

## Environment Variables

Most providers require API keys via environment variables:

```bash
# Native providers
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

 # Extended providers
 export DEEPSEEK_API_KEY="..."
 export OPENROUTER_API_KEY="..."
 export OPENROUTER_APP_URL="..."   # optional (app attribution)
 export OPENROUTER_APP_TITLE="..." # optional (app attribution)
 export GITHUB_API_KEY="..."
 export XAI_API_KEY="..."
 export PERPLEXITY_API_KEY="..."

# AWS Bedrock
export AWS_REGION="us-east-1"
export AWS_PROFILE="default"
```

## Migration Guide

### Old Approach
```python
if model.startswith("bedrock:"):
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider
    
    model_name = model.replace("bedrock:", "")
    bedrock_model = BedrockConverseModel(
        model_name,
        provider=BedrockProvider(
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            profile_name=os.getenv("AWS_PROFILE", "default")
        )
    )
    agent = Agent(bedrock_model, ...)
```

### New Approach
```python
from agents_mcp_usage.factory.model_factory import create_agent

agent = create_agent(
    model="bedrock:us.amazon.nova-pro-v1:0",
    mcp_servers=get_mcp_servers()
)
```

The model factory handles all the provider-specific logic internally, providing a cleaner and more maintainable approach.
