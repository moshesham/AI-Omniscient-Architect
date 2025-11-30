# Omniscient LLM

LLM abstraction layer for the Omniscient Architect platform. Provides a unified interface for multiple LLM providers with automatic fallback, streaming, and model management.

## Installation

```bash
# Base package (no providers)
pip install omniscient-llm

# With Ollama support (local LLM)
pip install omniscient-llm[ollama]

# With OpenAI support
pip install omniscient-llm[openai]

# With Anthropic support
pip install omniscient-llm[anthropic]

# All providers
pip install omniscient-llm[all]
```

## Quick Start

### Using Ollama (Local LLM)

```python
from omniscient_llm import LLMClient, OllamaProvider

# Create client with Ollama
client = LLMClient(
    provider=OllamaProvider(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
    )
)

# Generate completion
response = await client.generate(
    prompt="Analyze this code for potential issues...",
    temperature=0.7,
)

print(response.content)
print(f"Tokens: {response.usage.total_tokens}")
```

### Using OpenAI

```python
from omniscient_llm import LLMClient, OpenAIProvider

client = LLMClient(
    provider=OpenAIProvider(
        model="gpt-4-turbo",
        api_key="sk-...",
    )
)

response = await client.generate(
    prompt="Review this architecture...",
    max_tokens=2000,
)
```

### Multi-Provider with Fallback

```python
from omniscient_llm import LLMClient, ProviderChain

# Try Ollama first, fall back to OpenAI
client = LLMClient(
    provider=ProviderChain([
        OllamaProvider(model="codellama:13b"),
        OpenAIProvider(model="gpt-3.5-turbo"),
    ])
)

# Automatically uses first available provider
response = await client.generate(prompt="...")
```

### Streaming Responses

```python
async for chunk in client.stream(prompt="Explain this code..."):
    print(chunk.content, end="", flush=True)
```

## Model Management

```python
from omniscient_llm import ModelManager

manager = ModelManager()

# List available models
models = await manager.list_models()
for model in models:
    print(f"{model.name}: {model.size_gb}GB, {model.provider}")

# Pull a model (Ollama)
await manager.pull_model("codellama:13b")

# Check model status
status = await manager.get_model_status("llama3.2:latest")
print(f"Loaded: {status.is_loaded}, Memory: {status.memory_usage_mb}MB")
```

## Configuration

### Environment Variables

- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `OMNISCIENT_LLM_TIMEOUT`: Request timeout in seconds (default: 120)

### Config File

```yaml
# llm_config.yaml
default_provider: ollama

providers:
  ollama:
    base_url: http://localhost:11434
    model: llama3.2:latest
    timeout: 120
    
  openai:
    model: gpt-4-turbo
    max_tokens: 4096
    
  anthropic:
    model: claude-3-sonnet
    max_tokens: 4096

fallback_chain:
  - ollama
  - openai
  - anthropic
```

## Provider Features

| Feature | Ollama | OpenAI | Anthropic |
|---------|--------|--------|-----------|
| Streaming | ✓ | ✓ | ✓ |
| Function Calling | ✓ | ✓ | ✓ |
| Vision | ✓ | ✓ | ✓ |
| Local Execution | ✓ | ✗ | ✗ |
| Cost Tracking | N/A | ✓ | ✓ |

## Error Handling

```python
from omniscient_llm import LLMError, ProviderUnavailable, RateLimitError

try:
    response = await client.generate(prompt="...")
except ProviderUnavailable as e:
    print(f"Provider {e.provider} is not available: {e}")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except LLMError as e:
    print(f"LLM error: {e}")
```

## License

MIT License
