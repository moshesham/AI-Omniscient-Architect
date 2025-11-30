"""Omniscient LLM - Multi-provider LLM abstraction layer.

This package provides a unified interface for multiple LLM providers
with automatic fallback, streaming, and model management.

Supported providers:
- Ollama (local LLM)
- OpenAI
- Anthropic

Example:
    >>> from omniscient_llm import LLMClient, OllamaProvider
    >>> client = LLMClient(provider=OllamaProvider(model="llama3.2:latest"))
    >>> response = await client.generate("Analyze this code...")
"""

from omniscient_llm.models import (
    LLMConfig,
    LLMResponse,
    LLMUsage,
    ModelInfo,
    ModelStatus,
    ProviderType,
    Role,
    ChatMessage,
    GenerationRequest,
    StreamChunk,
)
from omniscient_llm.client import LLMClient
from omniscient_llm.base import BaseLLMProvider
from omniscient_llm.errors import (
    LLMError,
    ProviderUnavailable,
    RateLimitError,
    ModelNotFoundError,
    ModelNotFound,
    TokenLimitError,
)

# Lazy imports for optional providers
def get_ollama_provider():
    """Get Ollama provider (requires ollama extra)."""
    from omniscient_llm.providers.ollama import OllamaProvider
    return OllamaProvider

def get_openai_provider():
    """Get OpenAI provider (requires openai extra)."""
    from omniscient_llm.providers.openai import OpenAIProvider
    return OpenAIProvider

def get_anthropic_provider():
    """Get Anthropic provider (requires anthropic extra)."""
    from omniscient_llm.providers.anthropic import AnthropicProvider
    return AnthropicProvider

# Try to import providers if available
try:
    from omniscient_llm.providers.ollama import OllamaProvider
except ImportError:
    OllamaProvider = None  # type: ignore

try:
    from omniscient_llm.providers.openai import OpenAIProvider
except ImportError:
    OpenAIProvider = None  # type: ignore

try:
    from omniscient_llm.providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicProvider = None  # type: ignore

from omniscient_llm.chain import ProviderChain
from omniscient_llm.manager import ModelManager

__all__ = [
    # Client
    "LLMClient",
    # Base
    "BaseLLMProvider",
    # Models
    "LLMConfig",
    "LLMResponse",
    "LLMUsage",
    "ModelInfo",
    "ModelStatus",
    "ProviderType",
    "Role",
    "ChatMessage",
    "GenerationRequest",
    "StreamChunk",
    # Providers
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    # Chain
    "ProviderChain",
    # Manager
    "ModelManager",
    # Errors
    "LLMError",
    "ProviderUnavailable",
    "RateLimitError",
    "ModelNotFoundError",
    "ModelNotFound",
    "TokenLimitError",
    # Helpers
    "get_ollama_provider",
    "get_openai_provider",
    "get_anthropic_provider",
]

__version__ = "0.1.0"
