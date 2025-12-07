"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List

from omniscient_core import AsyncContextMixin

from .models import (
    LLMConfig,
    LLMResponse,
    StreamChunk,
    ModelInfo,
    ModelStatus,
    GenerationRequest,
    ProviderType,
)


class BaseLLMProvider(AsyncContextMixin, ABC):
    """Abstract base class for LLM providers.
    
    Implement this class to add support for new LLM providers.
    
    Example:
        class MyProvider(BaseLLMProvider):
            provider_type = ProviderType.CUSTOM
            
            async def generate(self, request):
                # Implementation
                pass
    """
    
    provider_type: ProviderType = ProviderType.CUSTOM
    
    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """Initialize provider.
        
        Args:
            config: LLM configuration
            **kwargs: Additional provider-specific options
        """
        self.config = config or LLMConfig(**kwargs)
        self._initialized = False
    
    @property
    def model(self) -> str:
        """Get the model name."""
        return self.config.model
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available.
        
        Override to implement connectivity checks.
        """
        return True
    
    async def initialize(self) -> None:
        """Initialize the provider.
        
        Called before first use. Override to perform setup.
        """
        self._initialized = True
    
    async def close(self) -> None:
        """Close the provider and release resources.
        
        Override to clean up connections, etc.
        """
        self._initialized = False
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> LLMResponse:
        """Generate text from prompt.
        
        Args:
            request: Generation request with prompt and parameters
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation.
        
        Args:
            request: Generation request with prompt and parameters
            
        Yields:
            StreamChunk objects with content fragments
        """
        pass
    
    async def check_health(self) -> bool:
        """Check if provider is healthy.
        
        Returns:
            True if provider is responding correctly
        """
        try:
            # Simple health check - try to list models
            await self.list_models()
            return True
        except Exception:
            return False
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models.
        
        Returns:
            List of available models
        """
        return []
    
    async def get_model_status(self, model_name: str) -> ModelStatus:
        """Get status of a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelStatus with current state
        """
        return ModelStatus(name=model_name, is_loaded=False)
    
    async def pull_model(self, model_name: str) -> None:
        """Pull/download a model.
        
        Args:
            model_name: Name of the model to pull
            
        Raises:
            NotImplementedError: If provider doesn't support model pulling
        """
        raise NotImplementedError(
            f"Provider {self.provider_type.value} doesn't support model pulling"
        )
    
    def _apply_defaults(self, request: GenerationRequest) -> GenerationRequest:
        """Apply default values from config to request.
        
        Args:
            request: Original request
            
        Returns:
            Request with defaults applied
        """
        if request.temperature is None:
            request.temperature = self.config.temperature
        if request.max_tokens is None:
            request.max_tokens = self.config.max_tokens
        if request.top_p is None:
            request.top_p = self.config.top_p
        if request.top_k is None:
            request.top_k = self.config.top_k
        return request
