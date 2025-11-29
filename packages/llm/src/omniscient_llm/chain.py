"""Provider chain for fallback support."""

from typing import AsyncIterator, List, Optional

from omniscient_core.logging import get_logger

from .base import BaseLLMProvider
from .models import (
    LLMResponse,
    StreamChunk,
    ModelInfo,
    GenerationRequest,
    ProviderType,
)
from .errors import ProviderUnavailable, LLMError

logger = get_logger(__name__)


class ProviderChain(BaseLLMProvider):
    """Chain of providers with automatic fallback.
    
    Tries providers in order, falling back to the next one if the
    current one fails or is unavailable.
    
    Example:
        >>> chain = ProviderChain([
        ...     OllamaProvider(model="llama3.2:latest"),
        ...     OpenAIProvider(model="gpt-3.5-turbo"),
        ... ])
        >>> async with chain:
        ...     response = await chain.generate(request)
    """
    
    provider_type = ProviderType.CUSTOM
    
    def __init__(
        self,
        providers: List[BaseLLMProvider],
        retry_all: bool = True,
    ):
        """Initialize provider chain.
        
        Args:
            providers: List of providers in priority order
            retry_all: If True, retry from beginning on all errors
        """
        super().__init__()
        
        if not providers:
            raise ValueError("At least one provider required")
        
        self.providers = providers
        self.retry_all = retry_all
        self._active_provider: Optional[BaseLLMProvider] = None
    
    @property
    def model(self) -> str:
        """Get model from active or first provider."""
        if self._active_provider:
            return self._active_provider.model
        return self.providers[0].model
    
    async def initialize(self) -> None:
        """Initialize all providers."""
        for provider in self.providers:
            try:
                await provider.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize {provider.provider_type}: {e}")
        await super().initialize()
    
    async def close(self) -> None:
        """Close all providers."""
        for provider in self.providers:
            try:
                await provider.close()
            except Exception:
                pass
        await super().close()
    
    async def _get_available_provider(self) -> BaseLLMProvider:
        """Find first available provider.
        
        Returns:
            First provider that passes health check
            
        Raises:
            ProviderUnavailable: If no providers are available
        """
        for provider in self.providers:
            try:
                if await provider.check_health():
                    self._active_provider = provider
                    return provider
            except Exception:
                continue
        
        raise ProviderUnavailable(
            "chain",
            "No providers available in chain"
        )
    
    async def generate(self, request: GenerationRequest) -> LLMResponse:
        """Generate using first available provider.
        
        Args:
            request: Generation request
            
        Returns:
            LLMResponse from successful provider
        """
        errors = []
        
        for provider in self.providers:
            try:
                if not provider._initialized:
                    await provider.initialize()
                
                response = await provider.generate(request)
                self._active_provider = provider
                
                logger.debug(
                    f"Generation successful with {provider.provider_type.value}"
                )
                
                return response
                
            except LLMError as e:
                errors.append(f"{provider.provider_type.value}: {e}")
                logger.warning(
                    f"Provider {provider.provider_type.value} failed: {e}"
                )
                
                if not self.retry_all:
                    raise
                continue
            
            except Exception as e:
                errors.append(f"{provider.provider_type.value}: {e}")
                logger.warning(
                    f"Provider {provider.provider_type.value} error: {e}"
                )
                continue
        
        raise ProviderUnavailable(
            "chain",
            f"All providers failed: {'; '.join(errors)}"
        )
    
    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Stream using first available provider.
        
        Args:
            request: Generation request
            
        Yields:
            StreamChunk objects
        """
        # Find available provider
        provider = await self._get_available_provider()
        
        async for chunk in provider.stream(request):
            yield chunk
    
    async def list_models(self) -> List[ModelInfo]:
        """List models from all providers.
        
        Returns:
            Combined list of models from all providers
        """
        all_models = []
        
        for provider in self.providers:
            try:
                models = await provider.list_models()
                all_models.extend(models)
            except Exception:
                continue
        
        return all_models
    
    async def check_health(self) -> bool:
        """Check if any provider is healthy.
        
        Returns:
            True if at least one provider is healthy
        """
        for provider in self.providers:
            try:
                if await provider.check_health():
                    return True
            except Exception:
                continue
        return False
    
    def get_provider_status(self) -> dict:
        """Get status of all providers in chain.
        
        Returns:
            Dict with provider statuses
        """
        return {
            provider.provider_type.value: {
                "initialized": provider._initialized,
                "model": provider.model,
                "active": provider == self._active_provider,
            }
            for provider in self.providers
        }
