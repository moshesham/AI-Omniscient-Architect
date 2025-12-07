"""LLM Client - Main interface for LLM operations."""

import asyncio
import time
from typing import AsyncIterator, Optional, List

from omniscient_core.logging import get_logger
from omniscient_core import AsyncContextMixin

from .base import BaseLLMProvider
from .models import (
    LLMConfig,
    LLMResponse,
    StreamChunk,
    GenerationRequest,
    ModelInfo,
)
from .errors import ProviderUnavailable, LLMError

logger = get_logger(__name__)


class LLMClient(AsyncContextMixin):
    """High-level client for LLM operations.
    
    Provides a unified interface for text generation with support for
    multiple providers, streaming, and automatic retries.
    
    Example:
        >>> from omniscient_llm import LLMClient, OllamaProvider
        >>> client = LLMClient(provider=OllamaProvider(model="llama3.2:latest"))
        >>> response = await client.generate("Analyze this code...")
        >>> print(response.content)
    """
    
    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        config: Optional[LLMConfig] = None,
    ):
        """Initialize LLM client.
        
        Args:
            provider: LLM provider instance
            config: Optional configuration override
        """
        self.provider = provider
        self.config = config or (provider.config if provider else LLMConfig())
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the client and provider."""
        if self.provider and not self.provider._initialized:
            await self.provider.initialize()
        self._initialized = True
    
    async def close(self) -> None:
        """Close the client and release resources."""
        if self.provider:
            await self.provider.close()
        self._initialized = False
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from prompt.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system/context prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            **kwargs: Additional provider-specific options
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            ProviderUnavailable: If no provider is configured
            LLMError: If generation fails
        """
        if not self.provider:
            raise ProviderUnavailable("none", "No provider configured")
        
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Build request
        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences or [],
            metadata=kwargs,
        )
        
        # Generate with timing
        start_time = time.time()
        
        try:
            response = await self.provider.generate(request)
            response.latency_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                "LLM generation complete",
                model=response.model,
                tokens=response.usage.total_tokens,
                latency_ms=response.latency_ms,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system/context prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options
            
        Yields:
            StreamChunk objects with content fragments
        """
        if not self.provider:
            raise ProviderUnavailable("none", "No provider configured")
        
        if not self._initialized:
            await self.initialize()
        
        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=kwargs,
        )
        
        async for chunk in self.provider.stream(request):
            yield chunk
    
    async def generate_with_retry(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate with automatic retry on failure.
        
        Args:
            prompt: The input prompt
            max_retries: Maximum retry attempts
            **kwargs: Additional generation options
            
        Returns:
            LLMResponse with generated content
        """
        retries = max_retries or self.config.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                return await self.generate(prompt, **kwargs)
            except LLMError as e:
                last_error = e
                if attempt < retries:
                    delay = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Generation failed, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{retries + 1})"
                    )
                    await asyncio.sleep(delay)
        
        # After all retries exhausted, last_error should be set
        assert last_error is not None, "Loop should have caught at least one error"
        raise last_error
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from the provider.
        
        Returns:
            List of ModelInfo objects
        """
        if not self.provider:
            return []
        
        if not self._initialized:
            await self.initialize()
        
        return await self.provider.list_models()
    
    async def check_health(self) -> bool:
        """Check if the provider is healthy.
        
        Returns:
            True if provider is responding
        """
        if not self.provider:
            return False
        
        try:
            return await self.provider.check_health()
        except Exception:
            return False
    
    def set_provider(self, provider: BaseLLMProvider) -> None:
        """Set the provider.
        
        Args:
            provider: New provider instance
        """
        self.provider = provider
        self._initialized = False
    
    @property
    def model(self) -> Optional[str]:
        """Get current model name."""
        return self.provider.model if self.provider else None
    
    @property
    def provider_type(self) -> Optional[str]:
        """Get current provider type."""
        return self.provider.provider_type.value if self.provider else None
