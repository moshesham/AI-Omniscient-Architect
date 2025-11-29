"""Anthropic LLM provider."""

import time
from typing import AsyncIterator, List, Optional

from omniscient_core.logging import get_logger

from ..base import BaseLLMProvider
from ..models import (
    LLMConfig,
    LLMResponse,
    LLMUsage,
    StreamChunk,
    ModelInfo,
    GenerationRequest,
    ProviderType,
)
from ..errors import (
    RateLimitError,
    AuthenticationError,
    GenerationError,
)

logger = get_logger(__name__)

# Check if anthropic is available
try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AsyncAnthropic = None  # type: ignore


# Cost per 1K tokens (as of 2024)
ANTHROPIC_COSTS = {
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-2.1": {"input": 0.008, "output": 0.024},
    "claude-2.0": {"input": 0.008, "output": 0.024},
}


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider.
    
    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 2 models with
    streaming and cost tracking.
    
    Example:
        >>> provider = AnthropicProvider(
        ...     model="claude-3-sonnet-20240229",
        ...     api_key="sk-ant-..."
        ... )
        >>> async with provider:
        ...     response = await provider.generate(
        ...         GenerationRequest(prompt="Hello!")
        ...     )
    """
    
    provider_type = ProviderType.ANTHROPIC
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs,
    ):
        """Initialize Anthropic provider.
        
        Args:
            model: Model name (e.g., "claude-3-sonnet-20240229")
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install with: pip install omniscient-llm[anthropic]"
            )
        
        config = LLMConfig(
            model=model,
            provider=ProviderType.ANTHROPIC,
            api_key=api_key,
            timeout_seconds=timeout,
            **kwargs,
        )
        super().__init__(config)
        
        self._client: Optional[AsyncAnthropic] = None
    
    async def initialize(self) -> None:
        """Initialize Anthropic client."""
        import os
        
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "anthropic",
                "API key required. Set api_key or ANTHROPIC_API_KEY environment variable."
            )
        
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=self.config.timeout_seconds,
        )
        
        await super().initialize()
        logger.debug(f"Anthropic provider initialized with model {self.model}")
    
    async def close(self) -> None:
        """Close Anthropic client."""
        if self._client:
            await self._client.close()
            self._client = None
        await super().close()
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple:
        """Calculate cost for token usage."""
        costs = ANTHROPIC_COSTS.get(model, {"input": 0, "output": 0})
        prompt_cost = (prompt_tokens / 1000) * costs["input"]
        completion_cost = (completion_tokens / 1000) * costs["output"]
        return prompt_cost, completion_cost
    
    async def generate(self, request: GenerationRequest) -> LLMResponse:
        """Generate text using Anthropic.
        
        Args:
            request: Generation request
            
        Returns:
            LLMResponse with generated content
        """
        if not self._client:
            await self.initialize()
        
        request = self._apply_defaults(request)
        
        messages = [{"role": "user", "content": request.prompt}]
        
        start_time = time.time()
        
        try:
            response = await self._client.messages.create(  # type: ignore
                model=self.model,
                messages=messages,
                system=request.system_prompt or "",
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences or None,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            
            # Calculate cost
            prompt_cost, completion_cost = self._calculate_cost(
                self.model, prompt_tokens, completion_tokens
            )
            
            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost,
            )
            
            # Extract content
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider=ProviderType.ANTHROPIC,
                usage=usage,
                finish_reason=response.stop_reason,
                latency_ms=latency_ms,
                raw_response=response.model_dump(),
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError("anthropic", message=str(e))
            if "auth" in error_str or "401" in error_str:
                raise AuthenticationError("anthropic", str(e))
            raise GenerationError("anthropic", str(e), e)
    
    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation.
        
        Args:
            request: Generation request
            
        Yields:
            StreamChunk objects
        """
        if not self._client:
            await self.initialize()
        
        request = self._apply_defaults(request)
        
        messages = [{"role": "user", "content": request.prompt}]
        
        try:
            async with self._client.messages.stream(  # type: ignore
                model=self.model,
                messages=messages,
                system=request.system_prompt or "",
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences or None,
            ) as stream:
                index = 0
                async for text in stream.text_stream:
                    yield StreamChunk(
                        content=text,
                        is_final=False,
                        index=index,
                    )
                    index += 1
                
                # Final chunk with usage
                message = await stream.get_final_message()
                
                prompt_cost, completion_cost = self._calculate_cost(
                    self.model,
                    message.usage.input_tokens,
                    message.usage.output_tokens,
                )
                
                usage = LLMUsage(
                    prompt_tokens=message.usage.input_tokens,
                    completion_tokens=message.usage.output_tokens,
                    total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                    prompt_cost=prompt_cost,
                    completion_cost=completion_cost,
                    total_cost=prompt_cost + completion_cost,
                )
                
                yield StreamChunk(
                    content="",
                    is_final=True,
                    index=index,
                    usage=usage,
                    finish_reason=message.stop_reason,
                )
                    
        except Exception as e:
            raise GenerationError("anthropic", str(e), e)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Anthropic models.
        
        Returns:
            List of ModelInfo objects (static list)
        """
        # Anthropic doesn't have a models API, so we return known models
        return [
            ModelInfo(
                name="claude-3-opus-20240229",
                provider=ProviderType.ANTHROPIC,
                context_length=200000,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
            ),
            ModelInfo(
                name="claude-3-sonnet-20240229",
                provider=ProviderType.ANTHROPIC,
                context_length=200000,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
            ),
            ModelInfo(
                name="claude-3-haiku-20240307",
                provider=ProviderType.ANTHROPIC,
                context_length=200000,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
            ),
            ModelInfo(
                name="claude-2.1",
                provider=ProviderType.ANTHROPIC,
                context_length=100000,
                supports_streaming=True,
                supports_functions=False,
                supports_vision=False,
            ),
        ]
