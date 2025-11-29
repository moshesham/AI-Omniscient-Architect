"""OpenAI LLM provider."""

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
    ProviderUnavailable,
    RateLimitError,
    AuthenticationError,
    GenerationError,
)

logger = get_logger(__name__)

# Check if openai is available
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = None  # type: ignore


# Cost per 1K tokens (as of 2024)
OPENAI_COSTS = {
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models with
    streaming, function calling, and cost tracking.
    
    Example:
        >>> provider = OpenAIProvider(
        ...     model="gpt-4-turbo",
        ...     api_key="sk-..."
        ... )
        >>> async with provider:
        ...     response = await provider.generate(
        ...         GenerationRequest(prompt="Hello!")
        ...     )
    """
    
    provider_type = ProviderType.OPENAI
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs,
    ):
        """Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., "gpt-4-turbo", "gpt-3.5-turbo")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: Custom API base URL (for Azure, etc.)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install with: pip install omniscient-llm[openai]"
            )
        
        config = LLMConfig(
            model=model,
            provider=ProviderType.OPENAI,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout,
            **kwargs,
        )
        super().__init__(config)
        
        self._organization = organization
        self._client: Optional[AsyncOpenAI] = None
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        import os
        
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "openai",
                "API key required. Set api_key or OPENAI_API_KEY environment variable."
            )
        
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=self._organization,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )
        
        await super().initialize()
        logger.debug(f"OpenAI provider initialized with model {self.model}")
    
    async def close(self) -> None:
        """Close OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
        await super().close()
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple:
        """Calculate cost for token usage."""
        costs = OPENAI_COSTS.get(model, {"input": 0, "output": 0})
        prompt_cost = (prompt_tokens / 1000) * costs["input"]
        completion_cost = (completion_tokens / 1000) * costs["output"]
        return prompt_cost, completion_cost
    
    async def generate(self, request: GenerationRequest) -> LLMResponse:
        """Generate text using OpenAI.
        
        Args:
            request: Generation request
            
        Returns:
            LLMResponse with generated content
        """
        if not self._client:
            await self.initialize()
        
        request = self._apply_defaults(request)
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        start_time = time.time()
        
        try:
            response = await self._client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop_sequences or None,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract usage
            usage_data = response.usage
            prompt_tokens = usage_data.prompt_tokens if usage_data else 0
            completion_tokens = usage_data.completion_tokens if usage_data else 0
            
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
            
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider=ProviderType.OPENAI,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                raw_response=response.model_dump(),
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError("openai", message=str(e))
            if "auth" in error_str or "401" in error_str:
                raise AuthenticationError("openai", str(e))
            raise GenerationError("openai", str(e), e)
    
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
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        try:
            stream = await self._client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop_sequences or None,
                stream=True,
            )
            
            index = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    finish_reason = chunk.choices[0].finish_reason
                    
                    yield StreamChunk(
                        content=content,
                        is_final=finish_reason is not None,
                        index=index,
                        finish_reason=finish_reason,
                    )
                    
                    index += 1
                    
        except Exception as e:
            raise GenerationError("openai", str(e), e)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models.
        
        Returns:
            List of ModelInfo objects
        """
        if not self._client:
            await self.initialize()
        
        try:
            models = await self._client.models.list()  # type: ignore
            
            result = []
            for model in models.data:
                # Only include chat models
                if "gpt" in model.id.lower():
                    result.append(ModelInfo(
                        name=model.id,
                        provider=ProviderType.OPENAI,
                        supports_streaming=True,
                        supports_functions=True,
                        supports_vision="vision" in model.id.lower() or "4o" in model.id.lower(),
                    ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []
