"""Ollama LLM provider for local model execution."""

import asyncio
import time
from typing import AsyncIterator, List, Optional
import httpx

from omniscient_core.logging import get_logger

from ..base import BaseLLMProvider
from ..models import (
    LLMConfig,
    LLMResponse,
    LLMUsage,
    StreamChunk,
    ModelInfo,
    ModelStatus,
    GenerationRequest,
    ProviderType,
)
from ..errors import (
    ModelNotFoundError,
    GenerationError,
    TimeoutError,
)

logger = get_logger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM execution.
    
    Uses the Ollama API to run LLMs locally. Supports model management,
    streaming, and all Ollama-compatible models.
    
    Example:
        >>> provider = OllamaProvider(model="llama3.2:latest")
        >>> async with provider:
        ...     response = await provider.generate(
        ...         GenerationRequest(prompt="Hello!")
        ...     )
    """
    
    provider_type = ProviderType.OLLAMA
    
    def __init__(
        self,
        model: str = "llama3.2:latest",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs,
    ):
        """Initialize Ollama provider.
        
        Args:
            model: Model name (e.g., "llama3.2:latest", "codellama:13b")
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        config = LLMConfig(
            model=model,
            provider=ProviderType.OLLAMA,
            base_url=base_url or DEFAULT_OLLAMA_URL,
            timeout_seconds=timeout,
            **kwargs,
        )
        super().__init__(config)
        
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def base_url(self) -> str:
        """Get Ollama base URL."""
        return self.config.base_url or DEFAULT_OLLAMA_URL
    
    async def initialize(self) -> None:
        """Initialize HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_seconds,
                    read=self.config.timeout_seconds,
                    write=self.config.timeout_seconds,
                    pool=self.config.timeout_seconds,
                ),
            )
        await super().initialize()
        logger.debug(f"Ollama provider initialized at {self.base_url}")
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        await super().close()
    
    @property
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        # Quick sync check
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    async def check_health(self) -> bool:
        """Check Ollama health."""
        try:
            if not self._client:
                await self.initialize()
            response = await self._client.get("/api/tags")  # type: ignore
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate(self, request: GenerationRequest) -> LLMResponse:
        """Generate text using Ollama.
        
        Args:
            request: Generation request
            
        Returns:
            LLMResponse with generated content
        """
        if not self._client:
            await self.initialize()
        
        request = self._apply_defaults(request)
        
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
                "top_k": request.top_k,
            },
        }
        
        if request.system_prompt:
            payload["system"] = request.system_prompt
        
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        
        start_time = time.time()
        
        try:
            response = await self._client.post("/api/generate", json=payload)  # type: ignore
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract usage info
            usage = LLMUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=(
                    data.get("prompt_eval_count", 0) + 
                    data.get("eval_count", 0)
                ),
            )
            
            return LLMResponse(
                content=data.get("response", ""),
                model=self.model,
                provider=ProviderType.OLLAMA,
                usage=usage,
                finish_reason="stop" if data.get("done") else None,
                latency_ms=latency_ms,
                raw_response=data,
            )
            
        except httpx.TimeoutException:
            raise TimeoutError("ollama", self.config.timeout_seconds)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(self.model, "ollama")
            raise GenerationError("ollama", str(e), e)
        except Exception as e:
            raise GenerationError("ollama", str(e), e)
    
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
        
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
                "top_k": request.top_k,
            },
        }
        
        if request.system_prompt:
            payload["system"] = request.system_prompt
        
        try:
            async with self._client.stream("POST", "/api/generate", json=payload) as response:  # type: ignore
                response.raise_for_status()
                
                index = 0
                total_prompt_tokens = 0
                total_completion_tokens = 0
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    import json
                    data = json.loads(line)
                    
                    content = data.get("response", "")
                    is_done = data.get("done", False)
                    
                    # Track tokens
                    if "prompt_eval_count" in data:
                        total_prompt_tokens = data["prompt_eval_count"]
                    if "eval_count" in data:
                        total_completion_tokens = data["eval_count"]
                    
                    usage = None
                    if is_done:
                        usage = LLMUsage(
                            prompt_tokens=total_prompt_tokens,
                            completion_tokens=total_completion_tokens,
                            total_tokens=total_prompt_tokens + total_completion_tokens,
                        )
                    
                    yield StreamChunk(
                        content=content,
                        is_final=is_done,
                        index=index,
                        usage=usage,
                        finish_reason="stop" if is_done else None,
                    )
                    
                    index += 1
                    
        except httpx.TimeoutException:
            raise TimeoutError("ollama", self.config.timeout_seconds)
        except Exception as e:
            raise GenerationError("ollama", str(e), e)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Ollama models.
        
        Returns:
            List of ModelInfo objects
        """
        if not self._client:
            await self.initialize()
        
        try:
            response = await self._client.get("/api/tags")  # type: ignore
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model_data in data.get("models", []):
                name = model_data.get("name", "")
                
                # Parse model info
                details = model_data.get("details", {})
                
                models.append(ModelInfo(
                    name=name,
                    provider=ProviderType.OLLAMA,
                    size_bytes=model_data.get("size"),
                    parameter_count=details.get("parameter_size"),
                    context_length=None,  # Not provided by Ollama API
                    family=details.get("family"),
                    quantization=details.get("quantization_level"),
                    modified_at=model_data.get("modified_at"),
                    supports_streaming=True,
                    supports_functions=False,  # Ollama doesn't support function calling yet
                    supports_vision="vision" in name.lower() or "llava" in name.lower(),
                ))
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def get_model_status(self, model_name: str) -> ModelStatus:
        """Get status of a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelStatus with current state
        """
        if not self._client:
            await self.initialize()
        
        try:
            response = await self._client.post(  # type: ignore
                "/api/show",
                json={"name": model_name}
            )
            
            if response.status_code == 404:
                return ModelStatus(name=model_name, is_loaded=False)
            
            response.raise_for_status()
            
            return ModelStatus(
                name=model_name,
                is_loaded=True,
            )
            
        except Exception as e:
            return ModelStatus(name=model_name, is_loaded=False, error=str(e))
    
    async def pull_model(self, model_name: str) -> None:
        """Pull/download a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
        """
        if not self._client:
            await self.initialize()
        
        logger.info(f"Pulling Ollama model: {model_name}")
        
        try:
            # Use a longer timeout for pulling
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(connect=10.0, read=3600.0, write=3600.0, pool=3600.0),
            ) as client:
                async with client.stream(
                    "POST",
                    "/api/pull",
                    json={"name": model_name, "stream": True}
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            status = data.get("status", "")
                            
                            if "error" in data:
                                raise GenerationError("ollama", data["error"])
                            
                            # Log progress
                            if "completed" in data and "total" in data:
                                progress = (data["completed"] / data["total"]) * 100
                                logger.debug(f"Pull progress: {status} ({progress:.1f}%)")
                            elif status:
                                logger.debug(f"Pull status: {status}")
            
            logger.info(f"Successfully pulled model: {model_name}")
            
        except Exception as e:
            raise GenerationError("ollama", f"Failed to pull model: {e}", e)
    
    # =========================================================================
    # Embedding Support for RAG
    # =========================================================================
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Embedding model (default: nomic-embed-text)
            
        Returns:
            List of floats (embedding vector)
        """
        if not self._client:
            await self.initialize()
        
        embed_model = model or "nomic-embed-text"
        
        try:
            response = await self._client.post(  # type: ignore
                "/api/embeddings",
                json={
                    "model": embed_model,
                    "prompt": text,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get("embedding", [])
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(embed_model, "ollama")
            raise GenerationError("ollama", f"Embedding failed: {e}", e)
        except Exception as e:
            raise GenerationError("ollama", f"Embedding failed: {e}", e)
    
    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 32,
        max_concurrency: int = 5,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with concurrent processing.
        
        Args:
            texts: List of texts to embed
            model: Embedding model (default: nomic-embed-text)
            batch_size: Number of texts per batch
            max_concurrency: Max concurrent embedding requests
            
        Returns:
            List of embedding vectors
        """
        if not self._client:
            await self.initialize()
        
        embed_model = model or "nomic-embed-text"
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        
        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def embed_with_index(idx: int, text: str) -> tuple[int, List[float]]:
            """Embed single text with concurrency limit."""
            async with semaphore:
                embedding = await self.embed(text, embed_model)
                return idx, embedding
        
        # Process in batches with concurrent embedding within each batch
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_indices = list(range(i, i + len(batch)))
            
            # Create tasks for concurrent embedding
            tasks = [
                embed_with_index(idx, text) 
                for idx, text in zip(batch_indices, batch)
            ]
            
            # Run batch concurrently
            results = await asyncio.gather(*tasks)
            
            # Store results in correct order
            for idx, embedding in results:
                embeddings[idx] = embedding
            
            logger.debug(f"Embedded batch {i//batch_size + 1}, total: {i + len(batch)}/{len(texts)}")
        
        return [e for e in embeddings if e is not None]  # type: ignore
    
    async def get_embedding_models(self) -> List[str]:
        """Get list of available embedding models.
        
        Returns:
            List of model names that support embeddings
        """
        models = await self.list_models()
        
        # Known embedding models in Ollama
        embedding_models = []
        embedding_keywords = ['embed', 'nomic', 'mxbai', 'all-minilm', 'bge']
        
        for model in models:
            name_lower = model.name.lower()
            if any(kw in name_lower for kw in embedding_keywords):
                embedding_models.append(model.name)
        
        return embedding_models
