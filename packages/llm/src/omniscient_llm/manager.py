"""Model manager for local LLM models via Ollama."""

import asyncio
from typing import Callable, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from omniscient_core.logging import get_logger

from .providers.ollama import OllamaProvider
from .models import ModelInfo
from .errors import ModelNotFound, LLMError

logger = get_logger(__name__)


@dataclass
class PullProgress:
    """Progress update during model pull."""
    
    status: str
    digest: Optional[str] = None
    total: int = 0
    completed: int = 0
    
    @property
    def percent(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    @property
    def size_mb(self) -> float:
        """Get total size in MB."""
        return self.total / (1024 * 1024)
    
    @property
    def downloaded_mb(self) -> float:
        """Get downloaded size in MB."""
        return self.completed / (1024 * 1024)


class ModelManager:
    """Manage local LLM models via Ollama.
    
    Provides functionality for:
    - Listing available models
    - Pulling new models
    - Removing models
    - Checking model status
    
    Example:
        >>> manager = ModelManager()
        >>> await manager.initialize()
        >>> 
        >>> # List models
        >>> models = await manager.list_models()
        >>> 
        >>> # Pull a model with progress
        >>> async for progress in manager.pull_model("llama3.2:latest"):
        ...     print(f"{progress.status}: {progress.percent:.1f}%")
        >>> 
        >>> await manager.close()
    """
    
    # Popular model recommendations
    RECOMMENDED_MODELS = {
        "code": [
            "codellama:13b",
            "codellama:7b",
            "deepseek-coder:6.7b",
            "codegemma:7b",
        ],
        "general": [
            "llama3.2:latest",
            "llama3.1:8b",
            "mistral:latest",
            "gemma2:9b",
        ],
        "small": [
            "llama3.2:3b",
            "phi3:mini",
            "tinyllama:latest",
            "qwen2:1.5b",
        ],
        "analysis": [
            "llama3.1:8b",
            "mistral:latest",
            "neural-chat:7b",
        ],
    }
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 600.0,  # Long timeout for pulls
    ):
        """Initialize model manager.
        
        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self._provider: Optional[OllamaProvider] = None
    
    async def initialize(self) -> None:
        """Initialize connection to Ollama."""
        self._provider = OllamaProvider(
            model="",  # No default model needed
            base_url=self.base_url,
            timeout=self.timeout,
        )
        await self._provider.initialize()
        
        if not await self._provider.check_health():
            raise LLMError("Ollama server not available")
    
    async def close(self) -> None:
        """Close connection."""
        if self._provider:
            await self._provider.close()
    
    async def __aenter__(self) -> "ModelManager":
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def is_available(self) -> bool:
        """Check if Ollama is available.
        
        Returns:
            True if server is running and responsive
        """
        if not self._provider:
            return False
        return await self._provider.check_health()
    
    async def list_models(self) -> list[ModelInfo]:
        """List all available local models.
        
        Returns:
            List of ModelInfo objects
        """
        if not self._provider:
            raise LLMError("Manager not initialized")
        return await self._provider.list_models()
    
    async def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get info for a specific model.
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo if found, None otherwise
        """
        models = await self.list_models()
        for model in models:
            if model.name == name or model.name.startswith(f"{name}:"):
                return model
        return None
    
    async def has_model(self, name: str) -> bool:
        """Check if a model is available locally.
        
        Args:
            name: Model name
            
        Returns:
            True if model exists
        """
        return await self.get_model(name) is not None
    
    async def pull_model(
        self,
        name: str,
        insecure: bool = False,
    ) -> AsyncIterator[PullProgress]:
        """Pull a model from registry with progress updates.
        
        Args:
            name: Model name to pull
            insecure: Allow insecure connections
            
        Yields:
            PullProgress updates
        """
        if not self._provider or not self._provider._client:
            raise LLMError("Manager not initialized")
        
        logger.info(f"Pulling model: {name}")
        
        url = f"{self.base_url}/api/pull"
        
        async with self._provider._client.stream(
            "POST",
            url,
            json={"name": name, "insecure": insecure},
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                import json
                data = json.loads(line)
                
                yield PullProgress(
                    status=data.get("status", ""),
                    digest=data.get("digest"),
                    total=data.get("total", 0),
                    completed=data.get("completed", 0),
                )
        
        logger.info(f"Model {name} pulled successfully")
    
    async def delete_model(self, name: str) -> bool:
        """Delete a local model.
        
        Args:
            name: Model name to delete
            
        Returns:
            True if deleted successfully
        """
        if not self._provider:
            raise LLMError("Manager not initialized")
        
        await self._provider.delete_model(name)
        return True
    
    async def copy_model(self, source: str, destination: str) -> bool:
        """Copy/rename a model.
        
        Args:
            source: Source model name
            destination: Destination model name
            
        Returns:
            True if copied successfully
        """
        if not self._provider or not self._provider._client:
            raise LLMError("Manager not initialized")
        
        response = await self._provider._client.post(
            f"{self.base_url}/api/copy",
            json={"source": source, "destination": destination},
        )
        
        return response.status_code == 200
    
    async def get_model_info(self, name: str) -> dict:
        """Get detailed model information.
        
        Args:
            name: Model name
            
        Returns:
            Model details including parameters, template, etc.
        """
        if not self._provider or not self._provider._client:
            raise LLMError("Manager not initialized")
        
        response = await self._provider._client.post(
            f"{self.base_url}/api/show",
            json={"name": name},
        )
        
        if response.status_code == 404:
            raise ModelNotFound(name)
        
        response.raise_for_status()
        return response.json()
    
    def get_recommendations(
        self,
        category: str = "general",
        max_size_gb: Optional[float] = None,
    ) -> list[str]:
        """Get model recommendations for a category.
        
        Args:
            category: Model category (code, general, small, analysis)
            max_size_gb: Maximum model size in GB (approximate)
            
        Returns:
            List of recommended model names
        """
        models = self.RECOMMENDED_MODELS.get(category, [])
        
        if max_size_gb is not None:
            # Filter by size hint in model name
            # This is approximate based on common naming conventions
            filtered = []
            for model in models:
                # Extract size hint from name if present
                if ":3b" in model or ":1.5b" in model or "mini" in model:
                    if max_size_gb >= 2:
                        filtered.append(model)
                elif ":7b" in model or ":6.7b" in model:
                    if max_size_gb >= 5:
                        filtered.append(model)
                elif ":8b" in model or ":9b" in model:
                    if max_size_gb >= 6:
                        filtered.append(model)
                elif ":13b" in model:
                    if max_size_gb >= 10:
                        filtered.append(model)
                else:
                    # Unknown size, include if reasonably large limit
                    if max_size_gb >= 5:
                        filtered.append(model)
            
            return filtered
        
        return models.copy()
    
    async def ensure_model(
        self,
        name: str,
        progress_callback: Optional[Callable[[PullProgress], None]] = None,
    ) -> ModelInfo:
        """Ensure a model is available, pulling if necessary.
        
        Args:
            name: Model name
            progress_callback: Optional callback for pull progress
            
        Returns:
            ModelInfo for the model
        """
        # Check if already available
        model = await self.get_model(name)
        if model:
            return model
        
        # Pull the model
        logger.info(f"Model {name} not found, pulling...")
        
        async for progress in self.pull_model(name):
            if progress_callback:
                progress_callback(progress)
        
        # Get model info after pull
        model = await self.get_model(name)
        if not model:
            raise ModelNotFound(name)
        
        return model


async def print_pull_progress(name: str, manager: ModelManager) -> None:
    """Helper to print pull progress to console.
    
    Args:
        name: Model name to pull
        manager: ModelManager instance
    """
    last_percent = -1
    
    async for progress in manager.pull_model(name):
        if progress.total > 0:
            percent = int(progress.percent)
            if percent != last_percent:
                last_percent = percent
                print(
                    f"\r{progress.status}: "
                    f"{progress.downloaded_mb:.1f}/{progress.size_mb:.1f} MB "
                    f"({percent}%)",
                    end="",
                    flush=True,
                )
        else:
            print(f"\r{progress.status}...", end="", flush=True)
    
    print("\nDone!")
