"""
Embedding Cache Module.

Provides caching for embeddings to avoid redundant computations.
Supports both in-memory LRU cache and optional persistent storage.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable
from collections import OrderedDict
import asyncio

from omniscient_core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_time_saved_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    embedding: List[float]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class EmbeddingCache:
    """
    LRU cache for embeddings with support for batch operations.
    
    Features:
    - LRU eviction policy
    - Content-based hashing (same text = same key)
    - Model-aware caching (different models have different embeddings)
    - Batch lookup and insertion
    - Performance statistics
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
        avg_embedding_time_ms: float = 300.0,
    ):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for entries (None = no expiry)
            avg_embedding_time_ms: Average time to generate embedding (for stats)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.avg_embedding_time_ms = avg_embedding_time_ms
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
    
    def _make_key(self, text: str, model: str = "nomic-embed-text") -> str:
        """Create a cache key from text and model."""
        # Use SHA-256 hash for consistent, fixed-length keys
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get(
        self,
        text: str,
        model: str = "nomic-embed-text"
    ) -> Optional[List[float]]:
        """
        Get cached embedding if available.
        
        Args:
            text: Text to lookup
            model: Model name
            
        Returns:
            Cached embedding or None
        """
        key = self._make_key(text, model)
        
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds and (time.time() - entry.created_at) > self.ttl_seconds:
                del self._cache[key]
                self._stats.misses += 1
                return None
            
            # Update access info and move to end (LRU)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.total_time_saved_ms += self.avg_embedding_time_ms
            
            return entry.embedding
    
    async def put(
        self,
        text: str,
        embedding: List[float],
        model: str = "nomic-embed-text"
    ) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Original text
            embedding: Embedding vector
            model: Model name
        """
        key = self._make_key(text, model)
        
        async with self._lock:
            # If already exists, update and move to end
            if key in self._cache:
                self._cache[key].embedding = embedding
                self._cache.move_to_end(key)
                return
            
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
            
            # Add new entry
            self._cache[key] = CacheEntry(embedding=embedding)
    
    async def get_batch(
        self,
        texts: List[str],
        model: str = "nomic-embed-text"
    ) -> Dict[int, List[float]]:
        """
        Get cached embeddings for multiple texts.
        
        Args:
            texts: List of texts to lookup
            model: Model name
            
        Returns:
            Dict mapping index -> embedding for cached entries
        """
        results = {}
        
        for i, text in enumerate(texts):
            embedding = await self.get(text, model)
            if embedding is not None:
                results[i] = embedding
        
        return results
    
    async def put_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model: str = "nomic-embed-text"
    ) -> None:
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of original texts
            embeddings: List of embedding vectors
            model: Model name
        """
        for text, embedding in zip(texts, embeddings):
            await self.put(text, embedding, model)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._stats = CacheStats()
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


# Global cache instance
_global_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(
    max_size: int = 10000,
    ttl_seconds: Optional[float] = None,
) -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _global_cache


async def cached_embed(
    text: str,
    embed_fn: Callable[[str], Awaitable[List[float]]],
    model: str = "nomic-embed-text",
    cache: Optional[EmbeddingCache] = None,
) -> List[float]:
    """
    Get embedding with caching.
    
    Args:
        text: Text to embed
        embed_fn: Async function to generate embedding
        model: Model name for cache key
        cache: Cache instance (uses global if None)
        
    Returns:
        Embedding vector
    """
    cache = cache or get_embedding_cache()
    
    # Check cache first
    cached = await cache.get(text, model)
    if cached is not None:
        return cached
    
    # Generate embedding
    embedding = await embed_fn(text)
    
    # Store in cache
    await cache.put(text, embedding, model)
    
    return embedding


async def cached_embed_batch(
    texts: List[str],
    embed_fn: Callable[[str], Awaitable[List[float]]],
    model: str = "nomic-embed-text",
    cache: Optional[EmbeddingCache] = None,
    max_concurrency: int = 5,
) -> List[List[float]]:
    """
    Get embeddings for multiple texts with caching and concurrency.
    
    Args:
        texts: List of texts to embed
        embed_fn: Async function to generate single embedding
        model: Model name for cache key
        cache: Cache instance (uses global if None)
        max_concurrency: Max concurrent embedding requests
        
    Returns:
        List of embedding vectors (same order as input texts)
    """
    cache = cache or get_embedding_cache()
    
    # Check cache for all texts
    cached = await cache.get_batch(texts, model)
    
    # Identify texts that need embedding
    results: List[Optional[List[float]]] = [None] * len(texts)
    texts_to_embed: List[tuple[int, str]] = []
    
    for i, text in enumerate(texts):
        if i in cached:
            results[i] = cached[i]
        else:
            texts_to_embed.append((i, text))
    
    if not texts_to_embed:
        # All texts were cached!
        return results  # type: ignore
    
    logger.debug(
        f"Embedding batch: {len(cached)} cached, {len(texts_to_embed)} to generate"
    )
    
    # Generate missing embeddings with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def embed_with_limit(idx: int, text: str) -> tuple[int, List[float]]:
        async with semaphore:
            embedding = await embed_fn(text)
            return idx, embedding
    
    # Run embeddings concurrently
    tasks = [embed_with_limit(idx, text) for idx, text in texts_to_embed]
    completed = await asyncio.gather(*tasks)
    
    # Store results and cache
    for idx, embedding in completed:
        results[idx] = embedding
        await cache.put(texts[idx], embedding, model)
    
    return results  # type: ignore


class CachedEmbedder:
    """
    Wrapper that adds caching to any embedding function.
    
    Usage:
        embedder = CachedEmbedder(ollama_provider.embed)
        embedding = await embedder.embed("Hello world")
        embeddings = await embedder.embed_batch(["Hello", "World"])
    """
    
    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[List[float]]],
        model: str = "nomic-embed-text",
        cache: Optional[EmbeddingCache] = None,
        max_concurrency: int = 5,
    ):
        """
        Initialize cached embedder.
        
        Args:
            embed_fn: Async function to generate embedding
            model: Model name
            cache: Cache instance (uses global if None)
            max_concurrency: Max concurrent requests for batch
        """
        self.embed_fn = embed_fn
        self.model = model
        self.cache = cache or get_embedding_cache()
        self.max_concurrency = max_concurrency
    
    async def embed(self, text: str) -> List[float]:
        """Embed single text with caching."""
        return await cached_embed(text, self.embed_fn, self.model, self.cache)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with caching and concurrency."""
        return await cached_embed_batch(
            texts, self.embed_fn, self.model, self.cache, self.max_concurrency
        )
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
