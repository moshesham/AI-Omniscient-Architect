"""Analysis cache utilities."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Cache for analysis results with file-based persistence."""

    def __init__(self, cache_dir: Path, ttl: Optional[int] = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live for cache entries in seconds (None = no expiration)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self._initialize_cache_dir()

    def _initialize_cache_dir(self) -> None:
        """Create cache directory structure if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory initialized at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize cache directory: {e}")
            raise

    def _get_cache_key_hash(self, key: str) -> str:
        """Generate a safe filesystem hash from a cache key.

        Args:
            key: The cache key to hash

        Returns:
            A hexadecimal hash string safe for use as a filename
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            key: The cache key

        Returns:
            Path to the cache file
        """
        key_hash = self._get_cache_key_hash(key)
        return self.cache_dir / f"{key_hash}.json"

    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired.

        Args:
            metadata: Cache metadata containing timestamp

        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False

        timestamp = metadata.get("timestamp", 0)
        age = time.time() - timestamp
        return age > self.ttl

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found or expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug(f"Cache miss for key: {key}")
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            metadata = cache_data.get("metadata", {})
            if self._is_expired(metadata):
                logger.debug(f"Cache entry expired for key: {key}")
                self.invalidate(key)
                return None

            logger.debug(f"Cache hit for key: {key}")
            return cache_data.get("value")

        except Exception as e:
            logger.warning(f"Failed to read cache for key {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: The cache key
            value: The value to cache (must be JSON-serializable)
        """
        cache_path = self._get_cache_path(key)

        cache_data = {
            "metadata": {
                "key": key,
                "timestamp": time.time(),
            },
            "value": value,
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Cached value for key: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache value for key {key}: {e}")

    def invalidate(self, key: str) -> None:
        """Remove a cache entry.

        Args:
            key: The cache key to invalidate
        """
        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Invalidated cache for key: {key}")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for key {key}: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Get cached value or compute and cache it.

        Args:
            key: The cache key
            compute_fn: Function to compute the value if not cached

        Returns:
            The cached or computed value
        """
        # Try to get from cache
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        # Compute and cache
        logger.debug(f"Computing value for key: {key}")
        value = compute_fn()
        self.set(key, value)
        return value
