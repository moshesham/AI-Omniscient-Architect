"""Analysis cache with file-hash based keys."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Dict

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


class AnalysisCache:
    """Cache for analysis results with file-based persistence.
    
    Uses file content hashes as cache keys for accurate invalidation.
    Supports TTL-based expiration.
    
    Attributes:
        cache_dir: Directory for storing cache files
        ttl: Time-to-live in seconds (None = no expiration)
        use_blake3: Whether to use blake3 for hashing (faster)
    """
    
    def __init__(
        self, 
        cache_dir: str = ".omniscient_cache",
        ttl: Optional[int] = None
    ):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live for entries in seconds (None = no expiration)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.use_blake3 = HAS_BLAKE3
        self._initialize_cache_dir()
    
    def _initialize_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash_file(self, file_path: str) -> str:
        """Generate hash of file contents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        if self.use_blake3:
            hasher = blake3.blake3()
        else:
            hasher = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _hash_key(self, key: str) -> str:
        """Hash a string key for use as filename.
        
        Args:
            key: The cache key
            
        Returns:
            Hashed key suitable for filename
        """
        if self.use_blake3:
            return blake3.blake3(key.encode()).hexdigest()[:32]
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key.
        
        Args:
            key: Cache key (file path or arbitrary string)
            
        Returns:
            Path to the cache file
        """
        # If key is a file path, use content hash
        if os.path.isfile(key):
            key_hash = self._hash_file(key)
        else:
            key_hash = self._hash_key(key)
        
        return self.cache_dir / f"{key_hash}.json"
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry has expired.
        
        Args:
            metadata: Cache entry metadata
            
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
            key: Cache key (can be file path)
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get("metadata", {})
            if self._is_expired(metadata):
                self.invalidate(key)
                return None
            
            return cache_data.get("value")
            
        except Exception:
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.
        
        Args:
            key: Cache key (can be file path)
            value: Value to cache (must be JSON-serializable)
        """
        cache_path = self._get_cache_path(key)
        
        cache_data = {
            "metadata": {
                "timestamp": time.time(),
                "key": key,
            },
            "value": value,
        }
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
        except Exception:
            pass  # Silently fail on cache write errors
    
    def invalidate(self, key: str) -> bool:
        """Remove a cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was removed, False if not found
        """
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                cache_path.unlink()
                return True
            except Exception:
                pass
        
        return False
    
    def clear(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of entries removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "entries": len(cache_files),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "ttl": self.ttl,
            "uses_blake3": self.use_blake3,
        }
