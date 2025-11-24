"""Example demonstrating cache usage."""

from pathlib import Path
from omniscient_architect.utils.cache import AnalysisCache


def expensive_computation():
    """Simulate an expensive computation."""
    print("Running expensive computation...")
    # In a real scenario, this might be:
    # - Complex file analysis
    # - API calls
    # - Database queries
    return {"result": "computed_value", "score": 95}


def main():
    """Demonstrate cache functionality."""
    # Initialize cache
    cache_dir = Path("/tmp/omniscient_cache")
    cache = AnalysisCache(cache_dir)

    print("=== Basic Cache Usage ===")
    
    # Set and get
    cache.set("my_key", "my_value")
    value = cache.get("my_key")
    print(f"Retrieved from cache: {value}")

    print("\n=== get_or_compute Pattern ===")
    
    # First call - computes
    result1 = cache.get_or_compute("computation_result", expensive_computation)
    print(f"First call result: {result1}")

    # Second call - uses cache
    result2 = cache.get_or_compute("computation_result", expensive_computation)
    print(f"Second call result: {result2}")

    print("\n=== Cache with TTL ===")
    
    # Cache with 3600 second (1 hour) TTL
    ttl_cache = AnalysisCache(cache_dir / "ttl", ttl=3600)
    ttl_cache.set("temp_data", {"expires": "in 1 hour"})
    print(f"TTL cache entry: {ttl_cache.get('temp_data')}")

    print("\n=== Cache Management ===")
    
    # Invalidate specific entry
    cache.invalidate("my_key")
    print(f"After invalidation: {cache.get('my_key')}")

    # Clear all cache
    cache.clear()
    print("Cache cleared")


if __name__ == "__main__":
    main()
