"""Tests for the analysis cache."""

import json
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from omniscient_architect.utils.cache import AnalysisCache


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    return tmp_path / "test_cache"


@pytest.fixture
def cache(temp_cache_dir):
    """Create a cache instance for testing."""
    return AnalysisCache(temp_cache_dir)


class TestCacheInitialization:
    """Test cache initialization."""

    def test_cache_directory_created(self, temp_cache_dir):
        """Test that cache directory is created during initialization."""
        assert not temp_cache_dir.exists()
        cache = AnalysisCache(temp_cache_dir)
        assert temp_cache_dir.exists()
        assert temp_cache_dir.is_dir()

    def test_cache_directory_already_exists(self, temp_cache_dir):
        """Test that cache handles existing directory gracefully."""
        temp_cache_dir.mkdir(parents=True)
        cache = AnalysisCache(temp_cache_dir)
        assert temp_cache_dir.exists()

    def test_cache_with_ttl(self, temp_cache_dir):
        """Test cache initialization with TTL."""
        cache = AnalysisCache(temp_cache_dir, ttl=3600)
        assert cache.ttl == 3600


class TestCacheOperations:
    """Test basic cache operations."""

    def test_set_and_get(self, cache):
        """Test setting and getting a value."""
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"

    def test_get_nonexistent_key(self, cache):
        """Test getting a non-existent key returns None."""
        value = cache.get("nonexistent")
        assert value is None

    def test_cache_complex_data(self, cache):
        """Test caching complex data structures."""
        data = {
            "files": ["file1.py", "file2.py"],
            "analysis": {
                "complexity": 10,
                "issues": ["issue1", "issue2"],
            },
            "score": 85.5,
        }
        cache.set("complex_data", data)
        cached_data = cache.get("complex_data")
        assert cached_data == data

    def test_invalidate(self, cache):
        """Test cache invalidation."""
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

        cache.invalidate("test_key")
        assert cache.get("test_key") is None

    def test_invalidate_nonexistent_key(self, cache):
        """Test invalidating a non-existent key doesn't raise error."""
        cache.invalidate("nonexistent")  # Should not raise

    def test_clear(self, cache):
        """Test clearing all cache entries."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None


class TestCacheTTL:
    """Test cache time-to-live functionality."""

    def test_cache_expiration(self, temp_cache_dir):
        """Test that cache entries expire after TTL."""
        cache = AnalysisCache(temp_cache_dir, ttl=1)  # 1 second TTL

        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

        # Wait for expiration
        time.sleep(1.5)

        value = cache.get("test_key")
        assert value is None

    def test_cache_no_expiration_without_ttl(self, cache):
        """Test that cache entries don't expire without TTL."""
        cache.set("test_key", "test_value")
        time.sleep(0.5)
        value = cache.get("test_key")
        assert value == "test_value"


class TestGetOrCompute:
    """Test the get_or_compute method."""

    def test_compute_on_cache_miss(self, cache):
        """Test that compute function is called on cache miss."""
        compute_fn = Mock(return_value="computed_value")

        result = cache.get_or_compute("new_key", compute_fn)

        assert result == "computed_value"
        compute_fn.assert_called_once()

    def test_use_cached_on_cache_hit(self, cache):
        """Test that compute function is not called on cache hit."""
        # Pre-populate cache
        cache.set("existing_key", "cached_value")

        compute_fn = Mock(return_value="computed_value")
        result = cache.get_or_compute("existing_key", compute_fn)

        assert result == "cached_value"
        compute_fn.assert_not_called()

    def test_cache_computed_value(self, cache):
        """Test that computed value is cached for future use."""
        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return f"computed_{call_count}"

        # First call should compute
        result1 = cache.get_or_compute("key", compute_fn)
        assert result1 == "computed_1"
        assert call_count == 1

        # Second call should use cache
        result2 = cache.get_or_compute("key", compute_fn)
        assert result2 == "computed_1"
        assert call_count == 1  # Not called again

    def test_cache_empty_values(self, cache):
        """Test that empty values (but not None) are cached correctly."""
        # Test empty list
        cache.set("empty_list", [])
        assert cache.get("empty_list") == []

        # Test empty dict
        cache.set("empty_dict", {})
        assert cache.get("empty_dict") == {}

        # Test empty string
        cache.set("empty_string", "")
        assert cache.get("empty_string") == ""

        # Test False boolean
        cache.set("false_value", False)
        assert cache.get("false_value") is False

        # Test zero
        cache.set("zero", 0)
        assert cache.get("zero") == 0


class TestCacheKeyHashing:
    """Test cache key hashing functionality."""

    def test_key_with_special_characters(self, cache):
        """Test that keys with special characters are handled."""
        special_key = "key/with:special*characters?"
        cache.set(special_key, "value")
        value = cache.get(special_key)
        assert value == "value"

    def test_long_key(self, cache):
        """Test that very long keys are handled."""
        long_key = "x" * 500
        cache.set(long_key, "value")
        value = cache.get(long_key)
        assert value == "value"

    def test_different_keys_different_files(self, cache, temp_cache_dir):
        """Test that different keys result in different cache files."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache_files = list(temp_cache_dir.glob("*.json"))
        assert len(cache_files) == 2


class TestCacheMetadata:
    """Test cache metadata."""

    def test_metadata_contains_timestamp(self, cache, temp_cache_dir):
        """Test that cache entries contain timestamp metadata."""
        cache.set("test_key", "test_value")

        cache_path = cache._get_cache_path("test_key")
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        assert "metadata" in cache_data
        assert "timestamp" in cache_data["metadata"]
        assert "key" in cache_data["metadata"]
        assert cache_data["metadata"]["key"] == "test_key"

    def test_metadata_contains_original_key(self, cache, temp_cache_dir):
        """Test that original key is stored in metadata."""
        original_key = "my/special/key"
        cache.set(original_key, "value")

        cache_path = cache._get_cache_path(original_key)
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        assert cache_data["metadata"]["key"] == original_key


class TestCacheErrorHandling:
    """Test error handling in cache operations."""

    def test_corrupted_cache_file(self, cache, temp_cache_dir):
        """Test that corrupted cache files are handled gracefully."""
        # Create a corrupted cache file
        cache_path = cache._get_cache_path("corrupted_key")
        cache_path.write_text("not valid json")

        # Should return None instead of raising
        value = cache.get("corrupted_key")
        assert value is None

    def test_cache_non_serializable_object(self, cache):
        """Test caching non-JSON-serializable objects fails gracefully."""

        class NonSerializable:
            pass

        # Should not raise, but log a warning
        cache.set("bad_key", NonSerializable())

        # Value should not be retrievable
        value = cache.get("bad_key")
        assert value is None
