"""Rate limit handling for GitHub API."""

import asyncio
import time
from datetime import datetime
from typing import Optional, Callable, Any

from omniscient_core.logging import get_logger
from .models import RateLimitInfo

logger = get_logger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded and cannot wait."""
    
    def __init__(self, message: str, rate_limit_info: Optional[RateLimitInfo] = None):
        super().__init__(message)
        self.rate_limit_info = rate_limit_info


class RateLimitHandler:
    """Handles GitHub API rate limiting with automatic backoff.
    
    Provides automatic retry with exponential backoff when rate limits
    are encountered, and can wait for rate limit reset if desired.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_wait_seconds: int = 300,
        auto_wait_for_reset: bool = True,
    ):
        """Initialize rate limit handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            max_wait_seconds: Maximum seconds to wait for reset
            auto_wait_for_reset: Whether to wait for rate limit reset
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_wait_seconds = max_wait_seconds
        self.auto_wait_for_reset = auto_wait_for_reset
        self._current_rate_limit: Optional[RateLimitInfo] = None
        self._last_request_time: float = 0
        self._min_request_interval: float = 0.1  # 100ms between requests
    
    @property
    def current_rate_limit(self) -> Optional[RateLimitInfo]:
        """Get current rate limit info."""
        return self._current_rate_limit
    
    def update_from_headers(self, headers: dict) -> Optional[RateLimitInfo]:
        """Update rate limit info from response headers.
        
        Args:
            headers: Response headers from GitHub API
            
        Returns:
            Updated RateLimitInfo or None
        """
        try:
            limit = int(headers.get("x-ratelimit-limit", 0))
            remaining = int(headers.get("x-ratelimit-remaining", 0))
            reset = int(headers.get("x-ratelimit-reset", 0))
            resource = headers.get("x-ratelimit-resource", "core")
            
            if limit > 0:
                self._current_rate_limit = RateLimitInfo(
                    limit=limit,
                    remaining=remaining,
                    reset_timestamp=reset,
                    resource=resource,
                )
                
                if remaining < 10:
                    logger.warning(
                        f"Rate limit low: {remaining}/{limit} remaining, "
                        f"resets in {self._current_rate_limit.seconds_until_reset}s"
                    )
                
                return self._current_rate_limit
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse rate limit headers: {e}")
        
        return None
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit is close to being exceeded."""
        if self._current_rate_limit and self._current_rate_limit.is_exceeded:
            if self.auto_wait_for_reset:
                wait_time = min(
                    self._current_rate_limit.seconds_until_reset + 1,
                    self.max_wait_seconds
                )
                
                if wait_time > 0:
                    logger.info(f"Rate limit exceeded, waiting {wait_time}s for reset")
                    await asyncio.sleep(wait_time)
            else:
                raise RateLimitError(
                    "Rate limit exceeded",
                    self._current_rate_limit
                )
        
        # Enforce minimum interval between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        
        self._last_request_time = time.time()
    
    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
        operation_name: str = "API call",
    ) -> Any:
        """Execute operation with retry on rate limit.
        
        Args:
            operation: Async callable to execute
            operation_name: Name for logging
            
        Returns:
            Result of the operation
            
        Raises:
            RateLimitError: If retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                await self.wait_if_needed()
                return await operation()
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if rate limit error
                is_rate_limit = (
                    "rate limit" in error_str or
                    "403" in error_str or
                    "429" in error_str
                )
                
                if is_rate_limit and attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(
                        f"{operation_name} hit rate limit, "
                        f"retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Not a rate limit error or out of retries
                    raise
        
        raise RateLimitError(
            f"Rate limit exceeded after {self.max_retries} retries",
            self._current_rate_limit
        )
    
    def get_status(self) -> dict:
        """Get current rate limit status.
        
        Returns:
            Dict with rate limit information
        """
        if self._current_rate_limit:
            return {
                "limit": self._current_rate_limit.limit,
                "remaining": self._current_rate_limit.remaining,
                "reset_time": self._current_rate_limit.reset_datetime.isoformat(),
                "seconds_until_reset": self._current_rate_limit.seconds_until_reset,
                "resource": self._current_rate_limit.resource,
                "is_exceeded": self._current_rate_limit.is_exceeded,
            }
        return {"status": "unknown"}
