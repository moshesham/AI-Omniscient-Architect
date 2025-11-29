"""Exception classes for LLM operations."""

from typing import Optional


class LLMError(Exception):
    """Base exception for LLM operations."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        self.message = message
        self.provider = provider
        super().__init__(message)


class ProviderUnavailable(LLMError):
    """Raised when a provider is not available."""
    
    def __init__(self, provider: str, reason: Optional[str] = None):
        self.reason = reason
        message = f"Provider '{provider}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message, provider)


class ModelNotFoundError(LLMError):
    """Raised when a model is not found."""
    
    def __init__(self, model: str, provider: str = "unknown"):
        self.model = model
        message = f"Model '{model}' not found on provider '{provider}'"
        super().__init__(message, provider)


# Alias for convenience
ModelNotFound = ModelNotFoundError


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self, 
        provider: str, 
        retry_after: Optional[float] = None,
        message: Optional[str] = None,
    ):
        self.retry_after = retry_after
        msg = message or f"Rate limit exceeded for provider '{provider}'"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(msg, provider)


class TokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    
    def __init__(
        self,
        provider: str,
        requested_tokens: int,
        max_tokens: int,
    ):
        self.requested_tokens = requested_tokens
        self.max_tokens = max_tokens
        message = (
            f"Token limit exceeded: requested {requested_tokens}, "
            f"max allowed {max_tokens}"
        )
        super().__init__(message, provider)


class AuthenticationError(LLMError):
    """Raised when authentication fails."""
    
    def __init__(self, provider: str, message: Optional[str] = None):
        msg = message or f"Authentication failed for provider '{provider}'"
        super().__init__(msg, provider)


class TimeoutError(LLMError):
    """Raised when a request times out."""
    
    def __init__(self, provider: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        message = f"Request to '{provider}' timed out after {timeout_seconds}s"
        super().__init__(message, provider)


class GenerationError(LLMError):
    """Raised when generation fails."""
    
    def __init__(
        self, 
        provider: str, 
        message: str,
        raw_error: Optional[Exception] = None,
    ):
        self.raw_error = raw_error
        super().__init__(message, provider)


class ConfigurationError(LLMError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str):
        super().__init__(message, None)
