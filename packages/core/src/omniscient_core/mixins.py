"""Mixins for common patterns across the codebase."""

from typing import TypeVar


Self = TypeVar("Self")


class AsyncContextMixin:
    """Mixin providing standard async context manager implementation.
    
    Classes using this mixin should implement initialize() and close()
    methods for resource management. The mixin provides __aenter__ and
    __aexit__ that call these methods automatically.
    
    Example:
        >>> class MyClient(AsyncContextMixin):
        ...     async def initialize(self):
        ...         self._client = await connect()
        ...     
        ...     async def close(self):
        ...         await self._client.disconnect()
        >>> 
        >>> async with MyClient() as client:
        ...     await client.do_work()
    """
    
    async def initialize(self) -> None:
        """Initialize resources.
        
        Override this method to perform setup operations like
        establishing connections, loading configuration, etc.
        """
        pass
    
    async def close(self) -> None:
        """Release resources.
        
        Override this method to perform cleanup operations like
        closing connections, flushing buffers, etc.
        """
        pass
    
    async def __aenter__(self: Self) -> Self:
        """Async context manager entry.
        
        Calls initialize() and returns self.
        """
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.
        
        Calls close() to cleanup resources.
        """
        await self.close()
