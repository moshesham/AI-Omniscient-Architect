"""Authentication dependencies."""

from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from .config import get_config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    api_key: str = Security(api_key_header),
) -> Optional[str]:
    """Verify API key if configured."""
    config = get_config()
    
    # If no API key is configured, allow all requests
    if not config.api_key:
        return api_key

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )

    if api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
        
    return api_key
