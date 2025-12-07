"""Utilities for handling optional imports.

Provides a clean pattern for handling optional dependencies with clear error messages.
"""
from typing import Optional, Tuple


def optional_import(
    module_name: str,
    package_name: Optional[str] = None,
    extra: Optional[str] = None,
) -> Tuple[bool, Optional[Exception]]:
    """Attempt to import an optional module.
    
    Args:
        module_name: Name of the module to import (e.g., "anthropic")
        package_name: Package name for error message (e.g., "omniscient-llm")
        extra: Optional extra name for installation (e.g., "anthropic")
        
    Returns:
        Tuple of (success: bool, error: Exception or None)
        
    Example:
        >>> HAS_ANTHROPIC, _ = optional_import("anthropic", "omniscient-llm", "anthropic")
        >>> if not HAS_ANTHROPIC:
        ...     raise ImportError("Install with: pip install omniscient-llm[anthropic]")
    """
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, e


def require_optional(
    module_name: str,
    package_name: str,
    extra: Optional[str] = None,
) -> None:
    """Require an optional module, raising ImportError with helpful message if missing.
    
    Args:
        module_name: Name of the module to require
        package_name: Package name for installation instructions
        extra: Optional extra name for installation
        
    Raises:
        ImportError: If the module cannot be imported
        
    Example:
        >>> # In __init__ of a class
        >>> require_optional("anthropic", "omniscient-llm", "anthropic")
    """
    success, error = optional_import(module_name, package_name, extra)
    if not success:
        if extra:
            msg = f"{module_name} package is required. Install with: pip install {package_name}[{extra}]"
        else:
            msg = f"{module_name} package is required. Install with: pip install {package_name}"
        raise ImportError(msg) from error
