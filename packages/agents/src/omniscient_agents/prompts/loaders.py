"""Prompt loading utilities."""

from typing import Dict, List, Optional

from .templates import (
    ARCHITECTURE_PROMPT,
    EFFICIENCY_PROMPT,
    RELIABILITY_PROMPT,
    ALIGNMENT_PROMPT,
)

# Registry of prompts
_PROMPTS: Dict[str, str] = {
    "architecture": ARCHITECTURE_PROMPT,
    "efficiency": EFFICIENCY_PROMPT,
    "reliability": RELIABILITY_PROMPT,
    "alignment": ALIGNMENT_PROMPT,
}

# Custom prompts added at runtime
_CUSTOM_PROMPTS: Dict[str, str] = {}


def load_prompt(key: str) -> str:
    """Load a prompt template by key.
    
    Args:
        key: Prompt identifier (e.g., "architecture", "efficiency")
        
    Returns:
        Prompt template string, or empty string if not found
    """
    # Check custom prompts first (allows overrides)
    if key in _CUSTOM_PROMPTS:
        return _CUSTOM_PROMPTS[key]
    return _PROMPTS.get(key, "")


def get_prompt_names() -> List[str]:
    """Get list of all available prompt names.
    
    Returns:
        List of prompt identifiers
    """
    return list(set(_PROMPTS.keys()) | set(_CUSTOM_PROMPTS.keys()))


def register_prompt(key: str, template: str, override: bool = False) -> bool:
    """Register a custom prompt template.
    
    Args:
        key: Unique identifier for the prompt
        template: The prompt template string
        override: If True, allow overriding built-in prompts
        
    Returns:
        True if registered successfully, False if key exists and override=False
    """
    if key in _PROMPTS and not override:
        return False
    _CUSTOM_PROMPTS[key] = template
    return True


def unregister_prompt(key: str) -> bool:
    """Remove a custom prompt.
    
    Args:
        key: Prompt identifier to remove
        
    Returns:
        True if removed, False if not found
    """
    if key in _CUSTOM_PROMPTS:
        del _CUSTOM_PROMPTS[key]
        return True
    return False


def get_prompt_info(key: str) -> Optional[Dict[str, str]]:
    """Get information about a prompt.
    
    Args:
        key: Prompt identifier
        
    Returns:
        Dict with prompt info or None if not found
    """
    template = load_prompt(key)
    if not template:
        return None
    
    return {
        "key": key,
        "is_custom": key in _CUSTOM_PROMPTS,
        "length": len(template),
        "preview": template[:200] + "..." if len(template) > 200 else template,
    }
