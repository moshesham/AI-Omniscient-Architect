"""Prompt loading utilities."""

from typing import Dict

from .templates import (
    ARCHITECTURE_PROMPT,
    EFFICIENCY_PROMPT,
    RELIABILITY_PROMPT,
    ALIGNMENT_PROMPT,
)

_PROMPTS: Dict[str, str] = {
    "architecture": ARCHITECTURE_PROMPT,
    "efficiency": EFFICIENCY_PROMPT,
    "reliability": RELIABILITY_PROMPT,
    "alignment": ALIGNMENT_PROMPT,
}

def load_prompt(key: str) -> str:
    """Load a prompt template by key."""
    return _PROMPTS.get(key, "")
