"""Prompt templates and loaders for agents."""

from .templates import (
    ARCHITECTURE_PROMPT,
    EFFICIENCY_PROMPT,
    RELIABILITY_PROMPT,
    ALIGNMENT_PROMPT,
)
from .loaders import load_prompt, get_prompt_names, register_prompt

__all__ = [
    "ARCHITECTURE_PROMPT",
    "EFFICIENCY_PROMPT",
    "RELIABILITY_PROMPT",
    "ALIGNMENT_PROMPT",
    "load_prompt",
    "get_prompt_names",
    "register_prompt",
]
