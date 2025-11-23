"""Simple registry for analysis agents.

Agents register themselves here so the AnalysisEngine can dynamically instantiate
the configured set of agents without hardcoding class names.
"""
from typing import Callable, Dict, List, Optional

_REGISTRY: Dict[str, Callable] = {}


def register_agent(key: str):
    """Decorator to register an agent class under a given key.

    Usage:
        @register_agent('architecture')
        class ArchitectureAgent(...):
            ...
    """
    def _decorator(cls):
        _REGISTRY[key] = cls
        return cls

    return _decorator


def get_registered_agents() -> Dict[str, Callable]:
    """Return the mapping of registered agents."""
    return dict(_REGISTRY)


def create_agents(llm, enabled: Optional[List[str]] = None):
    """Instantiate agents registered in the registry.

    If `enabled` is provided (list of keys), only those agents will be created.
    If `enabled` is empty or None, all registered agents are created.
    """
    agents = []
    if enabled:
        enabled_set = set(enabled)
    else:
        enabled_set = None

    for key, cls in _REGISTRY.items():
        if enabled_set is not None and key not in enabled_set:
            continue
        agents.append(cls(llm))

    return agents
