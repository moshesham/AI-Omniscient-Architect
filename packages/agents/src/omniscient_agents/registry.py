"""Agent registry for dynamic agent discovery and management."""

from typing import Dict, List, Type, Optional, Any

from omniscient_core import BaseAIAgent


class AgentRegistry:
    """Registry for managing available agents.
    
    Provides dynamic agent discovery, registration, and instantiation.
    Supports both built-in and custom (plugin) agents.
    """
    
    _agents: Dict[str, Type[BaseAIAgent]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        agent_class: Type[BaseAIAgent],
        description: str = "",
        category: str = "general",
        priority: int = 0
    ) -> None:
        """Register an agent class.
        
        Args:
            name: Unique identifier for the agent
            agent_class: The agent class to register
            description: Human-readable description
            category: Category for grouping (e.g., "architecture", "security")
            priority: Execution priority (higher = earlier)
        """
        cls._agents[name] = agent_class
        cls._metadata[name] = {
            "description": description,
            "category": category,
            "priority": priority,
            "class_name": agent_class.__name__,
        }
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister an agent.
        
        Args:
            name: Agent identifier to remove
            
        Returns:
            True if agent was removed, False if not found
        """
        if name in cls._agents:
            del cls._agents[name]
            del cls._metadata[name]
            return True
        return False
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseAIAgent]]:
        """Get an agent class by name.
        
        Args:
            name: Agent identifier
            
        Returns:
            Agent class or None if not found
        """
        return cls._agents.get(name)
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by name.
        
        Args:
            name: Agent identifier
            
        Returns:
            Metadata dict or None if not found
        """
        return cls._metadata.get(name)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent names.
        
        Returns:
            List of agent identifiers
        """
        return list(cls._agents.keys())
    
    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List agents in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of agent identifiers in the category
        """
        return [
            name for name, meta in cls._metadata.items()
            if meta.get("category") == category
        ]
    
    @classmethod
    def create_agent(
        cls,
        name: str,
        llm: Any,
        **kwargs
    ) -> Optional[BaseAIAgent]:
        """Create an agent instance.
        
        Args:
            name: Agent identifier
            llm: Language model to use
            **kwargs: Additional arguments for agent constructor
            
        Returns:
            Agent instance or None if not found
        """
        agent_class = cls.get(name)
        if agent_class is None:
            return None
        
        meta = cls._metadata.get(name, {})
        default_kwargs = {
            "name": name,
            "description": meta.get("description", ""),
            "analysis_focus": meta.get("category", "general"),
        }
        default_kwargs.update(kwargs)
        
        return agent_class(llm=llm, **default_kwargs)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents."""
        cls._agents.clear()
        cls._metadata.clear()


# Module-level convenience functions
def register_agent(
    name: str,
    agent_class: Type[BaseAIAgent],
    description: str = "",
    category: str = "general",
    priority: int = 0
) -> None:
    """Register an agent class."""
    AgentRegistry.register(name, agent_class, description, category, priority)


def get_agent(name: str) -> Optional[Type[BaseAIAgent]]:
    """Get an agent class by name."""
    return AgentRegistry.get(name)


def list_agents() -> List[str]:
    """List all registered agent names."""
    return AgentRegistry.list_agents()


# Auto-register built-in agents
def _register_builtin_agents():
    """Register all built-in agents."""
    from .architecture import ArchitectureAgent
    from .efficiency import EfficiencyAgent
    from .reliability import ReliabilityAgent
    from .alignment import AlignmentAgent
    
    register_agent(
        "architecture",
        ArchitectureAgent,
        "Analyzes code architecture, patterns, and design quality",
        "architecture",
        priority=10
    )
    register_agent(
        "efficiency",
        EfficiencyAgent,
        "Identifies performance bottlenecks and optimization opportunities",
        "performance",
        priority=5
    )
    register_agent(
        "reliability",
        ReliabilityAgent,
        "Checks error handling, testing coverage, and resilience",
        "reliability",
        priority=8
    )
    register_agent(
        "alignment",
        AlignmentAgent,
        "Verifies alignment between code, docs, and requirements",
        "quality",
        priority=3
    )


# Register on import
_register_builtin_agents()
