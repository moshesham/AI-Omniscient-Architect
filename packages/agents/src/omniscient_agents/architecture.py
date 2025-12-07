"""Architecture analysis agent."""

from typing import List

from omniscient_core import BaseAIAgent, AgentResponse, FileAnalysis, RepositoryInfo

from .prompts import load_prompt


class ArchitectureAgent(BaseAIAgent):
    """Agent for architectural analysis.
    
    Analyzes codebases for:
    - Architectural patterns and anti-patterns
    - Design principles adherence (SOLID, DRY, etc.)
    - Module organization and coupling
    - Dependency management
    - Scalability concerns
    """

    def get_prompt_template(self) -> str:
        """Load the architecture prompt template."""
        return load_prompt("architecture")
    
    def get_default_objective(self) -> str:
        """Get default analysis objective."""
        return "Analyze architectural patterns and design quality."
