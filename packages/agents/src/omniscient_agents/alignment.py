"""Alignment analysis agent."""

from typing import List

from omniscient_core import BaseAIAgent, AgentResponse, FileAnalysis, RepositoryInfo

from .prompts import load_prompt


class AlignmentAgent(BaseAIAgent):
    """Agent for code-documentation-requirements alignment analysis.
    
    Analyzes codebases for:
    - Code vs. documentation consistency
    - Implementation vs. requirements alignment
    - API contract adherence
    - Naming convention consistency
    - Standards compliance
    """

    def get_prompt_template(self) -> str:
        """Load the alignment prompt template."""
        return load_prompt("alignment")
    
    def get_default_objective(self) -> str:
        """Get default analysis objective."""
        return "Analyze alignment between code, documentation, and requirements."
