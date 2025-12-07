"""Efficiency analysis agent."""

from typing import List

from omniscient_core import BaseAIAgent, AgentResponse, FileAnalysis, RepositoryInfo

from .prompts import load_prompt


class EfficiencyAgent(BaseAIAgent):
    """Agent for efficiency and performance analysis.
    
    Analyzes codebases for:
    - Performance bottlenecks
    - Algorithm complexity issues
    - Resource utilization concerns
    - Memory management problems
    - I/O optimization opportunities
    """

    def get_prompt_template(self) -> str:
        """Load the efficiency prompt template."""
        return load_prompt("efficiency")
    
    def get_default_objective(self) -> str:
        """Get default analysis objective."""
        return "Analyze code efficiency and identify performance optimizations."
