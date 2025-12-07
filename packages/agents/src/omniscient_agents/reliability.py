"""Reliability analysis agent."""

from typing import List

from omniscient_core import BaseAIAgent, AgentResponse, FileAnalysis, RepositoryInfo

from .prompts import load_prompt


class ReliabilityAgent(BaseAIAgent):
    """Agent for reliability and robustness analysis.
    
    Analyzes codebases for:
    - Error handling completeness
    - Exception management patterns
    - Testing coverage gaps
    - Fault tolerance mechanisms
    - Recovery and resilience patterns
    """

    def get_prompt_template(self) -> str:
        """Load the reliability prompt template."""
        return load_prompt("reliability")
    
    def get_default_objective(self) -> str:
        """Get default analysis objective."""
        return "Analyze reliability, error handling, and testing coverage."
