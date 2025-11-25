"""Efficiency analysis agent."""

from .base import BaseAIAgent
from ..prompts.loaders import load_prompt
from ..models import FileAnalysis, RepositoryInfo
from typing import List

class EfficiencyAgent(BaseAIAgent):
    """Agent for efficiency analysis."""
    def get_prompt_template(self) -> str:
        """Load the efficiency prompt template from the prompts module."""
        return load_prompt("efficiency")

    async def analyze(self, files: List[FileAnalysis], repo_info: RepositoryInfo):
        """
        Analyze codebase for performance bottlenecks and optimization opportunities.

        Args:
            files: List of FileAnalysis objects to review.
            repo_info: RepositoryInfo object with repo metadata.

        Returns:
            AgentResponse: Structured findings and recommendations.
        """
        context = f"Repo path: {repo_info.path}\nBranch: {repo_info.branch}"
        objective = repo_info.project_objective or "Analyze code efficiency and identify performance optimizations."
        files_info = self.prepare_files_context(files)
        response = await self.call_llm(context, objective, files_info)
        response.agent_name = self.name
        return response
