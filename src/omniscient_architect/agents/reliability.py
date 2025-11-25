"""Reliability analysis agent."""

from .base import BaseAIAgent
from ..prompts.loaders import load_prompt
from ..models import FileAnalysis, RepositoryInfo
from typing import List

class ReliabilityAgent(BaseAIAgent):
    """Agent for reliability analysis."""
    def get_prompt_template(self) -> str:
        """Load the reliability prompt template from the prompts module."""
        return load_prompt("reliability")

    async def analyze(self, files: List[FileAnalysis], repo_info: RepositoryInfo):
        """
        Analyze codebase for reliability issues, error handling, and testing coverage.

        Args:
            files: List of FileAnalysis objects to review.
            repo_info: RepositoryInfo object with repo metadata.

        Returns:
            AgentResponse: Structured findings and recommendations.
        """
        context = f"Repo path: {repo_info.path}\nBranch: {repo_info.branch}"
        objective = repo_info.project_objective or "Analyze reliability, error handling, and testing coverage."
        files_info = self.prepare_files_context(files)
        response = await self.call_llm(context, objective, files_info)
        response.agent_name = self.name
        return response
