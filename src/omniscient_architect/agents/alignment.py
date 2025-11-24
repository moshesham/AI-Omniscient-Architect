"""Alignment analysis agent."""

from .base import BaseAIAgent
from ..prompts.loaders import load_prompt
from ..models import FileAnalysis, RepositoryInfo
from typing import List

class AlignmentAgent(BaseAIAgent):
    """Agent for alignment analysis."""
    def get_prompt_template(self) -> str:
        """Load the alignment prompt template from the prompts module."""
        return load_prompt("alignment")

    async def analyze(self, files: List[FileAnalysis], repo_info: RepositoryInfo):
        """
        Analyze codebase for alignment issues (e.g., code vs. requirements, documentation, standards).

        Args:
            files: List of FileAnalysis objects to review.
            repo_info: RepositoryInfo object with repo metadata.

        Returns:
            AgentResponse: Structured findings and recommendations.
        """
        context = f"Repo path: {repo_info.path}\nBranch: {repo_info.branch}"
        objective = repo_info.project_objective or "Analyze alignment between code, documentation, and requirements."
        files_info = self.prepare_files_context(files)
        response = await self.call_llm(context, objective, files_info)
        return response
