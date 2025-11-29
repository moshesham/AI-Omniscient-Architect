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

    async def analyze(
        self, 
        files: List[FileAnalysis], 
        repo_info: RepositoryInfo
    ) -> AgentResponse:
        """Analyze codebase for alignment issues.

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
        response.agent_name = self.name
        return response
