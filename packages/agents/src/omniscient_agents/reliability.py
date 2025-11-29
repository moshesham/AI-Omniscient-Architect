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

    async def analyze(
        self, 
        files: List[FileAnalysis], 
        repo_info: RepositoryInfo
    ) -> AgentResponse:
        """Analyze codebase for reliability and error handling.

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
