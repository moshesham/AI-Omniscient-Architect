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

    async def analyze(
        self, 
        files: List[FileAnalysis], 
        repo_info: RepositoryInfo
    ) -> AgentResponse:
        """Analyze codebase for performance and efficiency issues.

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
