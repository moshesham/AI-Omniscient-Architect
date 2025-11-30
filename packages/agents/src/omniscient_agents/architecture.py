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

    async def analyze(
        self, 
        files: List[FileAnalysis], 
        repo_info: RepositoryInfo
    ) -> AgentResponse:
        """Analyze codebase for architectural patterns and design quality.

        Args:
            files: List of FileAnalysis objects to review.
            repo_info: RepositoryInfo object with repo metadata.

        Returns:
            AgentResponse: Structured findings and recommendations.
        """
        context = f"Repo path: {repo_info.path}\nBranch: {repo_info.branch}"
        objective = repo_info.project_objective or "Analyze architectural patterns and design quality."
        files_info = self.prepare_files_context(files)
        response = await self.call_llm(context, objective, files_info)
        response.agent_name = self.name
        return response
