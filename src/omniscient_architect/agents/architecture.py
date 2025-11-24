"""Architecture analysis agent (stub)."""

from .base import BaseAIAgent

class ArchitectureAgent(BaseAIAgent):
    """Agent for architectural analysis (stub)."""
    def _get_prompt_template(self) -> str:
        # TODO: Load from prompts module
        return ""

    async def analyze(self, files, repo_info):
        # TODO: Implement analysis logic
        pass
