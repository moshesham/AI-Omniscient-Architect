"""Base classes for AI-powered analysis agents."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models import AgentFindings, FileAnalysis, RepositoryInfo

class AgentResponse(BaseModel):
    """Structured response from an AI agent."""
    findings: List[str] = Field(description="List of key findings from the analysis")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed reasoning for the findings")
    recommendations: List[str] = Field(description="Specific recommendations for improvement")

class BaseAIAgent(ABC):
    """Base class for AI-powered analysis agents."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        name: str,
        description: str,
        analysis_focus: str
    ):
        """
        Args:
            llm: The language model to use for analysis.
            name: The agent's name.
            description: Short description of the agent.
            analysis_focus: The focus area for analysis.
        """
        self.llm = llm
        self.name = name
        self.description = description
        self.analysis_focus = analysis_focus
        self.stream_callback: Optional[Callable[[str, str], None]] = None
        self.prompt_template = PromptTemplate(
            template=self._get_prompt_template(),
            input_variables=["context", "objective", "files_info", "format_instructions"],
        )
        self.output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

    @abstractmethod
    def _get_prompt_template(self) -> str:
        """Return the prompt template for this agent."""
        pass

    @abstractmethod
    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> AgentResponse:
        """Run analysis and return structured agent response."""
        pass
