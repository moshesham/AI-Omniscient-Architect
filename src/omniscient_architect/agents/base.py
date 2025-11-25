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
    agent_name: str = Field(description="Name of the agent that produced the findings")
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
            template=self.get_prompt_template(),
            input_variables=["context", "objective", "files_info", "format_instructions"],
        )
        self.output_parser = PydanticOutputParser(pydantic_object=AgentResponse)


    @abstractmethod
    def get_prompt_template(self) -> str:
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

    def prepare_files_context(self, files: List[FileAnalysis], max_files: int = 10) -> str:
        """Prepare a context string from file analyses."""
        context_parts = []
        for file in files[:max_files]:
            context_parts.append(f"File: {file.path}")
            context_parts.append(f"Language: {file.language}")
            context_parts.append(f"Size: {file.size} bytes")
            if file.content and len(file.content) < 2000:
                context_parts.append(f"Content preview:\n{file.content[:2000]}...")
            context_parts.append("---")
        return "\n".join(context_parts)

    async def call_llm(self, context: str, objective: str, files_info: str) -> AgentResponse:
        """Call the LLM with structured prompts."""
        try:
            prompt_text = self.prompt_template.format(
                context=context,
                objective=objective,
                files_info=files_info,
                format_instructions=self.output_parser.get_format_instructions()
            )
            if self.stream_callback and hasattr(self.llm, "astream"):
                chunks: List[str] = []
                try:
                    async for part in self.llm.astream(prompt_text):
                        token = part if isinstance(part, str) else getattr(part, "content", str(part))
                        if token:
                            chunks.append(token)
                            try:
                                self.stream_callback(self.name, token)
                            except Exception:
                                pass
                    response_text = "".join(chunks)
                except Exception:
                    llm_response = await self.llm.ainvoke(prompt_text)
                    response_text = getattr(llm_response, "content", str(llm_response))
            else:
                llm_response = await self.llm.ainvoke(prompt_text)
                response_text = getattr(llm_response, "content", str(llm_response))
            response = self.output_parser.parse(response_text)
            return response
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                findings=[f"Analysis failed due to: {str(e)}"],
                confidence=0.0,
                reasoning="LLM analysis encountered an error",
                recommendations=["Please check LLM configuration and try again"]
            )
