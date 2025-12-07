"""Base classes for AI-powered analysis agents."""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any
from pydantic import BaseModel, ConfigDict, Field

from .models import FileAnalysis, RepositoryInfo


class AgentResponse(BaseModel):
    """Structured response from an AI agent."""
    model_config = ConfigDict(extra="allow")
    
    agent_name: str = Field(default="", description="Name of the agent that produced the findings")
    findings: List[str] = Field(default_factory=list, description="List of key findings from the analysis")
    confidence: float = Field(default=0.0, description="Confidence score between 0 and 1", ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Detailed reasoning for the findings")
    recommendations: List[str] = Field(default_factory=list, description="Specific recommendations for improvement")


class BaseAIAgent(ABC):
    """Base class for AI-powered analysis agents.
    
    This abstract base class provides the foundation for implementing
    specialized AI analysis agents. Subclasses must implement:
    - get_prompt_template(): Returns the prompt template string
    - analyze(): Performs the actual analysis
    
    Attributes:
        llm: The language model to use for analysis
        name: The agent's display name
        description: Short description of what the agent does
        analysis_focus: The specific area this agent focuses on
        stream_callback: Optional callback for streaming responses
    """

    def __init__(
        self,
        llm: Any,  # BaseLanguageModel
        name: str,
        description: str,
        analysis_focus: str
    ):
        """Initialize the agent.
        
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
        
        # Lazy initialization of prompt template and parser
        self._prompt_template = None
        self._output_parser = None

    @property
    def prompt_template(self):
        """Lazy-load the prompt template."""
        if self._prompt_template is None:
            try:
                from langchain_core.prompts import PromptTemplate
                self._prompt_template = PromptTemplate(
                    template=self.get_prompt_template(),
                    input_variables=["context", "objective", "files_info", "format_instructions"],
                )
            except ImportError:
                raise ImportError("langchain-core is required for agent functionality")
        return self._prompt_template

    @property
    def output_parser(self):
        """Lazy-load the output parser."""
        if self._output_parser is None:
            try:
                from langchain_core.output_parsers import PydanticOutputParser
                self._output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
            except ImportError:
                raise ImportError("langchain-core is required for agent functionality")
        return self._output_parser

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Return the prompt template for this agent.
        
        Returns:
            A string containing the prompt template with placeholders
            for context, objective, files_info, and format_instructions.
        """
        pass

    def get_default_objective(self) -> str:
        """Get default analysis objective for this agent.
        
        Can be overridden by subclasses to provide agent-specific objectives.
        
        Returns:
            Default objective string
        """
        return f"Analyze codebase for {self.analysis_focus} issues."
    
    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> AgentResponse:
        """Run analysis and return structured agent response.
        
        Default implementation that can be overridden for custom behavior.
        
        Args:
            files: List of FileAnalysis objects to analyze
            repo_info: Repository information and context
            
        Returns:
            AgentResponse with findings, confidence, and recommendations
        """
        context = f"Repo path: {repo_info.path}\nBranch: {repo_info.branch}"
        objective = repo_info.project_objective or self.get_default_objective()
        files_info = self.prepare_files_context(files)
        response = await self.call_llm(context, objective, files_info)
        response.agent_name = self.name
        return response

    def prepare_files_context(self, files: List[FileAnalysis], max_files: int = 10) -> str:
        """Prepare a context string from file analyses.
        
        Args:
            files: List of FileAnalysis objects
            max_files: Maximum number of files to include
            
        Returns:
            Formatted string with file information
        """
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
        """Call the LLM with structured prompts.
        
        Args:
            context: Context information about the repository
            objective: The analysis objective
            files_info: Formatted file information string
            
        Returns:
            Parsed AgentResponse from the LLM
        """
        try:
            prompt_text = self.prompt_template.format(
                context=context,
                objective=objective,
                files_info=files_info,
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # Try streaming if callback is set
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
                                pass  # Ignore callback errors to avoid breaking streaming
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

    def set_stream_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set a callback function for streaming responses.
        
        Args:
            callback: Function that receives (agent_name, token) for each streamed token
        """
        self.stream_callback = callback
