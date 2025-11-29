"""LLM-integrated base agent using omniscient-llm package."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional, Any, Dict
from dataclasses import dataclass, field

from omniscient_core import FileAnalysis, RepositoryInfo
from omniscient_core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Issue:
    """Represents a code issue found during analysis."""
    severity: str = "medium"
    category: str = "general"
    description: str = ""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class LLMAgentResponse:
    """Response from an LLM agent analysis."""
    agent_name: str = ""
    summary: str = ""
    issues: List[Issue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_response: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMAgentResponse":
        """Create from dictionary."""
        issues = []
        for issue_data in data.get("issues", []):
            if isinstance(issue_data, dict):
                issues.append(Issue(**issue_data))
        
        return cls(
            agent_name=data.get("agent_name", ""),
            summary=data.get("summary", ""),
            issues=issues,
            recommendations=data.get("recommendations", []),
        )


# Lazy import to avoid circular dependencies
def _get_llm_models():
    """Get LLM model classes."""
    from omniscient_llm.models import (
        GenerationRequest,
        ChatMessage,
        Role,
        StreamChunk,
    )
    return GenerationRequest, ChatMessage, Role, StreamChunk


class LLMAgent(ABC):
    """Base agent that uses omniscient-llm for LLM operations.
    
    This provides a cleaner integration with local LLMs via Ollama
    and supports streaming responses.
    
    Subclasses should implement:
    - get_prompt_template(): Return the analysis prompt
    - parse_response(): Parse LLM response into LLMAgentResponse
    
    Example:
        >>> class MyAgent(LLMAgent):
        ...     def get_prompt_template(self) -> str:
        ...         return "Analyze this code: {files_info}"
    """
    
    def __init__(
        self,
        name: str = "llm_agent",
        description: str = "LLM-based analysis agent",
        llm_client = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """Initialize LLM agent.
        
        Args:
            name: Agent name
            description: Agent description  
            llm_client: LLMClient instance (optional, created if not provided)
            model: Model name override
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.name = name
        self.description = description
        self._llm_client = llm_client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._initialized = False
    
    def prepare_files_context(self, files: List[FileAnalysis]) -> str:
        """Prepare files for prompt context.
        
        Args:
            files: List of files to include
            
        Returns:
            Formatted string with file contents
        """
        parts = []
        for f in files:
            parts.append(f"--- File: {f.path} ({f.language}) ---")
            parts.append(f.content)
            parts.append("")
        return "\n".join(parts)
    
    async def _ensure_client(self):
        """Ensure LLM client is initialized."""
        if self._llm_client is None:
            # Try to create default client with Ollama
            try:
                from omniscient_llm import OllamaProvider, LLMClient
                
                model = self._model or "llama3.2:latest"
                provider = OllamaProvider(model=model)
                self._llm_client = LLMClient(provider=provider)
                await self._llm_client.initialize()
                self._initialized = True
                
                logger.info(f"Initialized LLM client with model: {model}")
                
            except ImportError:
                raise RuntimeError(
                    "omniscient-llm package not installed. "
                    "Run: pip install omniscient-llm[ollama]"
                )
    
    async def close(self):
        """Close LLM client."""
        if self._llm_client and self._initialized:
            await self._llm_client.close()
            self._initialized = False
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template for this agent.
        
        Returns:
            Prompt template string with placeholders
        """
        pass
    
    async def parse_response(self, content: str) -> LLMAgentResponse:
        """Parse LLM response into LLMAgentResponse.
        
        Default implementation tries to parse JSON, override for custom parsing.
        
        Args:
            content: Raw LLM response text
            
        Returns:
            Parsed LLMAgentResponse
        """
        import json
        import re
        
        # Try to extract JSON from the response
        try:
            # First try direct parse
            data = json.loads(content)
            response = LLMAgentResponse.from_dict(data)
            response.raw_response = content
            return response
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                response = LLMAgentResponse.from_dict(data)
                response.raw_response = content
                return response
            except json.JSONDecodeError:
                pass
        
        # Return raw content as summary if no JSON found
        return LLMAgentResponse(
            agent_name=self.name,
            summary=content,
            issues=[],
            raw_response=content,
        )
    
    def build_prompt(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo,
    ) -> str:
        """Build the complete prompt for analysis.
        
        Args:
            files: Files to analyze
            repo_info: Repository information
            
        Returns:
            Complete prompt string
        """
        template = self.get_prompt_template()
        
        # Build files context
        files_info = self.prepare_files_context(files)
        
        # Format the template
        return template.format(
            files_info=files_info,
            repo_name=repo_info.name,
            repo_path=repo_info.path,
            branch=repo_info.branch,
            objective=repo_info.project_objective or "General code analysis",
        )
    
    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo,
    ) -> LLMAgentResponse:
        """Analyze files using LLM.
        
        Args:
            files: Files to analyze
            repo_info: Repository information
            
        Returns:
            Analysis response
        """
        await self._ensure_client()
        
        prompt = self.build_prompt(files, repo_info)
        
        # Use the client's generate method with string prompt
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        
        result = await self.parse_response(response.content)
        result.agent_name = self.name
        
        return result
    
    async def analyze_streaming(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo,
    ) -> AsyncIterator[str]:
        """Analyze files with streaming response.
        
        Args:
            files: Files to analyze
            repo_info: Repository information
            
        Yields:
            Tokens as they are generated
        """
        await self._ensure_client()
        
        prompt = self.build_prompt(files, repo_info)
        
        async for chunk in self._llm_client.stream(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        ):
            if chunk.content:
                yield chunk.content


class CodeReviewAgent(LLMAgent):
    """Generic code review agent using LLM."""
    
    def __init__(
        self,
        focus_areas: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize code review agent.
        
        Args:
            focus_areas: Specific areas to focus on
            **kwargs: Passed to LLMAgent
        """
        super().__init__(
            name="code_review",
            description="General code review agent",
            **kwargs,
        )
        self.focus_areas = focus_areas or [
            "code quality",
            "potential bugs",
            "performance",
            "maintainability",
        ]
    
    def get_prompt_template(self) -> str:
        """Get code review prompt."""
        areas = ", ".join(self.focus_areas)
        
        return f"""You are an expert code reviewer. Analyze the following code focusing on: {areas}

Repository: {{repo_name}}
Branch: {{branch}}
Objective: {{objective}}

Files to review:
{{files_info}}

Provide your analysis in the following JSON format:
{{{{
    "summary": "Brief overall summary",
    "issues": [
        {{{{
            "severity": "critical|high|medium|low",
            "category": "category name",
            "description": "Issue description",
            "file_path": "path/to/file",
            "line_number": 123,
            "suggestion": "How to fix"
        }}}}
    ],
    "recommendations": ["List of general recommendations"]
}}}}

Be thorough but concise. Focus on actionable feedback."""
