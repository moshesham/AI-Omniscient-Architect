"""Base classes for AI-powered analysis agents."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .models import AgentFindings, FileAnalysis, RepositoryInfo
from .agent_registry import register_agent


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
        self.llm = llm
        self.name = name
        self.description = description
        self.analysis_focus = analysis_focus

        # Create the analysis prompt template
        self.prompt_template = PromptTemplate(
            template=self._get_prompt_template(),
            input_variables=["context", "objective", "files_info"],
        )

        # Set up output parser
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
    ) -> AgentFindings:
        """Perform the analysis and return findings."""
        pass

    async def _call_llm(self, context: str, objective: str, files_info: str) -> AgentResponse:
        """Call the LLM with structured prompts."""
        try:
            chain = self.prompt_template | self.llm | self.output_parser
            response = await chain.ainvoke({
                "context": context,
                "objective": objective,
                "files_info": files_info
            })
            return response
        except Exception as e:
            # Fallback to basic response if LLM fails
            return AgentResponse(
                findings=[f"Analysis failed due to: {str(e)}"],
                confidence=0.0,
                reasoning="LLM analysis encountered an error",
                recommendations=["Please check LLM configuration and try again"]
            )

    def _prepare_files_context(self, files: List[FileAnalysis], max_files: int = 10) -> str:
        """Prepare a context string from file analyses."""
        context_parts = []
        for file in files[:max_files]:
            context_parts.append(f"File: {file.path}")
            context_parts.append(f"Language: {file.language}")
            context_parts.append(f"Size: {file.size} bytes")
            if file.content and len(file.content) < 2000:  # Limit content size
                context_parts.append(f"Content preview:\n{file.content[:2000]}...")
            context_parts.append("---")

        return "\n".join(context_parts)


@register_agent('architecture')
class ArchitectureAgent(BaseAIAgent):
    """Agent specialized in architecture and design analysis."""

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(
            llm=llm,
            name="Agent Alpha (Architecture)",
            description="Analyzes file structure, design patterns, and scalability",
            analysis_focus="architecture"
        )

    def _get_prompt_template(self) -> str:
        return """You are Agent Alpha, an expert software architect analyzing a codebase.

CONTEXT:
{context}

PROJECT OBJECTIVE:
{objective}

FILES INFORMATION:
{files_info}

As an expert architect, analyze the codebase structure and provide insights on:
1. Overall architecture patterns and design decisions
2. Code organization and modularity
3. Scalability considerations
4. Design patterns usage
5. Potential architectural improvements

{format_instructions}

Focus on architectural strengths and weaknesses. Be specific about file locations and provide actionable recommendations."""

    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> AgentFindings:
        """Analyze architecture and design patterns."""
        context = self._prepare_files_context(files)
        files_info = f"Total files: {len(files)}, Languages: {set(f.language for f in files)}"

        response = await self._call_llm(context, repo_info.project_objective, files_info)

        return AgentFindings(
            agent_name=self.name,
            findings=response.findings,
            confidence=response.confidence,
            reasoning=response.reasoning
        )


@register_agent('efficiency')
class EfficiencyAgent(BaseAIAgent):
    """Agent specialized in efficiency and performance analysis."""

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(
            llm=llm,
            name="Agent Beta (Efficiency)",
            description="Identifies complexity, redundant code, and performance issues",
            analysis_focus="efficiency"
        )

    def _get_prompt_template(self) -> str:
        return """You are Agent Beta, an expert in code efficiency and performance analysis.

CONTEXT:
{context}

PROJECT OBJECTIVE:
{objective}

FILES INFORMATION:
{files_info}

As an efficiency expert, analyze the code for:
1. Code complexity and maintainability issues
2. Performance bottlenecks and optimization opportunities
3. Code duplication and redundancy
4. Algorithm efficiency
5. Resource usage patterns

{format_instructions}

Identify specific files with issues and provide concrete improvement suggestions."""

    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> AgentFindings:
        """Analyze code efficiency and performance."""
        context = self._prepare_files_context(files)
        files_info = f"Total files: {len(files)}, Complexity analysis available"

        response = await self._call_llm(context, repo_info.project_objective, files_info)

        return AgentFindings(
            agent_name=self.name,
            findings=response.findings,
            confidence=response.confidence,
            reasoning=response.reasoning
        )


@register_agent('reliability')
class ReliabilityAgent(BaseAIAgent):
    """Agent specialized in reliability and security analysis."""

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(
            llm=llm,
            name="Agent Gamma (Reliability & Security)",
            description="Examines error handling, security, and reliability",
            analysis_focus="reliability"
        )

    def _get_prompt_template(self) -> str:
        return """You are Agent Gamma, an expert in software reliability and security.

CONTEXT:
{context}

PROJECT OBJECTIVE:
{objective}

FILES INFORMATION:
{files_info}

As a reliability and security expert, analyze for:
1. Error handling and exception management
2. Security vulnerabilities and best practices
3. Input validation and sanitization
4. Resource management (memory, connections)
5. Edge cases and failure scenarios

{format_instructions}

Focus on potential security issues and reliability concerns with specific recommendations."""

    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> AgentFindings:
        """Analyze reliability and security."""
        context = self._prepare_files_context(files)
        files_info = f"Total files: {len(files)}, Security and reliability focus"

        response = await self._call_llm(context, repo_info.project_objective, files_info)

        return AgentFindings(
            agent_name=self.name,
            findings=response.findings,
            confidence=response.confidence,
            reasoning=response.reasoning
        )


@register_agent('alignment')
class AlignmentAgent(BaseAIAgent):
    """Agent specialized in objective alignment analysis."""

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(
            llm=llm,
            name="Agent Delta (Alignment)",
            description="Validates alignment with business objectives",
            analysis_focus="alignment"
        )

    def _get_prompt_template(self) -> str:
        return """You are Agent Delta, an expert in business-objective alignment analysis.

CONTEXT:
{context}

PROJECT OBJECTIVE:
{objective}

FILES INFORMATION:
{files_info}

As an alignment expert, evaluate how well the codebase achieves the stated objectives:
1. Feature completeness and implementation status
2. Alignment with business requirements
3. Missing components or functionality
4. Over-engineering or scope creep
5. Value delivery assessment

{format_instructions}

Assess whether the code delivers on the promised objectives and identify gaps."""

    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> AgentFindings:
        """Analyze objective alignment."""
        context = self._prepare_files_context(files)
        files_info = f"Total files: {len(files)}, Objective alignment analysis"

        response = await self._call_llm(context, repo_info.project_objective, files_info)

        return AgentFindings(
            agent_name=self.name,
            findings=response.findings,
            confidence=response.confidence,
            reasoning=response.reasoning
        )


@register_agent('github_repository')
class GitHubRepositoryAgent(BaseAIAgent):
    """Agent specialized in GitHub repository analysis and metadata assessment."""

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(
            llm=llm,
            name="GitHub Repository Analyst",
            description="Analyzes GitHub repository structure, metadata, and community health",
            analysis_focus="Repository health, community engagement, and project maturity"
        )

    def _get_prompt_template(self) -> str:
        return """
You are a GitHub Repository Analysis Expert. Analyze the provided repository information and provide insights about:

CONTEXT:
{context}

PROJECT OBJECTIVE:
{objective}

REPOSITORY INFORMATION:
{files_info}

Based on the repository metadata, structure, and content, provide a comprehensive analysis covering:

1. **Repository Health & Maturity**
   - Code quality indicators
   - Documentation completeness
   - Testing coverage and practices
   - CI/CD pipeline effectiveness

2. **Community & Collaboration**
   - Open source friendliness
   - Contribution guidelines
   - Issue and PR management
   - Community engagement metrics

3. **Technical Architecture**
   - Technology stack assessment
   - Code organization and structure
   - Dependencies and security
   - Scalability and maintainability

4. **Project Viability**
   - Alignment with stated objectives
   - Market fit and relevance
   - Development velocity
   - Long-term sustainability

Provide specific, actionable findings with confidence scores and detailed reasoning.
{format_instructions}
"""

    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> AgentFindings:
        """Analyze GitHub repository health and structure."""
        # Prepare enhanced context with repository metadata
        context_parts = []

        # Add repository structure information
        context_parts.append("REPOSITORY STRUCTURE ANALYSIS:")
        context_parts.append(f"- Total files analyzed: {len(files)}")

        # Language distribution
        languages = {}
        total_size = 0
        for file in files:
            languages[file.language] = languages.get(file.language, 0) + 1
            total_size += file.size

        context_parts.append(f"- Total codebase size: {total_size:,} bytes")
        context_parts.append("- Language distribution:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            context_parts.append(f"  {lang}: {count} files")

        # File type analysis
        extensions = {}
        for file in files:
            ext = file.path.split('.')[-1] if '.' in file.path else 'no_extension'
            extensions[ext] = extensions.get(ext, 0) + 1

        context_parts.append("- File extensions:")
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]:
            context_parts.append(f"  .{ext}: {count} files")

        # Quality indicators
        test_files = [f for f in files if 'test' in f.path.lower() or 'spec' in f.path.lower()]
        config_files = [f for f in files if any(x in f.path.lower() for x in ['config', 'setup', 'requirements', 'package'])]
        doc_files = [f for f in files if any(x in f.path.lower() for x in ['readme', 'doc', 'docs', 'wiki', '.md'])]

        context_parts.append(f"- Test files: {len(test_files)}")
        context_parts.append(f"- Configuration files: {len(config_files)}")
        context_parts.append(f"- Documentation files: {len(doc_files)}")

        # Issue analysis summary
        issues_found = sum(len(f.issues) for f in files)
        context_parts.append(f"- Total issues identified: {issues_found}")

        context = "\n".join(context_parts)
        files_info = f"""
Repository Analysis Summary:
- Files: {len(files)}
- Languages: {len(languages)}
- Test Coverage: {len(test_files)} test files
- Documentation: {len(doc_files)} doc files
- Issues Found: {issues_found}
"""

        response = await self._call_llm(context, repo_info.project_objective, files_info)

        return AgentFindings(
            agent_name=self.name,
            findings=response.findings,
            confidence=response.confidence,
            reasoning=response.reasoning
        )