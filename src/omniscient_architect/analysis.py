"""Core analysis engine for the Omniscient Architect."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLanguageModel

from .models import (
    FileAnalysis, AgentFindings, ReviewResult,
    RepositoryInfo, AnalysisConfig
)
from .agents import (
    ArchitectureAgent, EfficiencyAgent,
    ReliabilityAgent, AlignmentAgent, GitHubRepositoryAgent
)


logger = logging.getLogger(__name__)


class AnalysisEngine:
    """Main analysis engine coordinating AI agents."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.llm: Optional[BaseLanguageModel] = None
        self.agents = []

        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def initialize_llm(self) -> bool:
        """Initialize the LLM connection."""
        try:
            self.llm = Ollama(model=self.config.ollama_model, base_url=self.config.ollama_host)
            # Test the connection
            await self.llm.ainvoke("Hello")
            logger.info(f"Successfully initialized LLM: {self.config.ollama_model}")

            # Initialize agents (llm is guaranteed to be not None here)
            assert self.llm is not None
            self.agents = [
                ArchitectureAgent(self.llm),
                EfficiencyAgent(self.llm),
                ReliabilityAgent(self.llm),
                AlignmentAgent(self.llm),
                GitHubRepositoryAgent(self.llm),
            ]
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return False

    async def analyze_repository(
        self,
        repo_info: RepositoryInfo
    ) -> ReviewResult:
        """Perform complete analysis of a repository."""
        logger.info(f"Starting analysis of repository: {repo_info.path}")

        # Phase 1: Ingest and analyze files
        files = await self._ingest_files(repo_info)

        # Phase 2: Multi-agent analysis
        agent_findings = await self._run_multi_agent_analysis(files, repo_info)

        # Phase 3: Generate comprehensive report
        result = await self._generate_report(files, agent_findings, repo_info)

        logger.info("Analysis completed successfully")
        return result

    async def _ingest_files(self, repo_info: RepositoryInfo) -> List[FileAnalysis]:
        """Ingest and analyze files from the repository."""
        logger.info("Phase 1: Ingesting files")

        files = []
        loop = asyncio.get_event_loop()

        def scan_files():
            """Scan repository files synchronously."""
            scanned_files = []
            for file_path in repo_info.path.rglob('*'):
                if file_path.is_file() and self._should_analyze_file(file_path):
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        if len(content) > self.config.max_file_size:
                            logger.warning(f"File too large, skipping: {file_path}")
                            continue

                        analysis = FileAnalysis(
                            path=str(file_path.relative_to(repo_info.path)),
                            size=len(content),
                            language=self._detect_language(file_path),
                            content=content if len(content) < 50000 else None  # Store content for smaller files
                        )

                        # Calculate basic complexity
                        analysis.complexity_score = self._calculate_complexity(content, analysis.language)

                        scanned_files.append(analysis)

                        if len(scanned_files) >= self.config.max_files:
                            logger.warning(f"Reached maximum file limit: {self.config.max_files}")
                            break

                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
                        continue

            return scanned_files

        # Run file scanning in thread pool to avoid blocking
        with ThreadPoolExecutor() as executor:
            files = await loop.run_in_executor(executor, scan_files)

        logger.info(f"Ingested {len(files)} files")
        return files

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed."""
        # Check exclude patterns
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if pattern in path_str:
                return False

        # Check include patterns
        for pattern in self.config.include_patterns:
            if file_path.match(pattern):
                return True

        return False

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.html': 'HTML',
            '.css': 'CSS',
            '.sql': 'SQL',
        }
        return extension_map.get(file_path.suffix.lower(), 'Unknown')

    def _calculate_complexity(self, content: str, language: str) -> int:
        """Calculate basic complexity score."""
        if language == 'Python':
            # Count control structures and functions
            complexity = 0
            complexity += len([line for line in content.split('\n') if line.strip().startswith(('if ', 'for ', 'while ', 'def ', 'class '))])
            return min(complexity, 100)  # Cap at 100
        return 0

    async def _run_multi_agent_analysis(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> List[AgentFindings]:
        """Run all agents in parallel."""
        logger.info("Phase 2: Running multi-agent analysis")

        if not self.agents:
            logger.error("No agents initialized")
            return []

        # Run all agents concurrently
        tasks = [agent.analyze(files, repo_info) for agent in self.agents]
        findings = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        valid_findings = []
        for i, result in enumerate(findings):
            if isinstance(result, Exception):
                logger.error(f"Agent {self.agents[i].name} failed: {result}")
                # Create fallback findings
                valid_findings.append(AgentFindings(
                    agent_name=self.agents[i].name,
                    findings=[f"Analysis failed: {str(result)}"],
                    confidence=0.0
                ))
            else:
                valid_findings.append(result)

        logger.info(f"Completed analysis with {len(valid_findings)} agents")
        return valid_findings

    async def _generate_report(
        self,
        files: List[FileAnalysis],
        agent_findings: List[AgentFindings],
        repo_info: RepositoryInfo
    ) -> ReviewResult:
        """Generate the comprehensive review report."""
        logger.info("Phase 3: Generating comprehensive report")

        result = ReviewResult()

        # Basic project understanding
        result.project_understanding = self._generate_project_understanding(files, repo_info)

        # Component status
        result.component_status = self._analyze_components(files)

        # Goal alignment score
        result.goal_alignment_score = self._calculate_alignment_score(agent_findings, repo_info)

        # Process agent findings
        result.strengths = self._extract_strengths(agent_findings)
        result.weaknesses = self._extract_weaknesses(agent_findings)

        # Strategic advice
        result.strategic_advice = self._generate_strategic_advice(files, agent_findings, repo_info)

        # AI insights
        result.ai_insights = self._generate_ai_insights(agent_findings)

        return result

    def _generate_project_understanding(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo
    ) -> str:
        """Generate project understanding summary."""
        languages = set(f.language for f in files if f.language != 'Unknown')
        total_size = sum(f.size for f in files)

        understanding = (
            f"This repository contains {len(files)} analyzed files "
            f"spanning {len(languages)} programming languages ({', '.join(languages)}). "
            f"Total codebase size: {total_size:,} bytes."
        )

        if repo_info.project_objective:
            understanding += f" The stated objective is: {repo_info.project_objective}"

        return understanding

    def _analyze_components(self, files: List[FileAnalysis]) -> Dict[str, str]:
        """Analyze project components."""
        components = {}

        # Categorize files
        has_tests = any('test' in f.path.lower() for f in files)
        has_docs = any(f.language == 'Markdown' for f in files)
        has_config = any(f.language in ['JSON', 'YAML'] for f in files)

        components['Testing'] = f"Present ({sum(1 for f in files if 'test' in f.path.lower())} files)" if has_tests else "Missing"
        components['Documentation'] = f"Present ({sum(1 for f in files if f.language == 'Markdown')} files)" if has_docs else "Missing"
        components['Configuration'] = f"Present ({sum(1 for f in files if f.language in ['JSON', 'YAML'])} files)" if has_config else "Missing"

        return components

    def _calculate_alignment_score(
        self,
        agent_findings: List[AgentFindings],
        repo_info: RepositoryInfo
    ) -> int:
        """Calculate goal alignment score."""
        if not repo_info.project_objective:
            return 50  # Neutral score without objective

        # Simple heuristic based on agent confidence
        avg_confidence = sum(f.confidence for f in agent_findings) / len(agent_findings)
        return int(avg_confidence * 100)

    def _extract_strengths(self, agent_findings: List[AgentFindings]) -> List[Dict[str, str]]:
        """Extract strengths from agent findings."""
        strengths = []
        for findings in agent_findings:
            for finding in findings.findings:
                if any(keyword in finding.lower() for keyword in ['good', 'strong', 'excellent', 'well', '✅']):
                    strengths.append({
                        'strength': finding.replace('✅', '').strip(),
                        'evidence': f"Identified by {findings.agent_name}",
                        'why_it_matters': "Contributes to overall code quality and maintainability"
                    })
        return strengths[:5]  # Limit to top 5

    def _extract_weaknesses(self, agent_findings: List[AgentFindings]) -> Dict[str, List[Dict[str, str]]]:
        """Extract weaknesses from agent findings."""
        weaknesses = {
            'Efficiency': [],
            'Accuracy': [],
            'Reliability': []
        }

        for findings in agent_findings:
            for finding in findings.findings:
                if any(keyword in finding.lower() for keyword in ['issue', 'problem', 'warning', '⚠️', '❌']):
                    category = 'Reliability'  # Default category
                    if 'efficiency' in findings.agent_name.lower() or 'complexity' in finding.lower():
                        category = 'Efficiency'
                    elif 'accuracy' in finding.lower() or 'logic' in finding.lower():
                        category = 'Accuracy'

                    weaknesses[category].append({
                        'issue': finding.replace('⚠️', '').replace('❌', '').strip(),
                        'location': 'Codebase',
                        'fix': 'Review and refactor based on agent recommendations'
                    })

        return weaknesses

    def _generate_strategic_advice(
        self,
        files: List[FileAnalysis],
        agent_findings: List[AgentFindings],
        repo_info: RepositoryInfo
    ) -> Dict[str, str]:
        """Generate strategic advice."""
        return {
            'scalability': (
                "Focus on modular architecture and clean interfaces. "
                "Consider implementing proper dependency injection and "
                "design patterns for better scalability."
            ),
            'future_proofing': (
                "Add comprehensive testing, implement CI/CD pipelines, "
                "and establish coding standards. Consider API versioning "
                "and feature flags for gradual rollouts."
            ),
            'broader_application': (
                f"This codebase shows potential for expansion into "
                f"related domains while maintaining the core {self._infer_project_type(files)} architecture."
            )
        }

    def _generate_ai_insights(self, agent_findings: List[AgentFindings]) -> Dict[str, Any]:
        """Generate AI-specific insights."""
        return {
            'agent_confidence': {f.agent_name: f.confidence for f in agent_findings},
            'key_themes': self._extract_key_themes(agent_findings),
            'recommendation_priority': ['Security', 'Testing', 'Documentation', 'Performance']
        }

    def _extract_key_themes(self, agent_findings: List[AgentFindings]) -> List[str]:
        """Extract key themes from agent findings."""
        themes = []
        all_findings = [f for findings in agent_findings for f in findings.findings]

        if any('security' in f.lower() for f in all_findings):
            themes.append('Security')
        if any('test' in f.lower() for f in all_findings):
            themes.append('Testing')
        if any('documentation' in f.lower() or 'docs' in f.lower() for f in all_findings):
            themes.append('Documentation')
        if any('performance' in f.lower() or 'efficiency' in f.lower() for f in all_findings):
            themes.append('Performance')

        return themes or ['Code Quality']

    def _infer_project_type(self, files: List[FileAnalysis]) -> str:
        """Infer the project type from files."""
        languages = set(f.language for f in files)

        if 'Python' in languages and any('streamlit' in f.path.lower() for f in files):
            return "Streamlit web application"
        elif 'JavaScript' in languages or 'TypeScript' in languages:
            return "web application"
        elif 'Markdown' in languages and len([f for f in files if f.language == 'Markdown']) > 5:
            return "documentation project"
        elif 'Python' in languages:
            return "Python software project"

        return "software project"