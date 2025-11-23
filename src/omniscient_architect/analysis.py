"""Core analysis engine for the Omniscient Architect."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseLanguageModel

from .models import (
    FileAnalysis, AgentFindings, ReviewResult,
    RepositoryInfo, AnalysisConfig
)
from .agents import (
    ArchitectureAgent, EfficiencyAgent,
    ReliabilityAgent, AlignmentAgent
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

    async def _check_ollama_server(self, base_url: str) -> bool:
        """Check if Ollama server is responding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def _check_model_available(self, base_url: str, model: str) -> bool:
        """Check if the specified model is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    return model in models
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
        return False

    async def _pull_model(self, model: str):
        """Pull the model using docker exec."""
        try:
            logger.info(f"Pulling model {model}")
            process = await asyncio.create_subprocess_exec(
                "docker", "exec", "omniscient-architect-ollama", "ollama", "pull", model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Capture output and wait
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Failed to pull model: {stderr.decode()}")
                raise subprocess.CalledProcessError(process.returncode or 1, "ollama pull")
            else:
                logger.info(f"Model {model} pulled successfully")
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            raise

    async def _start_ollama(self):
        """Attempt to start Ollama using docker-compose."""
        try:
            project_root = Path(__file__).parent.parent.parent
            logger.info(f"Starting Ollama from {project_root}")
            process = await asyncio.create_subprocess_exec(
                "docker-compose", "up", "-d", "ollama",
                cwd=str(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Capture output and wait
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Docker-compose failed: {stderr.decode()}")
                raise subprocess.CalledProcessError(process.returncode or 1, "docker-compose")
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            raise

    async def initialize_llm(self) -> bool:
        """Initialize the LLM connection."""
        try:
            base_url = self.config.ollama_host or "http://localhost:11434"
            
            # Check if server is up
            if not await self._check_ollama_server(base_url):
                logger.info("Ollama server not responding, attempting to start...")
                await self._start_ollama()
                # Wait for startup
                await asyncio.sleep(10)
                if not await self._check_ollama_server(base_url):
                    logger.error("Ollama server still not responding after startup attempt")
                    return False
            
            # Check if model is available
            if not await self._check_model_available(base_url, self.config.ollama_model):
                logger.info(f"Model {self.config.ollama_model} not available, pulling...")
                await self._pull_model(self.config.ollama_model)
                # Wait for pull
                await asyncio.sleep(30)
                if not await self._check_model_available(base_url, self.config.ollama_model):
                    logger.error(f"Model {self.config.ollama_model} still not available after pull")
                    return False
            
            self.llm = ChatOllama(model=self.config.ollama_model, base_url=base_url)
            # Test the connection
            await self.llm.ainvoke("Hello")
            logger.info(f"Successfully initialized LLM: {self.config.ollama_model}")

            # Initialize agents using the registry (respect enabled_agents in config)
            assert self.llm is not None
            self.agents = [
                ArchitectureAgent(self.llm),
                EfficiencyAgent(self.llm),
                ReliabilityAgent(self.llm),
                AlignmentAgent(self.llm),
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

        files: List[FileAnalysis] = []
        loop = asyncio.get_event_loop()

        # Remote GitHub repository ingestion
        if getattr(repo_info, "is_remote", False) and repo_info.url:
            try:
                from .github_client import GitHubClient
                import os

                token = os.getenv("GITHUB_TOKEN")
                async with GitHubClient(token) as gh:
                    owner, repo = gh.parse_github_url(repo_info.url)
                    branch = repo_info.branch or "main"

                    queue = [""]  # start at repo root
                    seen_paths = set()

                    while queue and len(files) < self.config.max_files:
                        path = queue.pop(0)
                        if path in seen_paths:
                            continue
                        seen_paths.add(path)

                        try:
                            entries = await gh.get_repository_contents(owner, repo, path=path, branch=branch)
                        except Exception as e:
                            logger.warning(f"GitHub contents fetch failed for '{path}': {e}")
                            continue

                        for item in entries:
                            if item.type == "dir":
                                if len(files) < self.config.max_files:
                                    queue.append(item.path)
                                continue

                            if item.type != "file":
                                continue

                            remote_path = Path(item.path)
                            if not self._should_analyze_file(remote_path):
                                continue

                            # Skip too large files without downloading
                            if item.size and item.size > self.config.max_file_size:
                                logger.debug(f"Skipping large remote file: {item.path} ({item.size} bytes)")
                                continue

                            # Optionally download small files only (preview)
                            content: Optional[str] = None
                            try:
                                if item.download_url and item.size <= self.config.max_content_bytes_per_file:
                                    text = await gh.get_file_content(item.download_url)
                                    # Truncate to preview budget
                                    content = text[: self.config.max_content_bytes_per_file]
                            except Exception as e:
                                logger.debug(f"Failed to fetch content for {item.path}: {e}")

                            analysis = FileAnalysis(
                                path=item.path,
                                size=int(item.size or 0),
                                language=self._detect_language(remote_path),
                                content=content
                            )
                            analysis.complexity_score = self._calculate_complexity(content or "", analysis.language)
                            files.append(analysis)

                            if len(files) >= self.config.max_files:
                                break

                logger.info(f"Ingested {len(files)} files from GitHub")
                return files
            except Exception as e:
                logger.error(f"Remote ingestion failed, falling back to local scanning: {e}")

        if repo_info.path is None or not isinstance(repo_info.path, Path):
            logger.warning("Repository path is not set; skipping local file ingestion.")
            return files
        base_path: Path = cast(Path, repo_info.path)

        def scan_files():
            """Scan repository files synchronously."""
            scanned_files = []
            for file_path in base_path.rglob('*'):
                if file_path.is_file() and self._should_analyze_file(file_path):
                    try:
                        # Read file content
                        max_bytes = max(0, self.config.max_content_bytes_per_file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(max_bytes if max_bytes > 0 else None)

                        if len(content) > self.config.max_file_size:
                            logger.warning(f"File too large, skipping: {file_path}")
                            continue

                        analysis = FileAnalysis(
                            path=str(file_path.relative_to(base_path)),
                            size=len(content),
                            language=self._detect_language(file_path),
                            content=content  # store preview only (already size-limited)
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
        with ThreadPoolExecutor(max_workers=self.config.file_scan_workers) as executor:
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

        # Exclude by heavy/binary extensions
        ext = file_path.suffix.lower().lstrip('.')
        if ext in set(self.config.exclude_extensions):
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
        """Run all agents with budgeted context and limited concurrency."""
        logger.info("Phase 2: Running multi-agent analysis")

        if not self.agents:
            logger.error("No agents initialized")
            return []

        # Adjust budgets based on depth
        max_files_for_llm, max_total_bytes_for_llm = self._depth_budgets()

        # Select a subset of files for LLM context
        selected_files = self._select_files_for_llm(files, max_files_for_llm, max_total_bytes_for_llm)
        logger.info(
            f"Selected {len(selected_files)} files for LLM context (depth={self.config.analysis_depth})"
        )

        # Limit concurrent agent execution to reduce memory spikes
        sem = asyncio.Semaphore(max(1, self.config.agent_concurrency))

        async def run_agent(agent):
            async with sem:
                return await agent.analyze(selected_files, repo_info)

        tasks = [run_agent(agent) for agent in self.agents]
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

    def _depth_budgets(self) -> Tuple[int, int]:
        """Compute effective budgets based on analysis depth."""
        depth = (self.config.analysis_depth or "standard").lower()
        base_files = self.config.max_files_for_llm
        base_bytes = self.config.max_total_bytes_for_llm
        if depth == "quick":
            return max(5, base_files // 4), max(64 * 1024, base_bytes // 4)
        if depth == "deep":
            return base_files * 2, base_bytes * 2
        return base_files, base_bytes

    def _select_files_for_llm(
        self,
        files: List[FileAnalysis],
        max_files: int,
        max_bytes: int
    ) -> List[FileAnalysis]:
        """Heuristically select a representative subset of files under given budgets.

        Preference order:
        - Source files in primary languages (py, ts, js, go, rs, java)
        - Files with higher complexity scores
        - Keep some docs/config for context
        """
        if not files:
            return []

        def priority(f: FileAnalysis) -> int:
            lang_weight = 3 if f.language in {"Python", "TypeScript", "JavaScript", "Go", "Rust", "Java"} else 1
            doc_cfg = 1 if f.language in {"Markdown", "YAML", "JSON"} else 0
            return lang_weight * 100 + f.complexity_score + doc_cfg * 10

        # Sort by priority descending
        sorted_files = sorted(files, key=priority, reverse=True)

        selected: List[FileAnalysis] = []
        byte_sum = 0
        for f in sorted_files:
            if len(selected) >= max_files:
                break
            next_bytes = f.size
            if byte_sum + next_bytes > max_bytes:
                continue
            selected.append(f)
            byte_sum += next_bytes

        # Fallback: if nothing selected due to strict byte budget, pick the smallest ones
        if not selected:
            smallest = sorted(files, key=lambda x: x.size)[: max(1, max_files // 4)]
            return smallest

        return selected

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
                text = finding.lower()
                # Skip error/system messages
                if any(err in text for err in [
                    'analysis failed', 'invalid json', 'output_parsing_failure', 'langchain.com/oss'
                ]):
                    continue
                if any(keyword in text for keyword in ['good', 'strong', 'excellent', 'well', '✅']):
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
                text = finding.lower()
                # Skip error/system messages from LLM
                if any(err in text for err in [
                    'analysis failed', 'invalid json', 'output_parsing_failure', 'langchain.com/oss'
                ]):
                    continue
                if any(keyword in text for keyword in ['issue', 'problem', 'warning', '⚠️', '❌']):
                    category = 'Reliability'  # Default category
                    if 'efficiency' in findings.agent_name.lower() or 'complexity' in text:
                        category = 'Efficiency'
                    elif 'accuracy' in text or 'logic' in text:
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