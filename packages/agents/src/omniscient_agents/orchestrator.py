"""Analysis Orchestrator - Parallel analysis execution pipeline."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from omniscient_core import (
    BaseAIAgent,
    AgentResponse,
    FileAnalysis,
    RepositoryInfo,
)
from omniscient_core.logging import get_logger

logger = get_logger(__name__)


class AnalysisStatus(str, Enum):
    """Status of an analysis task."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AnalysisTask:
    """Represents a single analysis task."""
    
    agent_name: str
    agent: BaseAIAgent
    status: AnalysisStatus = AnalysisStatus.PENDING
    result: Optional[AgentResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class AnalysisProgress:
    """Progress update for analysis pipeline."""
    
    total_tasks: int
    completed_tasks: int
    running_tasks: int
    failed_tasks: int
    current_agents: List[str]
    
    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total_tasks == 0:
            return 100.0
        return (self.completed_tasks / self.total_tasks) * 100


@dataclass
class AnalysisResult:
    """Complete result of orchestrated analysis."""
    
    repository: RepositoryInfo
    files_analyzed: int
    agent_responses: Dict[str, AgentResponse] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get total duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def success_count(self) -> int:
        """Count of successful analyses."""
        return len(self.agent_responses)
    
    @property
    def failure_count(self) -> int:
        """Count of failed analyses."""
        return len(self.errors)
    
    def get_all_issues(self) -> List[Dict[str, Any]]:
        """Get all issues from all agents."""
        issues = []
        for agent_name, response in self.agent_responses.items():
            for issue in response.issues:
                issues.append({
                    "agent": agent_name,
                    "severity": issue.severity,
                    "category": issue.category,
                    "description": issue.description,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                })
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        all_issues = self.get_all_issues()
        
        severity_counts = {}
        for issue in all_issues:
            sev = issue["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            "repository": self.repository.name,
            "files_analyzed": self.files_analyzed,
            "agents_run": self.success_count,
            "agents_failed": self.failure_count,
            "total_issues": len(all_issues),
            "issues_by_severity": severity_counts,
            "duration_seconds": self.duration_seconds,
        }


class AnalysisOrchestrator:
    """Orchestrates parallel analysis across multiple agents.
    
    Features:
    - Parallel execution of independent agents
    - Progress streaming
    - Configurable concurrency
    - Error handling and retry
    - Result aggregation
    
    Example:
        >>> from omniscient_agents import ArchitectureAgent, EfficiencyAgent
        >>> 
        >>> orchestrator = AnalysisOrchestrator(
        ...     agents=[ArchitectureAgent(), EfficiencyAgent()],
        ...     max_concurrent=2,
        ... )
        >>> 
        >>> async for progress in orchestrator.analyze_with_progress(files, repo_info):
        ...     print(f"Progress: {progress.percent_complete:.1f}%")
        >>> 
        >>> result = orchestrator.get_result()
    """
    
    def __init__(
        self,
        agents: List[BaseAIAgent],
        max_concurrent: int = 3,
        retry_failed: bool = True,
        max_retries: int = 2,
    ):
        """Initialize orchestrator.
        
        Args:
            agents: List of agents to run
            max_concurrent: Maximum concurrent analyses
            retry_failed: Whether to retry failed analyses
            max_retries: Maximum retry attempts per agent
        """
        self.agents = agents
        self.max_concurrent = max_concurrent
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        
        self._tasks: Dict[str, AnalysisTask] = {}
        self._result: Optional[AnalysisResult] = None
        self._cancelled = False
    
    def _create_tasks(self) -> None:
        """Create analysis tasks for all agents."""
        self._tasks = {
            agent.name: AnalysisTask(
                agent_name=agent.name,
                agent=agent,
            )
            for agent in self.agents
        }
    
    async def _run_agent_task(
        self,
        task: AnalysisTask,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo,
        retry_count: int = 0,
    ) -> None:
        """Run a single agent task.
        
        Args:
            task: Analysis task to run
            files: Files to analyze
            repo_info: Repository information
            retry_count: Current retry attempt
        """
        if self._cancelled:
            task.status = AnalysisStatus.CANCELLED
            return
        
        task.status = AnalysisStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            logger.info(f"Starting analysis: {task.agent_name}")
            
            result = await task.agent.analyze(files, repo_info)
            
            task.result = result
            task.status = AnalysisStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(
                f"Completed analysis: {task.agent_name} "
                f"({len(result.issues)} issues found)"
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {task.agent_name} - {e}")
            
            if self.retry_failed and retry_count < self.max_retries:
                logger.info(f"Retrying {task.agent_name} (attempt {retry_count + 1})")
                task.status = AnalysisStatus.PENDING
                await self._run_agent_task(task, files, repo_info, retry_count + 1)
            else:
                task.status = AnalysisStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
    
    def _get_progress(self) -> AnalysisProgress:
        """Get current progress."""
        completed = sum(1 for t in self._tasks.values() if t.status == AnalysisStatus.COMPLETED)
        running = sum(1 for t in self._tasks.values() if t.status == AnalysisStatus.RUNNING)
        failed = sum(1 for t in self._tasks.values() if t.status == AnalysisStatus.FAILED)
        current = [t.agent_name for t in self._tasks.values() if t.status == AnalysisStatus.RUNNING]
        
        return AnalysisProgress(
            total_tasks=len(self._tasks),
            completed_tasks=completed,
            running_tasks=running,
            failed_tasks=failed,
            current_agents=current,
        )
    
    async def analyze(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo,
    ) -> AnalysisResult:
        """Run all analyses and return result.
        
        Args:
            files: Files to analyze
            repo_info: Repository information
            
        Returns:
            AnalysisResult with all agent responses
        """
        self._create_tasks()
        self._result = AnalysisResult(
            repository=repo_info,
            files_analyzed=len(files),
            started_at=datetime.now(),
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def run_with_semaphore(task: AnalysisTask):
            async with semaphore:
                await self._run_agent_task(task, files, repo_info)
        
        # Run all tasks concurrently
        await asyncio.gather(
            *[run_with_semaphore(task) for task in self._tasks.values()],
            return_exceptions=True,
        )
        
        # Collect results
        for name, task in self._tasks.items():
            if task.status == AnalysisStatus.COMPLETED and task.result:
                self._result.agent_responses[name] = task.result
            elif task.status == AnalysisStatus.FAILED:
                self._result.errors[name] = task.error or "Unknown error"
        
        self._result.completed_at = datetime.now()
        return self._result
    
    async def analyze_with_progress(
        self,
        files: List[FileAnalysis],
        repo_info: RepositoryInfo,
        progress_interval: float = 0.5,
    ) -> AsyncIterator[AnalysisProgress]:
        """Run analyses with progress updates.
        
        Args:
            files: Files to analyze
            repo_info: Repository information
            progress_interval: Seconds between progress updates
            
        Yields:
            AnalysisProgress updates
        """
        self._create_tasks()
        self._result = AnalysisResult(
            repository=repo_info,
            files_analyzed=len(files),
            started_at=datetime.now(),
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def run_with_semaphore(task: AnalysisTask):
            async with semaphore:
                await self._run_agent_task(task, files, repo_info)
        
        # Start all tasks
        tasks = [
            asyncio.create_task(run_with_semaphore(task))
            for task in self._tasks.values()
        ]
        
        # Yield progress until all complete
        all_tasks = asyncio.gather(*tasks, return_exceptions=True)
        
        while not all_tasks.done():
            yield self._get_progress()
            await asyncio.sleep(progress_interval)
        
        # Wait for final completion
        await all_tasks
        
        # Final progress
        yield self._get_progress()
        
        # Collect results
        for name, task in self._tasks.items():
            if task.status == AnalysisStatus.COMPLETED and task.result:
                self._result.agent_responses[name] = task.result
            elif task.status == AnalysisStatus.FAILED:
                self._result.errors[name] = task.error or "Unknown error"
        
        self._result.completed_at = datetime.now()
    
    def cancel(self) -> None:
        """Cancel ongoing analyses."""
        self._cancelled = True
        logger.info("Analysis cancelled")
    
    def get_result(self) -> Optional[AnalysisResult]:
        """Get the analysis result.
        
        Returns:
            AnalysisResult if analysis completed, None otherwise
        """
        return self._result
    
    def get_task_status(self) -> Dict[str, AnalysisStatus]:
        """Get status of all tasks.
        
        Returns:
            Dict mapping agent names to their status
        """
        return {name: task.status for name, task in self._tasks.items()}


class StreamingOrchestrator(AnalysisOrchestrator):
    """Orchestrator with streaming token output.
    
    Extends base orchestrator to support streaming LLM responses
    for real-time output display.
    """
    
    def __init__(
        self,
        agents: List[BaseAIAgent],
        max_concurrent: int = 1,  # Serial for streaming
        **kwargs,
    ):
        """Initialize streaming orchestrator.
        
        Note: max_concurrent defaults to 1 for cleaner streaming output.
        """
        super().__init__(agents, max_concurrent, **kwargs)
        self._stream_callbacks: List[Callable[[str, str], None]] = []
    
    def add_stream_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Add callback for streaming tokens.
        
        Callback receives (agent_name, token) for each token.
        """
        self._stream_callbacks.append(callback)
    
    async def _emit_token(self, agent_name: str, token: str) -> None:
        """Emit a streaming token to all callbacks."""
        for callback in self._stream_callbacks:
            try:
                callback(agent_name, token)
            except Exception:
                pass  # Don't let callback errors break streaming
