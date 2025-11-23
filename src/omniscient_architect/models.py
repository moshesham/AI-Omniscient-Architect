"""Data models for the Omniscient Architect."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os


@dataclass
class FileAnalysis:
    """Stores analysis results for a single file."""
    path: str
    size: int
    language: str
    complexity_score: int = 0
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    content: Optional[str] = None


@dataclass
class AgentFindings:
    """Stores findings from each specialist agent."""
    agent_name: str
    findings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ReviewResult:
    """Complete review result."""
    project_understanding: str = ""
    goal_alignment_score: int = 0
    component_status: Dict[str, str] = field(default_factory=dict)
    strengths: List[Dict[str, str]] = field(default_factory=list)
    weaknesses: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    strategic_advice: Dict[str, str] = field(default_factory=dict)
    ai_insights: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryInfo:
    """Information about a repository to analyze."""
    path: Optional[Path]
    url: Optional[str] = None
    branch: str = "main"
    is_remote: bool = False
    project_objective: str = ""


@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    # Repo scan limits
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_files: int = 1000
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs", "*.md", "*.json", "*.yaml", "*.yml"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv"
    ])
    # Exclude heavy/binary extensions proactively
    exclude_extensions: List[str] = field(default_factory=lambda: [
        "png", "jpg", "jpeg", "gif", "webp", "svg", "pdf", "docx", "pptx", "xlsx",
        "zip", "tar", "gz", "ipynb", "mp4", "mov", "avi", "mp3", "wav", "bin", "exe", "dll"
    ])

    # LLM limits and batching
    max_content_bytes_per_file: int = 20 * 1024  # 20KB per file content preview
    max_files_for_llm: int = 40                  # cap number of files we feed to LLM
    max_total_bytes_for_llm: int = 500 * 1024    # overall byte budget for LLM context
    agent_concurrency: int = 2                   # number of agents to run concurrently
    file_scan_workers: int = 8                   # workers for file scanning
    sampling_strategy: str = "heuristic"         # "heuristic" | "random"

    # LLM configuration
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "codellama:7b-instruct"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    analysis_depth: str = field(default_factory=lambda: os.getenv("ANALYSIS_DEPTH", "standard"))  # "quick", "standard", "deep"