"""Data models for the Omniscient Architect."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


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
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_files: int = 1000
    include_patterns: List[str] = field(default_factory=lambda: ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs"])
    exclude_patterns: List[str] = field(default_factory=lambda: [".git", "__pycache__", "node_modules", ".venv", "venv"])
    ollama_model: str = "codellama:7b-instruct"
    analysis_depth: str = "standard"  # "quick", "standard", "deep"