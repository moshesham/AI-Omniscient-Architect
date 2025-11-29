"""Omniscient Core - Models, config, and utilities for Omniscient Architect."""

from .models import (
    FileAnalysis,
    AgentFindings,
    ReviewResult,
    RepositoryInfo,
    AnalysisConfig,
    AnalysisStatus,
    AgentConfig,
    AnalysisJob,
)
from .config import load_config, get_config_path
from .logging import setup_logging, get_logger
from .base import BaseAIAgent, AgentResponse

__version__ = "0.1.0"

__all__ = [
    # Models
    "FileAnalysis",
    "AgentFindings",
    "ReviewResult",
    "RepositoryInfo",
    "AnalysisConfig",
    "AnalysisStatus",
    "AgentConfig",
    "AnalysisJob",
    # Config
    "load_config",
    "get_config_path",
    # Logging
    "setup_logging",
    "get_logger",
    # Base classes
    "BaseAIAgent",
    "AgentResponse",
]
