"""Omniscient Core - Models, config, and utilities for Omniscient Architect."""

from .models import (
    FileAnalysis,
    AgentFindings,
    ReviewResult,
    RepositoryInfo,
    AnalysisConfig,
    AgentConfig,
    AnalysisJob,
)
from .enums import (
    AnalysisStatus,
    Severity,
    FindingCategory,
)
from .language_utils import detect_language, get_language_for_ast, LANGUAGE_EXTENSIONS, CODE_EXTENSIONS, EXCLUDE_DIRS
from .config import load_config, get_config_path
from .logging import setup_logging, get_logger
from .base import BaseAIAgent, AgentResponse
from .mixins import AsyncContextMixin

__version__ = "0.1.0"

__all__ = [
    # Models
    "FileAnalysis",
    "AgentFindings",
    "ReviewResult",
    "RepositoryInfo",
    "AnalysisConfig",
    "AgentConfig",
    "AnalysisJob",
    # Enums
    "AnalysisStatus",
    "Severity",
    "FindingCategory",
    # Language utils
    "detect_language",
    "get_language_for_ast",
    "LANGUAGE_EXTENSIONS",
    "CODE_EXTENSIONS",
    "EXCLUDE_DIRS",
    # Config
    "load_config",
    "get_config_path",
    # Logging
    "setup_logging",
    "get_logger",
    # Base classes
    "BaseAIAgent",
    "AgentResponse",
    # Mixins
    "AsyncContextMixin",
]
