"""Omniscient Tools - Analysis tools for code review."""

from .complexity import ComplexityAnalyzer
from .cache import AnalysisCache
from .file_scanner import FileScanner

__version__ = "0.1.0"

__all__ = [
    "ComplexityAnalyzer",
    "AnalysisCache",
    "FileScanner",
]

# Optional imports for clustering
try:
    from .clustering import SemanticClusterer
    __all__.append("SemanticClusterer")
except ImportError:
    pass  # clustering extras not installed
