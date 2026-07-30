"""Omniscient Tools - Analysis tools for code review."""

from .complexity import ComplexityAnalyzer
from .cache import AnalysisCache
from .file_scanner import FileScanner
from .smart_prioritizer import SmartPrioritizer, ScoredFile, SymbolInfo

__version__ = "0.1.0"

__all__ = [
    "ComplexityAnalyzer",
    "AnalysisCache",
    "FileScanner",
    "SmartPrioritizer",
    "ScoredFile",
    "SymbolInfo",
]

# Optional imports for clustering
try:
    from .clustering import SemanticClusterer
    __all__.append("SemanticClusterer")
except ImportError:
    pass  # clustering extras not installed
