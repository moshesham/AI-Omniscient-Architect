"""Omniscient Architect - AI-powered code review and development assistant."""

__version__ = "0.2.0"
__author__ = "AI Omniscient Architect Team"
__description__ = "AI-powered code review and development assistant"

# Import basic modules that don't require external dependencies
from .models import (
    FileAnalysis, AgentFindings, ReviewResult,
    RepositoryInfo, AnalysisConfig
)

# Try to import optional modules
try:
    from .reporting import ReportGenerator  # noqa: F401
    _reporting_available = True
except ImportError:
    _reporting_available = False

try:
    from .agents import (  # noqa: F401
        BaseAIAgent, ArchitectureAgent, EfficiencyAgent,
        ReliabilityAgent, AlignmentAgent, GitHubRepositoryAgent
    )
    _agents_available = True
except ImportError:
    _agents_available = False

try:
    from .analysis import AnalysisEngine  # noqa: F401
    _analysis_available = True
except ImportError:
    _analysis_available = False

try:
    from .cli import CLI  # noqa: F401
    _cli_available = True
except ImportError:
    _cli_available = False

try:
    from .github_client import GitHubClient, create_repository_info_from_github  # noqa: F401
    _github_available = True
except ImportError:
    _github_available = False

# Build __all__ based on what's available
__all__ = [
    "FileAnalysis", "AgentFindings", "ReviewResult",
    "RepositoryInfo", "AnalysisConfig"
]

if _reporting_available:
    __all__.append("ReportGenerator")

if _agents_available:
    __all__.extend([
        "BaseAIAgent", "ArchitectureAgent", "EfficiencyAgent",
        "ReliabilityAgent", "AlignmentAgent", "GitHubRepositoryAgent"
    ])

if _analysis_available:
    __all__.append("AnalysisEngine")

if _cli_available:
    __all__.append("CLI")

if _github_available:
    __all__.extend(["GitHubClient", "create_repository_info_from_github"]) 
