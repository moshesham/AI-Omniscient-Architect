"""Omniscient Architect package public surface.

Keep imports lightweight at package import time. Heavy or optional
components (reporting, agents, analysis, CLI) are imported lazily
below and added to ``__all__`` only when available.
"""

__version__ = "0.2.0"

# Core model exports (always importable)
from .models import (
    FileAnalysis, AgentFindings, ReviewResult,
    RepositoryInfo, AnalysisConfig,
)

__all__ = [
    "FileAnalysis", "AgentFindings", "ReviewResult",
    "RepositoryInfo", "AnalysisConfig",
]

# Optional components: import if available and expose names
try:
    from .reporting import ReportGenerator  # noqa: F401
    __all__.append("ReportGenerator")
except Exception:
    pass

try:
    from .agents import (  # noqa: F401
        BaseAIAgent, ArchitectureAgent, EfficiencyAgent,
        ReliabilityAgent, AlignmentAgent, GitHubRepositoryAgent,
    )
    __all__.extend([
        "BaseAIAgent", "ArchitectureAgent", "EfficiencyAgent",
        "ReliabilityAgent", "AlignmentAgent", "GitHubRepositoryAgent",
    ])
except Exception:
    pass

try:
    from .analysis import AnalysisEngine  # noqa: F401
    __all__.append("AnalysisEngine")
except Exception:
    pass

try:
    from .cli import CLI  # noqa: F401
    __all__.append("CLI")
except Exception:
    pass

try:
    from .github_client import GitHubClient, create_repository_info_from_github  # noqa: F401
    __all__.extend(["GitHubClient", "create_repository_info_from_github"])
except Exception:
    pass

