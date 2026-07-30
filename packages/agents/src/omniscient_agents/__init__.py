"""Omniscient Agents - AI analysis agents for code review."""

from omniscient_core import AnalysisStatus
from .architecture import ArchitectureAgent
from .efficiency import EfficiencyAgent
from .reliability import ReliabilityAgent
from .alignment import AlignmentAgent
from .security import SecurityAgent
from .registry import AgentRegistry, get_agent, list_agents, register_agent
from .orchestrator import (
    AnalysisOrchestrator,
    StreamingOrchestrator,
    AnalysisResult,
    AnalysisProgress,
    AnalysisTask,
)
from .llm_agent import LLMAgent, CodeReviewAgent, LLMAgentResponse, Issue
from .workspace import AgentWorkspace, WorkspaceEntry, SessionManifest

__version__ = "0.2.0"

__all__ = [
    # Agents
    "ArchitectureAgent",
    "EfficiencyAgent",
    "ReliabilityAgent",
    "AlignmentAgent",
    "SecurityAgent",
    # LLM Agents
    "LLMAgent",
    "CodeReviewAgent",
    "LLMAgentResponse",
    "Issue",
    # Registry
    "AgentRegistry",
    "get_agent",
    "list_agents",
    "register_agent",
    # Orchestrator
    "AnalysisOrchestrator",
    "StreamingOrchestrator",
    "AnalysisResult",
    "AnalysisProgress",
    "AnalysisStatus",
    "AnalysisTask",
    # Workspace
    "AgentWorkspace",
    "WorkspaceEntry",
    "SessionManifest",
]
