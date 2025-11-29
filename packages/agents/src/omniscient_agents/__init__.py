"""Omniscient Agents - AI analysis agents for code review."""

from .architecture import ArchitectureAgent
from .efficiency import EfficiencyAgent
from .reliability import ReliabilityAgent
from .alignment import AlignmentAgent
from .registry import AgentRegistry, get_agent, list_agents, register_agent
from .orchestrator import (
    AnalysisOrchestrator,
    StreamingOrchestrator,
    AnalysisResult,
    AnalysisProgress,
    AnalysisStatus,
    AnalysisTask,
)
from .llm_agent import LLMAgent, CodeReviewAgent, LLMAgentResponse, Issue

__version__ = "0.1.0"

__all__ = [
    # Agents
    "ArchitectureAgent",
    "EfficiencyAgent",
    "ReliabilityAgent",
    "AlignmentAgent",
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
]
