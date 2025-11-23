from __future__ import annotations

"""Pydantic schemas for agent outputs and parsed results."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Finding(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class AgentOutput(BaseModel):
    agent_name: str = Field(..., description="Agent identifier, e.g. 'architecture'")
    findings: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    raw_output: Optional[str] = None
    parse_error: Optional[str] = None


__all__ = ["Finding", "AgentOutput"]
