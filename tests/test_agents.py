"""Tests for agent implementations."""

import pytest
from omniscient_architect.agents import (
    AlignmentAgent,
    ReliabilityAgent,
    ArchitectureAgent,
    EfficiencyAgent,
)
from omniscient_architect.models import FileAnalysis, RepositoryInfo
from unittest.mock import Mock

@pytest.mark.asyncio
async def test_alignment_agent_analyze():
    agent = AlignmentAgent(
        llm=Mock(),
        name="AlignmentAgent",
        description="Checks code alignment",
        analysis_focus="alignment"
    )
    files = [FileAnalysis(path="file.py", size=123, language="Python")]
    repo_info = RepositoryInfo(path=None, project_objective="Test objective")
    agent.call_llm = Mock(return_value=Mock(findings=["Aligned"], confidence=1.0, reasoning="OK", recommendations=["None"]))
    result = await agent.analyze(files, repo_info)
    assert hasattr(result, "findings")
    assert result.findings == ["Aligned"]

@pytest.mark.asyncio
async def test_reliability_agent_analyze():
    agent = ReliabilityAgent(
        llm=Mock(),
        name="ReliabilityAgent",
        description="Checks reliability",
        analysis_focus="reliability"
    )
    files = [FileAnalysis(path="file.py", size=123, language="Python")]
    repo_info = RepositoryInfo(path=None, project_objective="Test objective")
    agent.call_llm = Mock(return_value=Mock(findings=["Reliable"], confidence=1.0, reasoning="OK", recommendations=["None"]))
    result = await agent.analyze(files, repo_info)
    assert hasattr(result, "findings")
    assert result.findings == ["Reliable"]

@pytest.mark.asyncio
async def test_architecture_agent_analyze():
    agent = ArchitectureAgent(
        llm=Mock(),
        name="ArchitectureAgent",
        description="Analyzes architecture",
        analysis_focus="architecture"
    )
    files = [FileAnalysis(path="file.py", size=123, language="Python")]
    repo_info = RepositoryInfo(path=None, project_objective="Test objective")
    agent.call_llm = Mock(return_value=Mock(findings=["Good architecture"], confidence=1.0, reasoning="OK", recommendations=["None"]))
    result = await agent.analyze(files, repo_info)
    assert hasattr(result, "findings")
    assert result.findings == ["Good architecture"]

@pytest.mark.asyncio
async def test_efficiency_agent_analyze():
    agent = EfficiencyAgent(
        llm=Mock(),
        name="EfficiencyAgent",
        description="Analyzes efficiency",
        analysis_focus="efficiency"
    )
    files = [FileAnalysis(path="file.py", size=123, language="Python")]
    repo_info = RepositoryInfo(path=None, project_objective="Test objective")
    agent.call_llm = Mock(return_value=Mock(findings=["Efficient"], confidence=1.0, reasoning="OK", recommendations=["None"]))
    result = await agent.analyze(files, repo_info)
    assert hasattr(result, "findings")
    assert result.findings == ["Efficient"]
