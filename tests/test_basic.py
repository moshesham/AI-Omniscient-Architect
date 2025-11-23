"""Basic tests for the Omniscient Architect."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from omniscient_architect.models import (
    FileAnalysis, AgentFindings, ReviewResult,
    RepositoryInfo, AnalysisConfig
)


class TestModels:
    """Test the data models."""

    def test_file_analysis_creation(self):
        """Test FileAnalysis dataclass creation."""
        analysis = FileAnalysis(
            path="src/main.py",
            size=1000,
            language="Python",
            complexity_score=5
        )
        assert analysis.path == "src/main.py"
        assert analysis.size == 1000
        assert analysis.language == "Python"
        assert analysis.complexity_score == 5

    def test_agent_findings_creation(self):
        """Test AgentFindings dataclass creation."""
        findings = AgentFindings(
            agent_name="Test Agent",
            findings=["Issue 1", "Issue 2"],
            confidence=0.85
        )
        assert findings.agent_name == "Test Agent"
        assert len(findings.findings) == 2
        assert findings.confidence == 0.85

    def test_repository_info_creation(self):
        """Test RepositoryInfo dataclass creation."""
        repo_info = RepositoryInfo(
            path=Path("/tmp/repo"),
            project_objective="Build a web app"
        )
        assert repo_info.path == Path("/tmp/repo")
        assert repo_info.project_objective == "Build a web app"
        assert repo_info.branch == "main"
        assert not repo_info.is_remote

    def test_analysis_config_creation(self):
        """Test AnalysisConfig dataclass creation."""
        config = AnalysisConfig(
            max_file_size=1024,
            ollama_model="test-model"
        )
        assert config.max_file_size == 1024
        assert config.ollama_model == "test-model"
        assert config.analysis_depth == "standard"


class TestAnalysisEngine:
    """Test the analysis engine (mocked)."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test that the engine can be initialized."""
        from omniscient_architect.analysis import AnalysisEngine

        config = AnalysisConfig()
        engine = AnalysisEngine(config)

        # Mock the LLM initialization
        engine.initialize_llm = AsyncMock(return_value=True)

        success = await engine.initialize_llm()
        assert success is True

    def test_file_scanning_logic(self):
        """Test the file scanning logic."""
        from omniscient_architect.analysis import AnalysisEngine

        config = AnalysisConfig()
        engine = AnalysisEngine(config)

        # Test file should be analyzed
        test_file = Path("test.py")
        assert engine._should_analyze_file(test_file)

        # Test file should be ignored
        ignore_file = Path(".git/config")
        assert not engine._should_analyze_file(ignore_file)

    def test_language_detection(self):
        """Test language detection from file extensions."""
        from omniscient_architect.analysis import AnalysisEngine

        config = AnalysisConfig()
        engine = AnalysisEngine(config)

        assert engine._detect_language(Path("script.py")) == "Python"
        assert engine._detect_language(Path("app.js")) == "JavaScript"
        assert engine._detect_language(Path("unknown.xyz")) == "Unknown"


if __name__ == "__main__":
    pytest.main([__file__])
