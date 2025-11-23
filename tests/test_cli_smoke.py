import argparse
import asyncio
from pathlib import Path

import pytest

from omniscient_architect.cli import CLI
from omniscient_architect.models import ReviewResult


@pytest.mark.asyncio
async def test_cli_runs_with_monkeypatched_engine(tmp_path, monkeypatch):
    # Arrange: create fixture repo
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# sample")

    # Create a dummy ReviewResult
    dummy = ReviewResult()
    dummy.project_understanding = "Sample repo"
    dummy.goal_alignment_score = 75
    dummy.component_status = {"Testing": "Missing", "Documentation": "Present (1 files)", "Configuration": "Missing"}
    dummy.strengths = []
    dummy.weaknesses = {}
    dummy.strategic_advice = {"scalability": "none", "future_proofing": "none", "broader_application": "none"}
    dummy.ai_insights = {}

    # Monkeypatch AnalysisEngine to avoid LLM calls
    from omniscient_architect.analysis import AnalysisEngine

    async def fake_init(self):
        return True

    async def fake_analyze(self, repo_info):
        return dummy

    monkeypatch.setattr(AnalysisEngine, "initialize_llm", fake_init)
    monkeypatch.setattr(AnalysisEngine, "analyze_repository", fake_analyze)

    # Build args namespace similar to CLI parsing
    parser = CLI.create_parser()
    args = parser.parse_args([str(repo_dir)])

    # Act: run CLI
    cli = CLI()
    rc = await cli.run(args)

    # Assert: run returns 0 (success) and didn't crash
    assert rc == 0
