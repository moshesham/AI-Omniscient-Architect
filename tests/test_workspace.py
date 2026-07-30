"""Tests for the AgentWorkspace inter-agent coordination system."""

import json
import sys
import time
from pathlib import Path
import pytest

# Ensure the agents package is importable
_root = Path(__file__).parent.parent
for _pkg in ["core", "agents"]:
    _path = _root / "packages" / _pkg / "src"
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from omniscient_agents.workspace import AgentWorkspace, WorkspaceEntry, SessionManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def workspace(tmp_path):
    """Return a fresh AgentWorkspace backed by a temp directory."""
    return AgentWorkspace(
        workspace_dir=tmp_path / "ws",
        session_id="test-session-001",
        repository="my-repo",
        branch="main",
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_created_on_init(self, workspace):
        manifest_path = workspace._manifest_path
        assert manifest_path.exists(), "manifest.json should be created on init"

    def test_manifest_contents(self, workspace):
        manifest = workspace._load_manifest()
        assert manifest.session_id == "test-session-001"
        assert manifest.repository == "my-repo"
        assert manifest.branch == "main"
        assert manifest.total_entries == 0
        assert manifest.agents == []

    def test_manifest_not_overwritten_on_second_init(self, tmp_path):
        ws1 = AgentWorkspace(
            workspace_dir=tmp_path / "ws",
            session_id="s1",
            repository="repo-a",
        )
        ws1.write_entry("arch", {"findings": ["f1"]})

        # Re-open the same session; manifest should not be reset
        ws2 = AgentWorkspace(
            workspace_dir=tmp_path / "ws",
            session_id="s1",
        )
        manifest = ws2._load_manifest()
        assert manifest.total_entries == 1


# ---------------------------------------------------------------------------
# write_entry / read_entries
# ---------------------------------------------------------------------------

class TestWriteRead:
    def test_write_creates_file(self, workspace, tmp_path):
        entry = workspace.write_entry("security", {"issues": ["SQL injection"]})
        agent_dir = workspace._session_dir / "security"
        files = list(agent_dir.glob("*.json"))
        assert len(files) == 1

    def test_written_entry_is_valid_json(self, workspace, tmp_path):
        workspace.write_entry("arch", {"findings": ["coupling too high"]})
        agent_dir = workspace._session_dir / "arch"
        json_file = list(agent_dir.glob("*.json"))[0]
        data = json.loads(json_file.read_text())
        assert data["agent_name"] == "arch"
        assert data["data"]["findings"] == ["coupling too high"]

    def test_read_entries_returns_written(self, workspace):
        workspace.write_entry("efficiency", {"bottlenecks": ["N+1 query"]})
        entries = workspace.read_entries(agent_name="efficiency")
        assert len(entries) == 1
        assert entries[0].data["bottlenecks"] == ["N+1 query"]

    def test_read_all_agents(self, workspace):
        workspace.write_entry("security", {"issues": []})
        workspace.write_entry("arch", {"findings": []})
        all_entries = workspace.read_entries()
        assert len(all_entries) == 2

    def test_read_filtered_by_agent(self, workspace):
        workspace.write_entry("security", {"issues": ["xss"]})
        workspace.write_entry("arch", {"findings": ["tight coupling"]})
        sec = workspace.read_entries(agent_name="security")
        assert len(sec) == 1
        assert sec[0].agent_name == "security"

    def test_read_filtered_by_entry_type(self, workspace):
        workspace.write_entry("arch", {"x": 1}, entry_type="findings")
        workspace.write_entry("arch", {"x": 2}, entry_type="summary")
        findings = workspace.read_entries(entry_type="findings")
        assert all(e.entry_type == "findings" for e in findings)
        assert len(findings) == 1

    def test_read_filtered_by_tags(self, workspace):
        workspace.write_entry("security", {"a": 1}, tags=["critical", "owasp"])
        workspace.write_entry("security", {"b": 2}, tags=["low"])
        critical = workspace.read_entries(tags=["critical"])
        assert len(critical) == 1
        assert critical[0].data == {"a": 1}

    def test_read_with_limit(self, workspace):
        for i in range(5):
            workspace.write_entry("arch", {"i": i})
            time.sleep(0.01)  # ensure distinct timestamps
        limited = workspace.read_entries(agent_name="arch", limit=3)
        assert len(limited) == 3

    def test_entries_sorted_newest_first(self, workspace):
        for i in range(3):
            workspace.write_entry("arch", {"seq": i})
            time.sleep(0.01)
        entries = workspace.read_entries(agent_name="arch")
        timestamps = [e.timestamp for e in entries]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_manifest_updated_after_write(self, workspace):
        workspace.write_entry("security", {"issues": []})
        workspace.write_entry("arch", {"findings": []})
        manifest = workspace._load_manifest()
        assert manifest.total_entries == 2
        assert set(manifest.agents) == {"security", "arch"}


# ---------------------------------------------------------------------------
# get_latest_entry
# ---------------------------------------------------------------------------

class TestGetLatest:
    def test_returns_none_for_unknown_agent(self, workspace):
        assert workspace.get_latest_entry("nonexistent") is None

    def test_returns_most_recent(self, workspace):
        workspace.write_entry("arch", {"seq": 1})
        time.sleep(0.01)
        workspace.write_entry("arch", {"seq": 2})
        latest = workspace.get_latest_entry("arch")
        assert latest is not None
        assert latest.data["seq"] == 2

    def test_filtered_by_entry_type(self, workspace):
        workspace.write_entry("arch", {"x": 1}, entry_type="findings")
        workspace.write_entry("arch", {"x": 2}, entry_type="summary")
        latest_findings = workspace.get_latest_entry("arch", entry_type="findings")
        assert latest_findings is not None
        assert latest_findings.entry_type == "findings"


# ---------------------------------------------------------------------------
# list_agents / get_session_summary
# ---------------------------------------------------------------------------

class TestSessionInfo:
    def test_list_agents_empty(self, workspace):
        assert workspace.list_agents() == []

    def test_list_agents_after_writes(self, workspace):
        workspace.write_entry("security", {})
        workspace.write_entry("arch", {})
        agents = workspace.list_agents()
        assert set(agents) == {"security", "arch"}

    def test_get_session_summary(self, workspace):
        workspace.write_entry("security", {"issues": ["sqli"]})
        workspace.write_entry("arch", {"findings": []})
        summary = workspace.get_session_summary()
        assert summary["session_id"] == "test-session-001"
        assert summary["repository"] == "my-repo"
        assert summary["total_entries"] == 2
        assert "security" in summary["agent_summaries"]
        assert "arch" in summary["agent_summaries"]


# ---------------------------------------------------------------------------
# Shared notes
# ---------------------------------------------------------------------------

class TestShared:
    def test_write_and_read_shared(self, workspace):
        workspace.write_shared("summary", {"score": 0.9}, author="orchestrator")
        data = workspace.read_shared("summary")
        assert data is not None
        assert data["score"] == 0.9

    def test_read_missing_shared_returns_none(self, workspace):
        assert workspace.read_shared("nonexistent") is None

    def test_write_shared_overwrites(self, workspace):
        workspace.write_shared("summary", {"score": 0.5})
        workspace.write_shared("summary", {"score": 0.8})
        data = workspace.read_shared("summary")
        assert data["score"] == 0.8


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_list_sessions_empty(self, tmp_path):
        sessions = AgentWorkspace.list_sessions(tmp_path / "empty")
        assert sessions == []

    def test_list_sessions_finds_sessions(self, tmp_path):
        ws_dir = tmp_path / "ws"
        AgentWorkspace(workspace_dir=ws_dir, session_id="s1")
        time.sleep(0.01)
        AgentWorkspace(workspace_dir=ws_dir, session_id="s2")
        sessions = AgentWorkspace.list_sessions(ws_dir)
        assert set(sessions) == {"s1", "s2"}


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------

class TestClearSession:
    def test_clear_removes_session_dir(self, workspace, tmp_path):
        workspace.write_entry("arch", {})
        assert workspace._session_dir.exists()
        workspace.clear_session()
        assert not workspace._session_dir.exists()
