"""Agent Workspace - Shared directory for agent inter-communication and change tracking.

The AgentWorkspace provides a filesystem-backed coordination layer that lets
multiple agents write their findings, read what other agents have produced, and
track how those findings evolve across analysis sessions.

Directory layout
----------------
<workspace_root>/
├── <session_id>/
│   ├── manifest.json          # Session metadata and entry index
│   ├── architecture/
│   │   └── <timestamp>.json   # ArchitectureAgent entries
│   ├── security/
│   │   └── <timestamp>.json   # SecurityAgent entries
│   ├── efficiency/
│   │   └── <timestamp>.json   # EfficiencyAgent entries
│   ├── reliability/
│   │   └── <timestamp>.json   # ReliabilityAgent entries
│   ├── alignment/
│   │   └── <timestamp>.json   # AlignmentAgent entries
│   └── _shared/
│       └── <timestamp>.json   # Cross-agent notes / aggregated insights

Example usage
-------------
    workspace = AgentWorkspace(workspace_dir="./agent_workspace")

    # An agent writes its findings
    workspace.write_entry(
        agent_name="security",
        entry_type="findings",
        data={"issues": [...], "summary": "..."},
    )

    # Another agent reads findings from the security agent to avoid duplication
    security_findings = workspace.read_entries(agent_name="security")

    # The orchestrator writes a combined session summary
    summary = workspace.get_session_summary()
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceEntry:
    """A single record written by an agent into the workspace."""

    entry_id: str
    session_id: str
    agent_name: str
    entry_type: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkspaceEntry":
        """Deserialise from a dictionary (e.g. loaded from JSON)."""
        return cls(
            entry_id=raw.get("entry_id", ""),
            session_id=raw.get("session_id", ""),
            agent_name=raw.get("agent_name", ""),
            entry_type=raw.get("entry_type", "findings"),
            timestamp=raw.get("timestamp", ""),
            data=raw.get("data", {}),
            tags=raw.get("tags", []),
            metadata=raw.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)


@dataclass
class SessionManifest:
    """Tracks metadata for a single analysis session."""

    session_id: str
    created_at: str
    updated_at: str
    repository: str = ""
    branch: str = ""
    total_entries: int = 0
    agents: List[str] = field(default_factory=list)
    entry_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SessionManifest":
        return cls(
            session_id=raw.get("session_id", ""),
            created_at=raw.get("created_at", ""),
            updated_at=raw.get("updated_at", ""),
            repository=raw.get("repository", ""),
            branch=raw.get("branch", ""),
            total_entries=raw.get("total_entries", 0),
            agents=raw.get("agents", []),
            entry_ids=raw.get("entry_ids", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# AgentWorkspace
# ---------------------------------------------------------------------------

class AgentWorkspace:
    """Filesystem-backed shared workspace for agent coordination.

    All writes are atomic (write-then-rename) and protected by an exclusive
    POSIX lock so multiple concurrent agents cannot corrupt shared files.

    Parameters
    ----------
    workspace_dir:
        Root directory for all workspace data.  Will be created if it does
        not already exist.
    session_id:
        Identifier for the current analysis session.  A new UUID is generated
        automatically when not provided.
    repository:
        Optional repository name stored in the session manifest.
    branch:
        Optional branch name stored in the session manifest.
    """

    _MANIFEST_FILE = "manifest.json"
    _SHARED_DIR = "_shared"

    def __init__(
        self,
        workspace_dir: str | Path = "./agent_workspace",
        session_id: Optional[str] = None,
        repository: str = "",
        branch: str = "",
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id: str = session_id or str(uuid.uuid4())
        self._session_dir = self.workspace_dir / self.session_id
        self._manifest_path = self._session_dir / self._MANIFEST_FILE
        self._lock_path = self._session_dir / ".lock"

        self._session_dir.mkdir(parents=True, exist_ok=True)
        (self._session_dir / self._SHARED_DIR).mkdir(exist_ok=True)

        self._init_manifest(repository=repository, branch=branch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _lock_file(self):
        """Return an open file object that holds an exclusive lock.

        Callers are responsible for releasing the lock via ``fcntl.flock``.
        """
        fh = open(self._lock_path, "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
        return fh

    def _release_lock(self, fh) -> None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()

    def _write_json_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        """Write *data* to *path* atomically via a temp file + rename."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    # ------------------------------------------------------------------
    # Manifest management
    # ------------------------------------------------------------------

    def _init_manifest(self, repository: str = "", branch: str = "") -> None:
        """Create the manifest if it does not already exist."""
        if self._manifest_path.exists():
            return
        manifest = SessionManifest(
            session_id=self.session_id,
            created_at=self._now_iso(),
            updated_at=self._now_iso(),
            repository=repository,
            branch=branch,
        )
        self._write_json_atomic(self._manifest_path, manifest.to_dict())

    def _load_manifest(self) -> SessionManifest:
        return SessionManifest.from_dict(self._read_json(self._manifest_path))

    def _update_manifest(self, entry: WorkspaceEntry) -> None:
        """Add *entry* metadata to the manifest under an exclusive lock."""
        lock = self._lock_file()
        try:
            manifest = self._load_manifest()
            manifest.updated_at = self._now_iso()
            manifest.total_entries += 1
            if entry.agent_name not in manifest.agents:
                manifest.agents.append(entry.agent_name)
            manifest.entry_ids.append(entry.entry_id)
            self._write_json_atomic(self._manifest_path, manifest.to_dict())
        finally:
            self._release_lock(lock)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_entry(
        self,
        agent_name: str,
        data: Dict[str, Any],
        entry_type: str = "findings",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkspaceEntry:
        """Write agent findings to the workspace.

        Parameters
        ----------
        agent_name:
            Name of the agent writing the entry (e.g. ``"security"``).
        data:
            The actual findings / payload to persist.
        entry_type:
            Logical type of the entry (default ``"findings"``).
        tags:
            Optional list of string tags for filtering.
        metadata:
            Optional extra key/value pairs (model name, duration, etc.).

        Returns
        -------
        WorkspaceEntry
            The entry that was written.
        """
        ts = self._now_iso()
        entry = WorkspaceEntry(
            entry_id=str(uuid.uuid4()),
            session_id=self.session_id,
            agent_name=agent_name,
            entry_type=entry_type,
            timestamp=ts,
            data=data,
            tags=tags or [],
            metadata=metadata or {},
        )

        agent_dir = self._session_dir / agent_name
        agent_dir.mkdir(exist_ok=True)

        # File name uses a timestamp-safe slug
        filename = ts.replace(":", "-").replace(".", "-") + ".json"
        entry_path = agent_dir / filename
        self._write_json_atomic(entry_path, entry.to_dict())

        self._update_manifest(entry)
        return entry

    def read_entries(
        self,
        agent_name: Optional[str] = None,
        entry_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[WorkspaceEntry]:
        """Read workspace entries, optionally filtered.

        Parameters
        ----------
        agent_name:
            Only return entries from this agent.  ``None`` returns all agents.
        entry_type:
            Only return entries with this type.
        tags:
            Only return entries that include *all* of the given tags.
        limit:
            Maximum number of entries to return (most-recent first).

        Returns
        -------
        List[WorkspaceEntry]
            Matching entries sorted newest-first.
        """
        entries: List[WorkspaceEntry] = []

        # Determine which agent directories to scan
        if agent_name:
            dirs = [self._session_dir / agent_name]
        else:
            dirs = [
                p for p in self._session_dir.iterdir()
                if p.is_dir() and p.name not in {self._SHARED_DIR}
                and not p.name.startswith(".")
            ]

        for agent_dir in dirs:
            if not agent_dir.exists():
                continue
            for json_file in sorted(agent_dir.glob("*.json"), reverse=True):
                raw = self._read_json(json_file)
                if not raw:
                    continue
                entry = WorkspaceEntry.from_dict(raw)

                if entry_type and entry.entry_type != entry_type:
                    continue
                if tags and not all(t in entry.tags for t in tags):
                    continue

                entries.append(entry)

        # Sort newest-first across all agents
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:limit] if limit else entries

    def get_latest_entry(
        self,
        agent_name: str,
        entry_type: Optional[str] = None,
    ) -> Optional[WorkspaceEntry]:
        """Return the most-recent entry from *agent_name*.

        Parameters
        ----------
        agent_name:
            Agent whose latest entry to retrieve.
        entry_type:
            If provided, only entries with this type are considered.

        Returns
        -------
        WorkspaceEntry or None
        """
        results = self.read_entries(
            agent_name=agent_name,
            entry_type=entry_type,
            limit=1,
        )
        return results[0] if results else None

    def list_agents(self) -> List[str]:
        """Return the names of all agents that have written to this session."""
        manifest = self._load_manifest()
        return list(manifest.agents)

    def get_session_summary(self) -> Dict[str, Any]:
        """Return a summary dictionary for the current session.

        The summary includes manifest metadata, per-agent entry counts, and
        the most-recent entry from each agent.
        """
        manifest = self._load_manifest()
        all_entries = self.read_entries()

        per_agent: Dict[str, List[WorkspaceEntry]] = {}
        for entry in all_entries:
            per_agent.setdefault(entry.agent_name, []).append(entry)

        agent_summaries = {}
        for name, agent_entries in per_agent.items():
            latest = agent_entries[0]  # already sorted newest-first
            agent_summaries[name] = {
                "total_entries": len(agent_entries),
                "latest_timestamp": latest.timestamp,
                "latest_entry_type": latest.entry_type,
                "latest_tags": latest.tags,
            }

        return {
            "session_id": self.session_id,
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
            "repository": manifest.repository,
            "branch": manifest.branch,
            "total_entries": manifest.total_entries,
            "agents": manifest.agents,
            "agent_summaries": agent_summaries,
        }

    def write_shared(
        self,
        key: str,
        data: Dict[str, Any],
        author: str = "orchestrator",
    ) -> Path:
        """Write a cross-agent shared note to the ``_shared`` directory.

        Parameters
        ----------
        key:
            Logical key identifying the shared resource (e.g. ``"summary"``).
        data:
            Payload to persist.
        author:
            Name of the component writing the note.

        Returns
        -------
        Path
            Path to the written file.
        """
        shared_dir = self._session_dir / self._SHARED_DIR
        shared_dir.mkdir(exist_ok=True)

        payload = {
            "key": key,
            "author": author,
            "timestamp": self._now_iso(),
            "data": data,
        }
        out_path = shared_dir / f"{key}.json"
        lock = self._lock_file()
        try:
            self._write_json_atomic(out_path, payload)
        finally:
            self._release_lock(lock)
        return out_path

    def read_shared(self, key: str) -> Optional[Dict[str, Any]]:
        """Read a shared note by key.

        Returns
        -------
        dict or None
        """
        path = self._session_dir / self._SHARED_DIR / f"{key}.json"
        if not path.exists():
            return None
        raw = self._read_json(path)
        return raw.get("data") if raw else None

    def clear_session(self) -> None:
        """Delete all data for the current session.

        .. warning::
            This operation is irreversible.
        """
        import shutil
        if self._session_dir.exists():
            shutil.rmtree(self._session_dir)

    @classmethod
    def list_sessions(cls, workspace_dir: str | Path = "./agent_workspace") -> List[str]:
        """List all session IDs found in *workspace_dir*.

        Returns
        -------
        List[str]
            Session IDs sorted newest-first by manifest ``created_at``.
        """
        root = Path(workspace_dir)
        if not root.exists():
            return []

        sessions: List[tuple[str, str]] = []
        for item in root.iterdir():
            if item.is_dir():
                manifest_path = item / cls._MANIFEST_FILE
                if manifest_path.exists():
                    try:
                        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                        sessions.append((item.name, raw.get("created_at", "")))
                    except (json.JSONDecodeError, OSError):
                        sessions.append((item.name, ""))

        sessions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sessions]
