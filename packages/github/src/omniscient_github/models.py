"""Data models for GitHub integration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field


class GitHubConfig(BaseModel):
    """Configuration for GitHub client."""
    
    token: Optional[str] = Field(default=None, description="GitHub API token")
    api_url: str = Field(default="https://api.github.com", description="GitHub API URL")
    per_page: int = Field(default=100, description="Items per page for paginated requests")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    backoff_factor: float = Field(default=2.0, description="Backoff factor for retries")
    
    class Config:
        extra = "forbid"


@dataclass
class GitHubRepo:
    """GitHub repository information."""
    
    owner: str
    name: str
    full_name: str
    description: Optional[str] = None
    language: Optional[str] = None
    stars: int = 0
    forks: int = 0
    default_branch: str = "main"
    topics: List[str] = field(default_factory=list)
    size: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    clone_url: Optional[str] = None
    html_url: Optional[str] = None
    is_private: bool = False
    is_fork: bool = False
    open_issues_count: int = 0
    license_name: Optional[str] = None
    
    @property
    def repo_id(self) -> str:
        """Unique repository identifier."""
        return f"{self.owner}/{self.name}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "owner": self.owner,
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "language": self.language,
            "stars": self.stars,
            "forks": self.forks,
            "default_branch": self.default_branch,
            "topics": self.topics,
            "size": self.size,
            "clone_url": self.clone_url,
            "html_url": self.html_url,
        }


@dataclass
class GitHubFile:
    """GitHub file information."""
    
    path: str
    name: str
    type: str  # "file", "dir", "symlink", "submodule"
    size: int = 0
    sha: Optional[str] = None
    download_url: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    encoding: Optional[str] = None
    
    @property
    def is_file(self) -> bool:
        """Check if this is a file."""
        return self.type == "file"
    
    @property
    def is_directory(self) -> bool:
        """Check if this is a directory."""
        return self.type == "dir"
    
    @property
    def extension(self) -> Optional[str]:
        """Get file extension."""
        if "." in self.name:
            return self.name.rsplit(".", 1)[1].lower()
        return None


@dataclass
class BranchInfo:
    """GitHub branch information."""
    
    name: str
    sha: str
    protected: bool = False
    commit_message: Optional[str] = None
    commit_date: Optional[str] = None
    commit_author: Optional[str] = None


@dataclass  
class CommitInfo:
    """GitHub commit information."""
    
    sha: str
    message: str
    author_name: str
    author_email: str
    date: str
    url: Optional[str] = None
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0


class PullRequestState(str, Enum):
    """Pull request state."""
    
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


@dataclass
class PullRequestInfo:
    """GitHub pull request information."""
    
    number: int
    title: str
    body: Optional[str]
    state: str
    head_branch: str
    base_branch: str
    html_url: str
    created_at: str
    updated_at: Optional[str] = None
    merged_at: Optional[str] = None
    author: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    changed_files: int = 0
    additions: int = 0
    deletions: int = 0
    mergeable: Optional[bool] = None
    
    @property
    def is_merged(self) -> bool:
        """Check if PR is merged."""
        return self.merged_at is not None
    
    @property
    def is_open(self) -> bool:
        """Check if PR is open."""
        return self.state == "open"


@dataclass
class RateLimitInfo:
    """GitHub API rate limit information."""
    
    limit: int
    remaining: int
    reset_timestamp: int
    resource: str = "core"
    
    @property
    def reset_datetime(self) -> datetime:
        """Get reset time as datetime."""
        return datetime.fromtimestamp(self.reset_timestamp)
    
    @property
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        return self.remaining <= 0
    
    @property
    def seconds_until_reset(self) -> int:
        """Seconds until rate limit resets."""
        return max(0, self.reset_timestamp - int(datetime.now().timestamp()))


@dataclass
class ScanResult:
    """Result of repository scanning."""
    
    repository: GitHubRepo
    files: List[GitHubFile] = field(default_factory=list)
    total_size: int = 0
    file_count: int = 0
    language_breakdown: dict = field(default_factory=dict)
    scan_duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if scan had errors."""
        return len(self.errors) > 0
