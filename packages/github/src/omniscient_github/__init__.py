"""Omniscient GitHub - GitHub integration for the Omniscient Architect platform.

This package provides comprehensive GitHub integration including:
- GitHubClient: Async client for GitHub API operations
- RepositoryScanner: Scan and discover repository files
- PullRequestManager: Create and manage PRs
- Rate limit handling with automatic backoff
"""

from omniscient_github.client import GitHubClient
from omniscient_github.models import (
    GitHubConfig,
    GitHubRepo,
    GitHubFile,
    BranchInfo,
    PullRequestInfo,
    CommitInfo,
)
from omniscient_github.scanner import RepositoryScanner
from omniscient_github.rate_limit import RateLimitHandler
from omniscient_github.utils import parse_github_url

__all__ = [
    # Client
    "GitHubClient",
    # Models
    "GitHubConfig",
    "GitHubRepo",
    "GitHubFile",
    "BranchInfo",
    "PullRequestInfo",
    "CommitInfo",
    # Scanner
    "RepositoryScanner",
    # Rate limiting
    "RateLimitHandler",
    # Utilities
    "parse_github_url",
]

__version__ = "0.1.0"
