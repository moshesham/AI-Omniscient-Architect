# Omniscient GitHub

GitHub integration package for the Omniscient Architect platform. Provides repository access, file retrieval, and PR workflow support.

## Installation

```bash
pip install omniscient-github
```

## Features

- **GitHubClient**: High-level client for repository operations
- **RepositoryScanner**: Scan repositories for analysis
- **PullRequestManager**: Create and manage review PRs
- **RateLimitHandler**: Automatic rate limit handling

## Quick Start

### Basic Repository Access

```python
from omniscient_github import GitHubClient

# Initialize with token
client = GitHubClient(token="ghp_your_token")

# Get repository info
repo = client.get_repository("owner/repo")
print(f"Repository: {repo.name}")
print(f"Language: {repo.language}")
print(f"Stars: {repo.stargazers_count}")

# List files
files = client.list_files("owner/repo", path="src")
for file in files:
    print(f"  {file.path} ({file.size} bytes)")
```

### Scanning for Analysis

```python
from omniscient_github import RepositoryScanner

scanner = RepositoryScanner(token="ghp_your_token")

# Scan a repository
result = await scanner.scan_repository(
    "owner/repo",
    include_patterns=["*.py", "*.js"],
    exclude_patterns=["**/test_*", "**/node_modules/**"],
    max_file_size=100_000,  # 100KB
)

print(f"Found {len(result.files)} files")
print(f"Total size: {result.total_size} bytes")
```

### Pull Request Workflow

```python
from omniscient_github import PullRequestManager

pr_manager = PullRequestManager(token="ghp_your_token")

# Create a review PR
pr = await pr_manager.create_review_pr(
    repo="owner/repo",
    title="Code Review: Architecture Analysis",
    body="## Analysis Results\n\n...",
    branch_name="code-review/architecture",
    files_to_add={
        "docs/review-report.md": report_content,
    }
)

print(f"Created PR: {pr.html_url}")
```

### With Rate Limit Handling

```python
from omniscient_github import GitHubClient, RateLimitHandler

# Automatic retry with backoff
client = GitHubClient(
    token="ghp_your_token",
    rate_limit_handler=RateLimitHandler(
        max_retries=3,
        backoff_factor=2.0,
    )
)
```

## Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token
- `GITHUB_API_URL`: Custom GitHub API URL (for Enterprise)

## Configuration

```python
from omniscient_github import GitHubConfig

config = GitHubConfig(
    token="ghp_your_token",
    api_url="https://api.github.com",
    per_page=100,
    timeout=30.0,
    max_retries=3,
)

client = GitHubClient(config=config)
```

## Models

```python
from omniscient_github import (
    RepositoryInfo,
    FileInfo,
    BranchInfo,
    PullRequestInfo,
    CommitInfo,
)
```

## License

MIT License
