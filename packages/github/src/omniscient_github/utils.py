"""Utility functions for GitHub operations."""

from urllib.parse import urlparse
from typing import Tuple, Optional
import re


def parse_github_url(url: str) -> Tuple[str, str]:
    """Parse GitHub URL to extract owner and repository name.
    
    Supports various GitHub URL formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - git@github.com:owner/repo.git
    - github.com/owner/repo
    - owner/repo (shorthand)
    
    Args:
        url: GitHub repository URL or shorthand
        
    Returns:
        Tuple of (owner, repository_name)
        
    Raises:
        ValueError: If URL is not a valid GitHub repository URL
        
    Examples:
        >>> parse_github_url("https://github.com/microsoft/vscode")
        ('microsoft', 'vscode')
        >>> parse_github_url("git@github.com:python/cpython.git")
        ('python', 'cpython')
        >>> parse_github_url("owner/repo")
        ('owner', 'repo')
    """
    url = url.strip()
    
    # Handle shorthand format (owner/repo)
    if re.match(r'^[\w.-]+/[\w.-]+$', url):
        parts = url.split('/')
        return parts[0], parts[1].rstrip('.git')
    
    # Handle SSH format (git@github.com:owner/repo.git)
    ssh_match = re.match(r'^git@github\.com:(.+)/(.+)$', url)
    if ssh_match:
        owner = ssh_match.group(1)
        repo = ssh_match.group(2).rstrip('.git')
        return owner, repo
    
    # Handle HTTPS URLs
    parsed = urlparse(url)
    
    # Add scheme if missing
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    
    # Validate GitHub domain
    if "github.com" not in parsed.netloc.lower():
        raise ValueError(f"Not a GitHub URL: {url}")
    
    # Extract path parts
    path_parts = parsed.path.strip("/").split("/")
    
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    
    owner = path_parts[0]
    repo = path_parts[1]
    
    # Remove .git suffix
    if repo.endswith(".git"):
        repo = repo[:-4]
    
    # Validate owner and repo names
    if not owner or not repo:
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    
    return owner, repo


def build_github_url(owner: str, repo: str, https: bool = True) -> str:
    """Build GitHub URL from owner and repo.
    
    Args:
        owner: Repository owner
        repo: Repository name
        https: Use HTTPS format (True) or SSH format (False)
        
    Returns:
        GitHub URL string
    """
    if https:
        return f"https://github.com/{owner}/{repo}"
    else:
        return f"git@github.com:{owner}/{repo}.git"


def build_file_url(
    owner: str,
    repo: str,
    path: str,
    branch: str = "main",
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
) -> str:
    """Build URL to a file in a GitHub repository.
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: File path within repository
        branch: Branch name
        line_start: Starting line number (optional)
        line_end: Ending line number (optional)
        
    Returns:
        URL to the file
    """
    url = f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"
    
    if line_start:
        url += f"#L{line_start}"
        if line_end and line_end != line_start:
            url += f"-L{line_end}"
    
    return url


def build_raw_url(owner: str, repo: str, path: str, branch: str = "main") -> str:
    """Build raw content URL for a file.
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: File path within repository
        branch: Branch name
        
    Returns:
        Raw content URL
    """
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def build_api_url(endpoint: str, api_base: str = "https://api.github.com") -> str:
    """Build GitHub API URL.
    
    Args:
        endpoint: API endpoint (e.g., "/repos/owner/repo")
        api_base: API base URL
        
    Returns:
        Full API URL
    """
    endpoint = endpoint.lstrip("/")
    return f"{api_base}/{endpoint}"


def is_github_url(url: str) -> bool:
    """Check if a URL is a GitHub URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is a GitHub URL
    """
    try:
        parse_github_url(url)
        return True
    except ValueError:
        return False


def extract_pr_number(url: str) -> Optional[int]:
    """Extract pull request number from GitHub URL.
    
    Args:
        url: GitHub PR URL
        
    Returns:
        PR number or None if not a PR URL
    """
    match = re.search(r'/pull/(\d+)', url)
    if match:
        return int(match.group(1))
    return None


def extract_issue_number(url: str) -> Optional[int]:
    """Extract issue number from GitHub URL.
    
    Args:
        url: GitHub issue URL
        
    Returns:
        Issue number or None if not an issue URL
    """
    match = re.search(r'/issues/(\d+)', url)
    if match:
        return int(match.group(1))
    return None


def normalize_path(path: str) -> str:
    """Normalize file path for GitHub API.
    
    Args:
        path: File path
        
    Returns:
        Normalized path without leading/trailing slashes
    """
    return path.strip("/").replace("\\", "/")
