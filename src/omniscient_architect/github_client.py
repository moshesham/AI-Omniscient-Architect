"""GitHub API client for repository analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse
import httpx

from .models import RepositoryInfo
from .logging_config import get_logger

logger = get_logger(__name__)


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


@dataclass
class GitHubFile:
    """GitHub file information."""
    path: str
    name: str
    type: str  # "file" or "dir"
    size: int = 0
    download_url: Optional[str] = None
    url: Optional[str] = None
    sha: Optional[str] = None


class GitHubClient:
    """Client for interacting with GitHub API."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub client.

        Args:
            token: GitHub personal access token (optional, increases rate limits)
        """
        self.token = token
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Omniscient-Architect/1.0"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

        self.client = httpx.AsyncClient(
            headers=self.headers,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def get_repository(self, owner: str, repo: str) -> GitHubRepo:
        """Get repository information.

        Args:
            owner: Repository owner/organization
            repo: Repository name

        Returns:
            GitHubRepo object with repository details

        Raises:
            httpx.HTTPStatusError: If repository not found or access denied
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            return GitHubRepo(
                owner=data["owner"]["login"],
                name=data["name"],
                full_name=data["full_name"],
                description=data.get("description"),
                language=data.get("language"),
                stars=data.get("stargazers_count", 0),
                forks=data.get("forks_count", 0),
                default_branch=data.get("default_branch", "main"),
                topics=data.get("topics", []),
                size=data.get("size", 0),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
                clone_url=data.get("clone_url"),
                html_url=data.get("html_url")
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Repository {owner}/{repo} not found")
            elif e.response.status_code == 403:
                raise ValueError(f"Access denied to repository {owner}/{repo}. Check token permissions.")
            else:
                raise
        except Exception as e:
            logger.error(f"Error fetching repository {owner}/{repo}: {e}")
            raise

    async def get_repository_contents(
        self,
        owner: str,
        repo: str,
        path: str = "",
        branch: str = "main"
    ) -> List[GitHubFile]:
        """Get repository contents for a specific path.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Path within repository (empty for root)
            branch: Branch name

        Returns:
            List of GitHubFile objects
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch} if branch != "main" else {}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Handle single file response
            if isinstance(data, dict):
                data = [data]

            files = []
            for item in data:
                files.append(GitHubFile(
                    path=item["path"],
                    name=item["name"],
                    type=item["type"],
                    size=item.get("size", 0),
                    download_url=item.get("download_url"),
                    url=item.get("url"),
                    sha=item.get("sha")
                ))

            return files

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Path '{path}' not found in {owner}/{repo}")
            else:
                raise
        except Exception as e:
            logger.error(f"Error fetching contents for {owner}/{repo}/{path}: {e}")
            raise

    async def get_file_content(self, download_url: str) -> str:
        """Download file content from GitHub.

        Args:
            download_url: Direct download URL for the file

        Returns:
            File content as string
        """
        try:
            response = await self.client.get(download_url)
            response.raise_for_status()
            return response.text

        except Exception as e:
            logger.error(f"Error downloading file from {download_url}: {e}")
            raise

    async def get_repository_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """Get language breakdown for repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dictionary mapping language names to byte counts
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/languages"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Error fetching languages for {owner}/{repo}: {e}")
            raise

    def parse_github_url(self, url: str) -> tuple[str, str]:
        """Parse GitHub URL to extract owner and repo.

        Args:
            url: GitHub repository URL

        Returns:
            Tuple of (owner, repo)

        Raises:
            ValueError: If URL is not a valid GitHub repository URL
        """
        parsed = urlparse(url)

        # Handle different GitHub URL formats
        if "github.com" not in parsed.netloc:
            raise ValueError("Not a GitHub URL")

        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub repository URL")

        owner, repo = path_parts[0], path_parts[1]

        # Remove .git suffix if present
        if repo.endswith(".git"):
            repo = repo[:-4]

        return owner, repo


async def create_repository_info_from_github(
    github_url: str,
    token: Optional[str] = None,
    project_objective: str = ""
) -> RepositoryInfo:
    """Create RepositoryInfo from GitHub URL.

    Args:
        github_url: GitHub repository URL
        token: GitHub token (optional)
        project_objective: Description of project goals

    Returns:
        RepositoryInfo object configured for GitHub analysis
    """
    async with GitHubClient(token) as client:
        owner, repo = client.parse_github_url(github_url)

        # Get repository info
        github_repo = await client.get_repository(owner, repo)

        return RepositoryInfo(
            path=None,  # Will be set when cloned locally
            url=github_url,
            branch=github_repo.default_branch,
            is_remote=True,
            project_objective=project_objective
        )