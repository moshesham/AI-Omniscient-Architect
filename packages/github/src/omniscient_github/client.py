"""GitHub API client implementation."""

from typing import Dict, List, Optional, Any
import httpx

from omniscient_core.logging import get_logger
from .models import (
    GitHubConfig,
    GitHubRepo,
    GitHubFile,
    BranchInfo,
    PullRequestInfo,
    CommitInfo,
    RateLimitInfo,
)
from .rate_limit import RateLimitHandler
from .utils import parse_github_url, normalize_path

logger = get_logger(__name__)


class GitHubClient:
    """Async client for GitHub API operations.
    
    Provides high-level methods for interacting with GitHub repositories,
    including file operations, branch management, and pull request handling.
    
    Example:
        >>> async with GitHubClient(token="ghp_...") as client:
        ...     repo = await client.get_repository("owner", "repo")
        ...     files = await client.list_files("owner", "repo")
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        config: Optional[GitHubConfig] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ):
        """Initialize GitHub client.
        
        Args:
            token: GitHub personal access token
            config: Full configuration object (overrides token)
            rate_limit_handler: Custom rate limit handler
        """
        self.config = config or GitHubConfig(token=token)
        self.rate_limit_handler = rate_limit_handler or RateLimitHandler(
            max_retries=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
        )
        
        # Build headers
        self._headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Omniscient-Architect/1.0",
        }
        if self.config.token:
            self._headers["Authorization"] = f"token {self.config.token}"
        
        # Create HTTP client
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self._headers,
                timeout=self.config.timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                ),
            )
        return self._client
    
    async def __aenter__(self) -> "GitHubClient":
        """Enter async context."""
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.close()
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _build_url(self, endpoint: str) -> str:
        """Build full API URL."""
        return f"{self.config.api_url}/{endpoint.lstrip('/')}"
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> Any:
        """Make authenticated API request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON body
            
        Returns:
            Response JSON
        """
        client = await self._get_client()
        url = self._build_url(endpoint)
        
        async def do_request():
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=json,
            )
            
            # Update rate limit info
            self.rate_limit_handler.update_from_headers(dict(response.headers))
            
            response.raise_for_status()
            return response.json() if response.content else None
        
        return await self.rate_limit_handler.execute_with_retry(
            do_request,
            f"{method} {endpoint}",
        )
    
    async def _get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        """Make GET request."""
        return await self._request("GET", endpoint, params=params)
    
    async def _post(self, endpoint: str, json: dict) -> Any:
        """Make POST request."""
        return await self._request("POST", endpoint, json=json)
    
    async def _patch(self, endpoint: str, json: dict) -> Any:
        """Make PATCH request."""
        return await self._request("PATCH", endpoint, json=json)
    
    # Repository operations
    
    async def get_repository(self, owner: str, repo: str) -> GitHubRepo:
        """Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            GitHubRepo with repository details
        """
        data = await self._get(f"/repos/{owner}/{repo}")
        
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
            html_url=data.get("html_url"),
            is_private=data.get("private", False),
            is_fork=data.get("fork", False),
            open_issues_count=data.get("open_issues_count", 0),
            license_name=data.get("license", {}).get("name") if data.get("license") else None,
        )
    
    async def get_repository_from_url(self, url: str) -> GitHubRepo:
        """Get repository from URL.
        
        Args:
            url: GitHub repository URL
            
        Returns:
            GitHubRepo with repository details
        """
        owner, repo = parse_github_url(url)
        return await self.get_repository(owner, repo)
    
    async def get_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """Get language breakdown for repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Dict mapping language names to byte counts
        """
        return await self._get(f"/repos/{owner}/{repo}/languages")
    
    # File operations
    
    async def list_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        branch: Optional[str] = None,
    ) -> List[GitHubFile]:
        """List files in a directory.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: Directory path (empty for root)
            branch: Branch name (uses default if not specified)
            
        Returns:
            List of GitHubFile objects
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{normalize_path(path)}"
        params = {"ref": branch} if branch else {}
        
        data = await self._get(endpoint, params)
        
        # Handle single file response
        if isinstance(data, dict):
            data = [data]
        
        return [
            GitHubFile(
                path=item["path"],
                name=item["name"],
                type=item["type"],
                size=item.get("size", 0),
                sha=item.get("sha"),
                download_url=item.get("download_url"),
                url=item.get("url"),
            )
            for item in data
        ]
    
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: Optional[str] = None,
    ) -> str:
        """Get file content as string.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            branch: Branch name
            
        Returns:
            File content as string
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{normalize_path(path)}"
        params = {"ref": branch} if branch else {}
        
        data = await self._get(endpoint, params)
        
        if data.get("encoding") == "base64":
            import base64
            return base64.b64decode(data["content"]).decode("utf-8")
        
        return data.get("content", "")
    
    async def download_raw_file(self, download_url: str) -> str:
        """Download file from raw URL.
        
        Args:
            download_url: Direct download URL
            
        Returns:
            File content as string
        """
        client = await self._get_client()
        response = await client.get(download_url)
        response.raise_for_status()
        return response.text
    
    async def list_files_recursive(
        self,
        owner: str,
        repo: str,
        path: str = "",
        branch: Optional[str] = None,
        max_depth: int = 10,
    ) -> List[GitHubFile]:
        """Recursively list all files in a directory.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: Starting path
            branch: Branch name
            max_depth: Maximum recursion depth
            
        Returns:
            List of all files (not directories)
        """
        all_files = []
        
        async def scan_dir(dir_path: str, depth: int):
            if depth > max_depth:
                return
            
            items = await self.list_files(owner, repo, dir_path, branch)
            
            for item in items:
                if item.is_file:
                    all_files.append(item)
                elif item.is_directory:
                    await scan_dir(item.path, depth + 1)
        
        await scan_dir(path, 0)
        return all_files
    
    # Branch operations
    
    async def list_branches(self, owner: str, repo: str) -> List[BranchInfo]:
        """List repository branches.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            List of BranchInfo objects
        """
        data = await self._get(
            f"/repos/{owner}/{repo}/branches",
            {"per_page": self.config.per_page},
        )
        
        return [
            BranchInfo(
                name=branch["name"],
                sha=branch["commit"]["sha"],
                protected=branch.get("protected", False),
            )
            for branch in data
        ]
    
    async def get_branch(self, owner: str, repo: str, branch: str) -> BranchInfo:
        """Get branch information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            
        Returns:
            BranchInfo object
        """
        data = await self._get(f"/repos/{owner}/{repo}/branches/{branch}")
        
        commit = data.get("commit", {}).get("commit", {})
        
        return BranchInfo(
            name=data["name"],
            sha=data["commit"]["sha"],
            protected=data.get("protected", False),
            commit_message=commit.get("message"),
            commit_date=commit.get("author", {}).get("date"),
            commit_author=commit.get("author", {}).get("name"),
        )
    
    # Pull request operations
    
    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
    ) -> List[PullRequestInfo]:
        """List pull requests.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: PR state (open, closed, all)
            
        Returns:
            List of PullRequestInfo objects
        """
        data = await self._get(
            f"/repos/{owner}/{repo}/pulls",
            {"state": state, "per_page": self.config.per_page},
        )
        
        return [
            PullRequestInfo(
                number=pr["number"],
                title=pr["title"],
                body=pr.get("body"),
                state=pr["state"],
                head_branch=pr["head"]["ref"],
                base_branch=pr["base"]["ref"],
                html_url=pr["html_url"],
                created_at=pr["created_at"],
                updated_at=pr.get("updated_at"),
                merged_at=pr.get("merged_at"),
                author=pr["user"]["login"],
                labels=[label["name"] for label in pr.get("labels", [])],
            )
            for pr in data
        ]
    
    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        number: int,
    ) -> PullRequestInfo:
        """Get pull request details.
        
        Args:
            owner: Repository owner
            repo: Repository name
            number: PR number
            
        Returns:
            PullRequestInfo object
        """
        data = await self._get(f"/repos/{owner}/{repo}/pulls/{number}")
        
        return PullRequestInfo(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=data["state"],
            head_branch=data["head"]["ref"],
            base_branch=data["base"]["ref"],
            html_url=data["html_url"],
            created_at=data["created_at"],
            updated_at=data.get("updated_at"),
            merged_at=data.get("merged_at"),
            author=data["user"]["login"],
            labels=[label["name"] for label in data.get("labels", [])],
            reviewers=[r["login"] for r in data.get("requested_reviewers", [])],
            changed_files=data.get("changed_files", 0),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            mergeable=data.get("mergeable"),
        )
    
    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool = False,
    ) -> PullRequestInfo:
        """Create a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: PR title
            body: PR body/description
            head: Head branch
            base: Base branch
            draft: Create as draft PR
            
        Returns:
            Created PullRequestInfo
        """
        data = await self._post(
            f"/repos/{owner}/{repo}/pulls",
            {
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            },
        )
        
        return PullRequestInfo(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=data["state"],
            head_branch=data["head"]["ref"],
            base_branch=data["base"]["ref"],
            html_url=data["html_url"],
            created_at=data["created_at"],
            author=data["user"]["login"],
        )
    
    # Commit operations
    
    async def list_commits(
        self,
        owner: str,
        repo: str,
        branch: Optional[str] = None,
        limit: int = 30,
    ) -> List[CommitInfo]:
        """List repository commits.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name (uses default if not specified)
            limit: Maximum number of commits
            
        Returns:
            List of CommitInfo objects
        """
        params = {"per_page": min(limit, self.config.per_page)}
        if branch:
            params["sha"] = branch
        
        data = await self._get(f"/repos/{owner}/{repo}/commits", params)
        
        return [
            CommitInfo(
                sha=commit["sha"],
                message=commit["commit"]["message"],
                author_name=commit["commit"]["author"]["name"],
                author_email=commit["commit"]["author"]["email"],
                date=commit["commit"]["author"]["date"],
                url=commit.get("html_url"),
            )
            for commit in data[:limit]
        ]
    
    # Rate limit
    
    async def get_rate_limit(self) -> RateLimitInfo:
        """Get current rate limit status.
        
        Returns:
            RateLimitInfo object
        """
        data = await self._get("/rate_limit")
        core = data.get("resources", {}).get("core", {})
        
        return RateLimitInfo(
            limit=core.get("limit", 0),
            remaining=core.get("remaining", 0),
            reset_timestamp=core.get("reset", 0),
            resource="core",
        )
