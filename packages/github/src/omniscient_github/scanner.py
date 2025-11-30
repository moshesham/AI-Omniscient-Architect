"""Repository scanner for discovering and filtering files."""

import fnmatch
import time
from typing import List, Optional, Set
from pathlib import Path

from omniscient_core.logging import get_logger
from omniscient_core import FileAnalysis
from .client import GitHubClient
from .models import GitHubRepo, ScanResult, GitHubConfig

logger = get_logger(__name__)


# Default patterns
DEFAULT_EXCLUDE_PATTERNS = {
    # Directories
    "**/node_modules/**",
    "**/.git/**",
    "**/venv/**",
    "**/.venv/**",
    "**/env/**",
    "**/__pycache__/**",
    "**/dist/**",
    "**/build/**",
    "**/.next/**",
    "**/coverage/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/vendor/**",
    
    # Files
    "**/*.min.js",
    "**/*.min.css",
    "**/package-lock.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/poetry.lock",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.so",
    "**/*.dll",
    "**/*.exe",
    "**/*.bin",
    
    # Generated
    "**/*.generated.*",
    "**/auto-generated/**",
}

# Language extensions mapping
LANGUAGE_EXTENSIONS = {
    "py": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "jsx": "JavaScript",
    "tsx": "TypeScript",
    "java": "Java",
    "kt": "Kotlin",
    "go": "Go",
    "rs": "Rust",
    "rb": "Ruby",
    "php": "PHP",
    "cs": "C#",
    "cpp": "C++",
    "c": "C",
    "h": "C",
    "hpp": "C++",
    "swift": "Swift",
    "scala": "Scala",
    "r": "R",
    "sql": "SQL",
    "sh": "Shell",
    "bash": "Shell",
    "ps1": "PowerShell",
    "yaml": "YAML",
    "yml": "YAML",
    "json": "JSON",
    "xml": "XML",
    "html": "HTML",
    "css": "CSS",
    "scss": "SCSS",
    "less": "Less",
    "md": "Markdown",
    "rst": "reStructuredText",
}


class RepositoryScanner:
    """Scans GitHub repositories to discover and filter files.
    
    Provides pattern-based filtering, size limits, and language detection
    for efficient repository analysis.
    
    Example:
        >>> scanner = RepositoryScanner(token="ghp_...")
        >>> result = await scanner.scan_repository(
        ...     "owner/repo",
        ...     include_patterns=["*.py"],
        ...     max_file_size=50000,
        ... )
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        config: Optional[GitHubConfig] = None,
        client: Optional[GitHubClient] = None,
    ):
        """Initialize scanner.
        
        Args:
            token: GitHub token
            config: GitHub configuration
            client: Existing GitHubClient instance
        """
        self._token = token
        self._config = config
        self._client = client
        self._owns_client = client is None
    
    async def _get_client(self) -> GitHubClient:
        """Get or create client."""
        if self._client is None:
            self._client = GitHubClient(
                token=self._token,
                config=self._config,
            )
            await self._client.__aenter__()
        return self._client
    
    async def close(self):
        """Close owned client."""
        if self._owns_client and self._client:
            await self._client.close()
            self._client = None
    
    async def __aenter__(self) -> "RepositoryScanner":
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _matches_pattern(self, path: str, patterns: Set[str]) -> bool:
        """Check if path matches any pattern.
        
        Args:
            path: File path
            patterns: Set of glob patterns
            
        Returns:
            True if path matches any pattern
        """
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
            # Also check just the filename
            if fnmatch.fnmatch(Path(path).name, pattern):
                return True
        return False
    
    def _detect_language(self, path: str) -> str:
        """Detect programming language from file extension.
        
        Args:
            path: File path
            
        Returns:
            Language name or "Unknown"
        """
        ext = Path(path).suffix.lstrip(".").lower()
        return LANGUAGE_EXTENSIONS.get(ext, "Unknown")
    
    async def scan_repository(
        self,
        repo_url_or_name: str,
        branch: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size: int = 100_000,  # 100KB
        max_files: int = 500,
        include_content: bool = False,
    ) -> ScanResult:
        """Scan a repository for files.
        
        Args:
            repo_url_or_name: Repository URL or "owner/repo"
            branch: Branch to scan (uses default if not specified)
            include_patterns: Patterns for files to include (e.g., ["*.py"])
            exclude_patterns: Patterns for files to exclude
            max_file_size: Maximum file size in bytes
            max_files: Maximum number of files to return
            include_content: Whether to download file content
            
        Returns:
            ScanResult with discovered files
        """
        start_time = time.time()
        client = await self._get_client()
        errors = []
        
        # Parse repository
        try:
            from .utils import parse_github_url
            owner, repo = parse_github_url(repo_url_or_name)
        except ValueError:
            # Try as owner/repo
            parts = repo_url_or_name.split("/")
            if len(parts) == 2:
                owner, repo = parts
            else:
                raise ValueError(f"Invalid repository: {repo_url_or_name}")
        
        # Get repository info
        logger.info(f"Scanning repository {owner}/{repo}")
        github_repo = await client.get_repository(owner, repo)
        
        if branch is None:
            branch = github_repo.default_branch
        
        # Build pattern sets
        include_set = set(include_patterns) if include_patterns else None
        exclude_set = set(exclude_patterns) if exclude_patterns else DEFAULT_EXCLUDE_PATTERNS
        
        # Scan files recursively
        logger.info(f"Scanning files on branch {branch}")
        all_files = await client.list_files_recursive(owner, repo, branch=branch)
        
        # Filter files
        filtered_files = []
        language_counts = {}
        total_size = 0
        
        for file in all_files:
            # Size filter
            if file.size > max_file_size:
                continue
            
            # Exclude patterns
            if self._matches_pattern(file.path, exclude_set):
                continue
            
            # Include patterns (if specified)
            if include_set and not self._matches_pattern(file.path, include_set):
                continue
            
            # Max files limit
            if len(filtered_files) >= max_files:
                logger.warning(f"Reached max files limit ({max_files})")
                break
            
            # Track language
            lang = self._detect_language(file.path)
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Download content if requested
            if include_content and file.download_url:
                try:
                    content = await client.download_raw_file(file.download_url)
                    file.content = content
                except Exception as e:
                    errors.append(f"Failed to download {file.path}: {e}")
            
            filtered_files.append(file)
            total_size += file.size
        
        duration = time.time() - start_time
        logger.info(
            f"Scan complete: {len(filtered_files)} files, "
            f"{total_size:,} bytes, {duration:.2f}s"
        )
        
        return ScanResult(
            repository=github_repo,
            files=filtered_files,
            total_size=total_size,
            file_count=len(filtered_files),
            language_breakdown=language_counts,
            scan_duration_seconds=duration,
            errors=errors,
        )
    
    async def scan_to_file_analysis(
        self,
        repo_url_or_name: str,
        **kwargs,
    ) -> List[FileAnalysis]:
        """Scan repository and convert to FileAnalysis objects.
        
        This is useful for integrating with the analysis pipeline.
        
        Args:
            repo_url_or_name: Repository URL or "owner/repo"
            **kwargs: Additional arguments for scan_repository
            
        Returns:
            List of FileAnalysis objects
        """
        result = await self.scan_repository(
            repo_url_or_name,
            include_content=True,
            **kwargs,
        )
        
        analyses = []
        for file in result.files:
            analyses.append(FileAnalysis(
                path=file.path,
                content=file.content or "",
                language=self._detect_language(file.path),
                size=file.size,
            ))
        
        return analyses
    
    async def get_file_tree(
        self,
        repo_url_or_name: str,
        branch: Optional[str] = None,
        max_depth: int = 5,
    ) -> dict:
        """Get file tree structure of repository.
        
        Args:
            repo_url_or_name: Repository URL or "owner/repo"
            branch: Branch to scan
            max_depth: Maximum depth to traverse
            
        Returns:
            Nested dict representing file tree
        """
        result = await self.scan_repository(
            repo_url_or_name,
            branch=branch,
            max_files=10000,
            include_content=False,
        )
        
        tree = {}
        
        for file in result.files:
            parts = file.path.split("/")
            current = tree
            
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add file
            current[parts[-1]] = {
                "type": "file",
                "size": file.size,
                "language": self._detect_language(file.path),
            }
        
        return {
            "repository": result.repository.full_name,
            "branch": branch or result.repository.default_branch,
            "tree": tree,
            "file_count": result.file_count,
            "total_size": result.total_size,
        }
