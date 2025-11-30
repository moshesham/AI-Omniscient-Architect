"""File scanner for repository analysis."""

import fnmatch
from pathlib import Path
from typing import List, Optional

from omniscient_core import FileAnalysis


class FileScanner:
    """Scans repositories for files matching specified patterns.
    
    Supports include/exclude patterns, size limits, and extension filtering.
    
    Attributes:
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files/directories to exclude
        exclude_extensions: File extensions to exclude
        max_file_size: Maximum file size in bytes
        max_files: Maximum number of files to return
    """
    
    DEFAULT_INCLUDE = ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs", "*.rb"]
    DEFAULT_EXCLUDE = [".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"]
    DEFAULT_EXCLUDE_EXT = ["png", "jpg", "jpeg", "gif", "svg", "pdf", "zip", "tar", "gz", "exe", "dll"]
    
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        exclude_extensions: Optional[List[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        max_files: int = 1000,
    ):
        """Initialize the scanner.
        
        Args:
            include_patterns: Glob patterns to include (default: common code files)
            exclude_patterns: Patterns to exclude (default: common non-code dirs)
            exclude_extensions: Extensions to exclude (default: binary/media)
            max_file_size: Max file size in bytes
            max_files: Max number of files to return
        """
        self.include_patterns = include_patterns or self.DEFAULT_INCLUDE
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE
        self.exclude_extensions = set(
            ext.lstrip(".") for ext in (exclude_extensions or self.DEFAULT_EXCLUDE_EXT)
        )
        self.max_file_size = max_file_size
        self.max_files = max_files
    
    def scan(
        self, 
        repo_path: str,
        load_content: bool = False,
        content_limit: int = 20 * 1024  # 20KB
    ) -> List[FileAnalysis]:
        """Scan a repository for matching files.
        
        Args:
            repo_path: Path to the repository root
            load_content: Whether to load file contents
            content_limit: Max content size to load per file
            
        Returns:
            List of FileAnalysis objects for matching files
        """
        results: List[FileAnalysis] = []
        repo = Path(repo_path)
        
        if not repo.exists():
            return results
        
        for path in repo.rglob("*"):
            if len(results) >= self.max_files:
                break
            
            if not path.is_file():
                continue
            
            # Check exclusions
            if self._should_exclude(path, repo):
                continue
            
            # Check inclusions
            if not self._should_include(path):
                continue
            
            # Check file size
            try:
                size = path.stat().st_size
                if size > self.max_file_size:
                    continue
            except Exception:
                continue
            
            # Create FileAnalysis
            file_analysis = self._create_file_analysis(
                path, size, load_content, content_limit
            )
            results.append(file_analysis)
        
        return results
    
    def _should_exclude(self, path: Path, repo: Path) -> bool:
        """Check if a path should be excluded.
        
        Args:
            path: File path to check
            repo: Repository root path
            
        Returns:
            True if path should be excluded
        """
        # Check extension
        ext = path.suffix.lstrip(".")
        if ext in self.exclude_extensions:
            return True
        
        # Check path parts against exclude patterns
        rel_path = path.relative_to(repo)
        for part in rel_path.parts:
            for pattern in self.exclude_patterns:
                if fnmatch.fnmatch(part, pattern):
                    return True
        
        return False
    
    def _should_include(self, path: Path) -> bool:
        """Check if a path matches include patterns.
        
        Args:
            path: File path to check
            
        Returns:
            True if path matches any include pattern
        """
        filename = path.name
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False
    
    def _create_file_analysis(
        self, 
        path: Path, 
        size: int,
        load_content: bool,
        content_limit: int
    ) -> FileAnalysis:
        """Create a FileAnalysis object for a file.
        
        Args:
            path: File path
            size: File size in bytes
            load_content: Whether to load content
            content_limit: Max content size
            
        Returns:
            FileAnalysis object
        """
        language = self._detect_language(path)
        content = None
        
        if load_content and size <= content_limit:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass  # Silently skip files that cannot be read
        
        return FileAnalysis(
            path=str(path),
            size=size,
            language=language,
            content=content,
        )
    
    def _detect_language(self, path: Path) -> str:
        """Detect programming language from file extension.
        
        Args:
            path: File path
            
        Returns:
            Language name
        """
        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "JavaScript",
            ".tsx": "TypeScript",
            ".java": "Java",
            ".go": "Go",
            ".rs": "Rust",
            ".rb": "Ruby",
            ".php": "PHP",
            ".c": "C",
            ".cpp": "C++",
            ".h": "C",
            ".hpp": "C++",
            ".cs": "C#",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".md": "Markdown",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".toml": "TOML",
            ".xml": "XML",
            ".html": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".sql": "SQL",
            ".sh": "Shell",
            ".bash": "Bash",
            ".ps1": "PowerShell",
        }
        return ext_map.get(path.suffix.lower(), "Unknown")
    
    def get_language_stats(self, files: List[FileAnalysis]) -> dict:
        """Get statistics by programming language.
        
        Args:
            files: List of FileAnalysis objects
            
        Returns:
            Dict with language statistics
        """
        stats: dict = {}
        for file in files:
            lang = file.language
            if lang not in stats:
                stats[lang] = {"count": 0, "total_size": 0}
            stats[lang]["count"] += 1
            stats[lang]["total_size"] += file.size
        
        return stats
