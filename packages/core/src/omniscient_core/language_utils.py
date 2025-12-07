"""Language detection utilities."""

from pathlib import Path
from typing import Dict


# Comprehensive language extensions mapping
LANGUAGE_EXTENSIONS: Dict[str, str] = {
    # Python
    "py": "Python", "pyw": "Python", "pyi": "Python",
    
    # JavaScript/TypeScript
    "js": "JavaScript", "jsx": "JavaScript", "mjs": "JavaScript", "cjs": "JavaScript",
    "ts": "TypeScript", "tsx": "TypeScript",
    
    # JVM Languages
    "java": "Java", "kt": "Kotlin", "kts": "Kotlin", "scala": "Scala",
    
    # Systems Languages
    "c": "C", "h": "C", "cpp": "C++", "cc": "C++", "cxx": "C++",
    "hpp": "C++", "hxx": "C++", "go": "Go", "rs": "Rust",
    
    # Other Languages
    "rb": "Ruby", "php": "PHP", "r": "R", "cs": "C#", "swift": "Swift",
    "sh": "Shell", "bash": "Bash", "zsh": "Zsh", "ps1": "PowerShell",
    
    # Data/Config
    "json": "JSON", "yaml": "YAML", "yml": "YAML", "toml": "TOML",
    "xml": "XML", "sql": "SQL",
    
    # Web
    "html": "HTML", "htm": "HTML", "css": "CSS", "scss": "SCSS",
    "sass": "SASS", "less": "Less",
    
    # Documentation
    "md": "Markdown", "rst": "reStructuredText", "txt": "Text",
}


def detect_language(file_path: str | Path) -> str:
    """Detect programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Language name or "Unknown"
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    ext = file_path.suffix.lstrip(".").lower()
    return LANGUAGE_EXTENSIONS.get(ext, "Unknown")


def get_language_for_ast(file_path: str | Path) -> str:
    """Get normalized language name for AST parsing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Normalized language name (lowercase)
    """
    lang = detect_language(file_path)
    return lang.lower() if lang != "Unknown" else "unknown"
