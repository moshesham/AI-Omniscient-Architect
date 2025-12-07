"""Complexity analysis using Lizard."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from omniscient_core import optional_import

HAS_LIZARD, _ = optional_import("lizard")
if HAS_LIZARD:
    import lizard


class ComplexityAnalyzer:
    """Analyze code complexity using Lizard metrics.
    
    Provides cyclomatic complexity, lines of code, and other metrics.
    Optionally integrates with cache for faster repeated analysis.
    
    Attributes:
        cache: Optional AnalysisCache for caching results
        supported_extensions: File extensions that can be analyzed
    """
    
    SUPPORTED_EXTENSIONS = {
        ".py", ".java", ".js", ".ts", ".c", ".cpp", ".h", ".hpp",
        ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt"
    }
    
    def __init__(self, cache: Optional[Any] = None):
        """Initialize the analyzer.
        
        Args:
            cache: Optional AnalysisCache instance for caching results
        """
        if not HAS_LIZARD:
            raise ImportError("lizard is required for ComplexityAnalyzer")
        self.cache = cache
    
    def analyze_file(
        self, 
        file_path: str, 
        churn: int = 0
    ) -> Dict[str, Any]:
        """Analyze complexity of a single file.
        
        Args:
            file_path: Path to the file to analyze
            churn: Git churn score (number of changes) for weighting
            
        Returns:
            Dict with complexity metrics and calculated score
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(file_path)
            if cached:
                return cached
        
        path = Path(file_path)
        if not path.exists():
            return self._empty_result(file_path)
        
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            return self._empty_result(file_path)
        
        try:
            analysis = lizard.analyze_file(str(path))
            size = os.path.getsize(file_path)
            
            # Calculate weighted score
            # Higher = more complex/risky
            complexity = analysis.average_cyclomatic_complexity
            nloc = analysis.nloc
            token_count = analysis.token_count
            
            # Score formula: complexity + size factor + churn factor
            score = complexity + (size / 1000) + (churn / 10)
            
            result = {
                "file": str(file_path),
                "complexity": complexity,
                "nloc": nloc,
                "token_count": token_count,
                "size": size,
                "churn": churn,
                "score": round(score, 2),
                "function_count": len(analysis.function_list),
                "functions": [
                    {
                        "name": f.name,
                        "complexity": f.cyclomatic_complexity,
                        "nloc": f.nloc,
                        "start_line": f.start_line,
                        "end_line": f.end_line,
                    }
                    for f in analysis.function_list
                ]
            }
            
            # Cache the result
            if self.cache:
                self.cache.set(file_path, result)
            
            return result
            
        except Exception as e:
            return {
                "file": str(file_path),
                "error": str(e),
                "complexity": 0,
                "nloc": 0,
                "size": 0,
                "churn": churn,
                "score": 0,
            }
    
    def _empty_result(self, file_path: str) -> Dict[str, Any]:
        """Return empty result for unsupported/missing files."""
        return {
            "file": str(file_path),
            "complexity": 0,
            "nloc": 0,
            "token_count": 0,
            "size": 0,
            "churn": 0,
            "score": 0,
            "function_count": 0,
            "functions": [],
        }
    
    def generate_heatmap(
        self, 
        repo_path: str,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Generate a complexity heatmap for a repository.
        
        Args:
            repo_path: Path to the repository root
            extensions: Optional list of extensions to include
                       (defaults to SUPPORTED_EXTENSIONS)
                       
        Returns:
            Dict mapping file paths to complexity scores
        """
        heatmap: Dict[str, float] = {}
        
        if extensions is None:
            extensions = list(self.SUPPORTED_EXTENSIONS)
        else:
            extensions = [e if e.startswith(".") else f".{e}" for e in extensions]
        
        repo = Path(repo_path)
        if not repo.exists():
            return heatmap
        
        for path in repo.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in extensions:
                continue
            
            # Skip common non-source directories
            parts = path.parts
            if any(skip in parts for skip in [
                ".git", "__pycache__", "node_modules", 
                ".venv", "venv", "dist", "build"
            ]):
                continue
            
            try:
                metrics = self.analyze_file(str(path))
                heatmap[str(path)] = metrics["score"]
            except Exception:
                continue
        
        return heatmap
    
    def get_top_complex_files(
        self, 
        repo_path: str, 
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the N most complex files in a repository.
        
        Args:
            repo_path: Path to the repository
            n: Number of files to return
            
        Returns:
            List of file analysis results, sorted by score descending
        """
        heatmap = self.generate_heatmap(repo_path)
        
        # Sort by score and get top N
        sorted_files = sorted(
            heatmap.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        # Get full analysis for top files
        results = []
        for file_path, _ in sorted_files:
            result = self.analyze_file(file_path)
            results.append(result)
        
        return results
