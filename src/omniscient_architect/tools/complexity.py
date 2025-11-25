"""Complexity analysis utilities (Lizard integration stub)."""

import lizard
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

class ComplexityAnalyzer:
    """Analyze code complexity using Lizard, file size, and churn."""
    def __init__(self, cache=None):
        self.cache = cache

    def analyze_file(self, file_path: str, churn: int = 0) -> Dict[str, Any]:
        if self.cache:
            cached = self.cache.get(file_path)
            if cached:
                return cached
        metrics = lizard.analyze_file(file_path)
        size = os.path.getsize(file_path)
        score = metrics.average_cyclomatic_complexity + size / 1000 + churn / 10
        result = {
            "file": file_path,
            "complexity": metrics.average_cyclomatic_complexity,
            "size": size,
            "churn": churn,
            "score": score,
        }
        if self.cache:
            self.cache.set(file_path, result)
        return result

    def generate_heatmap(self, repo: str) -> Dict[str, float]:
        """Create a complexity heatmap for the repo."""
        # Scan repo and aggregate metrics
        heatmap = {}
        for root, _, files in os.walk(repo):
            for f in files:
                if f.endswith(('.py', '.js', '.ts', '.java', '.go', '.rs')):
                    path = os.path.join(root, f)
                    try:
                        metrics = self.analyze_file(path)
                        heatmap[path] = metrics["score"]
                    except Exception:
                        continue
        return heatmap
