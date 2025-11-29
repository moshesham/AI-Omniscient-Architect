"""Semantic clustering for file grouping."""

from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from omniscient_core import FileAnalysis


class SemanticClusterer:
    """Clusters files based on semantic similarity.
    
    Uses TF-IDF vectorization and KMeans clustering to group
    similar files together for focused analysis.
    
    Requires: pip install omniscient-tools[clustering]
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        min_files_per_cluster: int = 2,
        max_features: int = 1000,
    ):
        """Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters to create
            min_files_per_cluster: Minimum files per cluster
            max_features: Maximum TF-IDF features
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for SemanticClusterer. "
                "Install with: pip install omniscient-tools[clustering]"
            )
        
        self.n_clusters = n_clusters
        self.min_files_per_cluster = min_files_per_cluster
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b",
        )
    
    def cluster_files(
        self, 
        files: List[FileAnalysis]
    ) -> Dict[int, List[FileAnalysis]]:
        """Cluster files based on content similarity.
        
        Args:
            files: List of FileAnalysis objects with content
            
        Returns:
            Dict mapping cluster IDs to lists of files
        """
        # Filter files with content
        files_with_content = [f for f in files if f.content]
        
        if len(files_with_content) < self.n_clusters:
            # Not enough files, put all in one cluster
            return {0: files_with_content}
        
        # Extract content for vectorization
        contents = [f.content for f in files_with_content]
        
        try:
            # Vectorize
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            
            # Cluster
            n_clusters = min(self.n_clusters, len(files_with_content))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group files by cluster
            clusters: Dict[int, List[FileAnalysis]] = {}
            for file, label in zip(files_with_content, labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(file)
            
            return clusters
            
        except Exception:
            # Fallback: single cluster
            return {0: files_with_content}
    
    def get_cluster_summary(
        self, 
        clusters: Dict[int, List[FileAnalysis]]
    ) -> List[Dict[str, Any]]:
        """Get summary information for each cluster.
        
        Args:
            clusters: Dict from cluster_files()
            
        Returns:
            List of cluster summaries
        """
        summaries = []
        
        for cluster_id, files in clusters.items():
            languages = {}
            total_size = 0
            paths = []
            
            for file in files:
                lang = file.language
                languages[lang] = languages.get(lang, 0) + 1
                total_size += file.size
                paths.append(file.path)
            
            # Determine dominant language
            dominant_lang = max(languages.items(), key=lambda x: x[1])[0]
            
            # Find common path prefix
            if paths:
                common_prefix = self._find_common_prefix(paths)
            else:
                common_prefix = ""
            
            summaries.append({
                "cluster_id": cluster_id,
                "file_count": len(files),
                "total_size": total_size,
                "dominant_language": dominant_lang,
                "languages": languages,
                "common_path": common_prefix,
                "file_paths": paths,
            })
        
        return summaries
    
    def _find_common_prefix(self, paths: List[str]) -> str:
        """Find common path prefix for a list of paths.
        
        Args:
            paths: List of file paths
            
        Returns:
            Common path prefix
        """
        if not paths:
            return ""
        
        if len(paths) == 1:
            return str(Path(paths[0]).parent)
        
        # Split all paths into parts
        split_paths = [Path(p).parts for p in paths]
        
        # Find common prefix
        prefix_parts = []
        for parts in zip(*split_paths):
            if len(set(parts)) == 1:
                prefix_parts.append(parts[0])
            else:
                break
        
        return str(Path(*prefix_parts)) if prefix_parts else ""
    
    def select_representative_files(
        self,
        clusters: Dict[int, List[FileAnalysis]],
        per_cluster: int = 3
    ) -> List[FileAnalysis]:
        """Select representative files from each cluster.
        
        Useful for reducing the number of files sent to LLM while
        maintaining coverage across different code areas.
        
        Args:
            clusters: Dict from cluster_files()
            per_cluster: Number of files to select per cluster
            
        Returns:
            List of representative files
        """
        representatives = []
        
        for cluster_id, files in clusters.items():
            # Sort by size (prefer medium-sized files)
            sorted_files = sorted(files, key=lambda f: f.size)
            
            # Select from middle (avoid very small/large files)
            n = len(sorted_files)
            if n <= per_cluster:
                representatives.extend(sorted_files)
            else:
                # Take from middle of the size distribution
                start = max(0, (n - per_cluster) // 2)
                representatives.extend(sorted_files[start:start + per_cluster])
        
        return representatives
