# omniscient-tools

Analysis tools for the Omniscient Architect platform.

## Installation

```bash
pip install omniscient-tools

# With clustering support
pip install omniscient-tools[clustering]

# With parsing support
pip install omniscient-tools[parsing]
```

## Available Tools

### ComplexityAnalyzer
Analyzes code complexity using Lizard metrics.

```python
from omniscient_tools import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()
metrics = analyzer.analyze_file("src/main.py")
print(f"Complexity: {metrics['complexity']}")
print(f"Score: {metrics['score']}")

# Generate heatmap for entire repo
heatmap = analyzer.generate_heatmap("/path/to/repo")
```

### FileScanner
Scans repositories for files matching patterns.

```python
from omniscient_tools import FileScanner

scanner = FileScanner(
    include_patterns=["*.py", "*.js"],
    exclude_patterns=["**/test_*"]
)
files = scanner.scan("/path/to/repo")
```

### AnalysisCache
File-hash based caching for analysis results.

```python
from omniscient_tools import AnalysisCache

cache = AnalysisCache(cache_dir=".cache", ttl=3600)
cache.set("file.py", {"complexity": 10})
result = cache.get("file.py")
```

### SemanticClusterer (requires [clustering])
Clusters files based on semantic similarity.

```python
from omniscient_tools import SemanticClusterer

clusterer = SemanticClusterer(n_clusters=5)
clusters = clusterer.cluster_files(files)
```
