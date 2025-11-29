# omniscient-core

Core models, configuration, and utilities for the Omniscient Architect platform.

## Installation

```bash
pip install omniscient-core
```

## Usage

```python
from omniscient_core import AnalysisConfig, FileAnalysis, RepositoryInfo

# Load configuration
config = AnalysisConfig()

# Create repository info
repo = RepositoryInfo(path="/path/to/repo", branch="main")

# Create file analysis
file = FileAnalysis(path="src/main.py", size=1024, language="Python")
```

## Components

- **Models**: `FileAnalysis`, `RepositoryInfo`, `AnalysisConfig`, `AgentFindings`, `ReviewResult`
- **Config**: Configuration loading from YAML, environment variables, and overrides
- **Logging**: Structured logging with Rich console output
- **Base Classes**: `BaseAIAgent` abstract base class for agent implementations
