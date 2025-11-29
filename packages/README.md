# Omniscient Architect Packages

This directory contains the modular packages that make up the Omniscient Architect platform.

## Package Overview

| Package | Description | Key Features |
|---------|-------------|--------------|
| `omniscient-core` | Core models, configuration, and base classes | FileAnalysis, RepositoryInfo, BaseAIAgent |
| `omniscient-agents` | AI analysis agents with prompts | ArchitectureAgent, ReliabilityAgent, etc. |
| `omniscient-tools` | Analysis utilities and tools | ComplexityAnalyzer, AnalysisCache, FileScanner |
| `omniscient-github` | GitHub integration | GitHubClient, RepositoryScanner, PRManager |
| `omniscient-api` | REST/GraphQL API server | FastAPI endpoints, async analysis |

## Installation

### Full Platform
```bash
pip install omniscient-architect
```

### Individual Packages
```bash
# Core only (minimal)
pip install omniscient-core

# Core + Agents (analysis without API)
pip install omniscient-agents

# Full analysis tools
pip install omniscient-tools[clustering]

# GitHub integration
pip install omniscient-github

# API server
pip install omniscient-api[graphql]
```

### Development Installation

From the repository root:

```bash
# Install all packages in development mode
pip install -e packages/core
pip install -e packages/agents
pip install -e packages/tools
pip install -e packages/github
pip install -e packages/api

# Or use the convenience script
python scripts/install_dev.py
```

## Package Dependencies

```
omniscient-core (no dependencies)
    ├── omniscient-agents (depends on core)
    ├── omniscient-tools (depends on core)
    ├── omniscient-github (depends on core)
    └── omniscient-api (depends on all above)
```

## Package Structure

Each package follows a standard structure:

```
packages/<package-name>/
├── pyproject.toml      # Package metadata and dependencies
├── README.md           # Package documentation
├── src/
│   └── omniscient_<name>/
│       ├── __init__.py
│       └── ...modules
└── tests/
    └── ...test files
```

## Development

### Running Tests

```bash
# Test all packages
pytest packages/

# Test specific package
pytest packages/core/tests/
pytest packages/agents/tests/
```

### Building Packages

```bash
# Build all packages
python scripts/build_packages.py

# Build specific package
cd packages/core
python -m build
```

### Publishing

```bash
# Publish to PyPI (requires credentials)
python scripts/publish_packages.py
```

## Migration Guide

If you're migrating from the monolithic structure:

### Before (deprecated)
```python
from omniscient_architect import FileAnalysis, ArchitectureAgent, GitHubClient
```

### After (recommended)
```python
from omniscient_core import FileAnalysis
from omniscient_agents import ArchitectureAgent
from omniscient_github import GitHubClient
```

The old imports will continue to work but will emit deprecation warnings.

## Version Compatibility

All packages are versioned together and should be compatible with each other
when using the same minor version:

- `omniscient-core==0.1.x` compatible with `omniscient-agents==0.1.x`
- Cross-minor version compatibility is not guaranteed

## Contributing

When adding new features:

1. Determine which package the feature belongs to
2. Add the feature to the appropriate package
3. Update the package's `__init__.py` to export new public APIs
4. Add tests in the package's `tests/` directory
5. Update the package's README.md

## License

All packages are released under the MIT License.
