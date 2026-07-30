# Omniscient Architect Packages

This directory contains the modular packages that make up the Omniscient Architect platform.

## Package Overview

| Package | Description | Key Features |
|---------|-------------|--------------|
| `omniscient-core` | Core models, configuration, and base classes | FileAnalysis, RepositoryInfo, BaseAIAgent |
| `omniscient-llm` | Provider abstraction layer | Ollama, OpenAI, Anthropic integrations |
| `omniscient-agents` | AI analysis agents with prompts | ArchitectureAgent, ReliabilityAgent, etc. |
| `omniscient-tools` | Analysis utilities and tools | ComplexityAnalyzer, AnalysisCache, FileScanner |
| `omniscient-github` | GitHub integration | GitHubClient, RepositoryScanner, PRManager |
| `omniscient-api` | REST/GraphQL API server | FastAPI endpoints, async analysis |
| `omniscient-rag` | Retrieval pipeline | Vector store, hybrid search, learning features |

## Installation

### Published Meta-Package
```bash
pip install omniscient-architect
```

The root package is a meta-package that installs the published sub-packages from PyPI. It does not ship its own runtime source tree.

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
# Install app/runtime dependency bootstrap
pip install -r requirements-dev.txt

# Install workspace packages in dependency order
pip install -e packages/core
pip install -e packages/llm
pip install -e packages/tools
pip install -e packages/github
pip install -e packages/agents
pip install -e packages/api
pip install -e packages/rag
```

## Package Dependencies

```
omniscient-core
    ├── omniscient-llm
    ├── omniscient-tools
    ├── omniscient-github
    ├── omniscient-agents (depends on core, optional llm extras)
    ├── omniscient-rag (depends on core + llm)
    └── omniscient-api (depends on core + agents + tools + github)
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
# Build the meta-package
python -m build

# Build all sub-packages
for pkg in core llm tools github agents api rag; do
  python -m build "packages/$pkg"
done
```

### Publishing

```bash
# Dry-run or tag-based publishing is handled by .github/workflows/publish.yml
# See RELEASE.md for the supported tags and release order.
```

## Migration Guide

If you're migrating from the monolithic structure:

### After (recommended)
```python
from omniscient_core import FileAnalysis
from omniscient_agents import ArchitectureAgent
from omniscient_github import GitHubClient
```

## Version Compatibility

All packages are released on the same minor version line and should be used together within the same minor version:

- `omniscient-core==0.2.x` compatible with `omniscient-agents==0.2.x`
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
