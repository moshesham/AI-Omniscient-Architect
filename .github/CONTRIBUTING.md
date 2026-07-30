# Contributing to Omniscient Architect

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/moshesham/AI-Omniscient-Architect.git
cd AI-Omniscient-Architect

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install developer tooling
pip install -r requirements-dev.txt

# Install workspace packages
pip install -e packages/core -e packages/llm -e packages/tools -e packages/github -e packages/agents -e packages/api -e packages/rag

# Pull a model for testing
ollama pull qwen2.5-coder:1.5b
```

## Development Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

### Code Style

We use the following tools for code quality:

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy packages/
```

### Testing

```bash
pytest -q
python -m build
for pkg in core llm tools github agents api rag; do python -m build "packages/$pkg"; done
```

Local validation should mirror CI:

- `pytest -q`
- `python -m build` plus builds for each package under `packages/`
- `twine check dist/* packages/*/dist/*`

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** your changes
5. **Submit** a pull request

### PR Checklist

- [ ] Code follows the project style guide
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear and descriptive
- [ ] Packaging and release changes follow [RELEASE.md](../RELEASE.md)

## Package Structure

```
packages/
├── core/       # Base models and configuration
├── llm/        # LLM provider abstraction
├── agents/     # Analysis agents
├── tools/      # Utility tools
├── github/     # GitHub integration
├── api/        # REST/GraphQL API
└── rag/        # Retrieval and knowledge pipeline
```

### Adding a New LLM Provider

1. Create provider in `packages/llm/src/omniscient_llm/providers/`
2. Implement `BaseLLMProvider` interface
3. Add to `__init__.py` exports
4. Update documentation

### Adding a New Agent

1. Create agent in `packages/agents/src/omniscient_agents/`
2. Extend `LLMAgent` base class
3. Implement `analyze()` method
4. Add to registry

## Questions?

- Open an [issue](https://github.com/moshesham/AI-Omniscient-Architect/issues)
- Check existing discussions
- See [RELEASE.md](../RELEASE.md) for package publishing and rollback guidance

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
