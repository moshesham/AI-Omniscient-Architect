<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/LLM-Ollama-orange.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/UI-Streamlit-red.svg" alt="Streamlit">
</p>

<h1 align="center">🏗️ Omniscient Architect</h1>

<p align="center">
  <strong>AI-Powered Code Analysis Platform with Local LLM Support</strong>
</p>

<p align="center">
  Analyze codebases using local AI models for privacy-first, intelligent code review.<br/>
  No data leaves your machine. No API costs. Full control.
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **Privacy-First** | All analysis runs locally via Ollama - your code never leaves your machine |
| 🤖 **Multi-Provider LLM** | Support for Ollama, OpenAI, and Anthropic with automatic fallback |
| 📊 **Smart Analysis** | Security vulnerabilities, architecture patterns, code quality, best practices |
| 🌐 **Web UI** | Beautiful Streamlit interface for interactive analysis |
| 📦 **Modular Architecture** | Six independent packages for flexibility and extensibility |
| ⚡ **Parallel Execution** | Concurrent agent analysis with progress streaming |
| 🐙 **GitHub Integration** | Analyze repositories directly from GitHub URLs |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running

### Installation

```bash
# Clone the repository
git clone https://github.com/moshesham/AI-Omniscient-Architect.git
cd AI-Omniscient-Architect

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull a code-focused model
ollama pull qwen2.5-coder:1.5b
```

### Launch the Web UI

```bash
streamlit run web_app.py
```

Open http://localhost:8501 in your browser.

---

## 📦 Package Architecture

```
packages/
├── omniscient-core     # Base models, configuration, logging
├── omniscient-llm      # Multi-provider LLM abstraction layer
├── omniscient-agents   # AI analysis agents with orchestration
├── omniscient-tools    # Code complexity, clustering, file scanning
├── omniscient-github   # GitHub API client with rate limiting
└── omniscient-api      # FastAPI REST/GraphQL server
```

### Package Overview

| Package | Purpose | Key Components |
|---------|---------|----------------|
| `omniscient-core` | Foundation | `FileAnalysis`, `RepositoryInfo`, `AnalysisConfig` |
| `omniscient-llm` | LLM Integration | `OllamaProvider`, `OpenAIProvider`, `ProviderChain` |
| `omniscient-agents` | Analysis | `CodeReviewAgent`, `AnalysisOrchestrator` |
| `omniscient-tools` | Utilities | `ComplexityAnalyzer`, `FileScanner`, `Clustering` |
| `omniscient-github` | GitHub | `GitHubClient`, `RateLimitHandler` |
| `omniscient-api` | API Server | FastAPI routes, GraphQL schema |

---

## 🖥️ Usage

### Web Interface (Recommended)

The Streamlit UI provides the easiest way to analyze code:

1. **Check Ollama Status** - Verify your LLM is running
2. **Select Model** - Choose from available Ollama models
3. **Choose Focus Areas** - Security, Architecture, Code Quality, etc.
4. **Analyze** - Point to a local directory or GitHub URL

### Programmatic Usage

```python
import asyncio
from omniscient_llm import OllamaProvider, LLMClient
from omniscient_agents.llm_agent import CodeReviewAgent
from omniscient_core import FileAnalysis, RepositoryInfo

async def analyze_code():
    # Setup LLM
    provider = OllamaProvider(model="qwen2.5-coder:1.5b")
    client = LLMClient(provider=provider)
    
    async with client:
        # Create agent
        agent = CodeReviewAgent(
            llm_client=client,
            focus_areas=["security", "architecture"]
        )
        
        # Prepare files
        files = [
            FileAnalysis(
                path="main.py",
                content="your code here",
                language="Python",
                size=100
            )
        ]
        
        repo = RepositoryInfo(
            path="./my-project",
            name="my-project",
            branch="main"
        )
        
        # Run analysis
        result = await agent.analyze(files, repo)
        print(result.summary)

asyncio.run(analyze_code())
```

### LLM CLI Tool

```bash
# Check Ollama status
python -m omniscient_llm status

# List available models
python -m omniscient_llm list

# Pull a new model
python -m omniscient_llm pull codellama:7b-instruct

# Get model recommendations
python -m omniscient_llm recommend --category code
```

### API Server

Start the REST API server:

```bash
# Ensure packages are in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:packages/api/src:packages/core/src:packages/rag/src:packages/llm/src

# Run the server
python -m omniscient_api.cli serve
```

The API will be available at `http://localhost:8000`.
Documentation is available at `http://localhost:8000/docs`.

#### Authentication

The API is protected by an API Key. Set the `OMNISCIENT_API_KEY` environment variable to enable authentication.

```bash
export OMNISCIENT_API_KEY="your-secret-key"
```

Pass the key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/api/v1/analyze ...
```

---

## 🐳 Docker Deployment

```bash
# Development (with hot reload)
docker compose -f docker-compose.dev.yml up --build

# Production
docker compose up --build -d

# Check health
curl http://localhost:8501/_stcore/health
```

### Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model | `qwen2.5-coder:1.5b` |
| `MAX_FILES` | Max files to analyze | `100` |
| `ANALYSIS_DEPTH` | `quick`, `standard`, `deep` | `standard` |

---

## 🔍 Analysis Capabilities

### What Gets Analyzed

| Category | Checks |
|----------|--------|
| **Security** | SQL injection, XSS, hardcoded secrets, CORS misconfig |
| **Architecture** | Design patterns, separation of concerns, scalability |
| **Code Quality** | Complexity, duplication, naming conventions |
| **Best Practices** | Error handling, logging, documentation |
| **Performance** | Bottlenecks, caching opportunities, async patterns |

### Sample Output

```
📋 Summary:
The codebase has several security concerns that need immediate attention.

📊 Issues Found: 3

🔴 [HIGH] Security
   Hardcoded database credentials found
   📁 File: api/routes/data.py
   📍 Line: 20
   💡 Use environment variables or a secrets manager

🟡 [MEDIUM] Architecture  
   Global state can cause race conditions
   📁 File: api/routes/data.py
   💡 Use dependency injection or request-scoped state

🟢 [LOW] Code Quality
   Missing docstrings in public functions
   💡 Add docstrings for better maintainability

💡 Recommendations:
  • Move credentials to environment variables
  • Implement proper dependency injection
  • Add comprehensive documentation
```

---

## 🛠️ Development

### Project Structure

```
AI-Omniscient-Architect/
├── web_app.py           # Streamlit UI
├── packages/            # Modular packages
├── scripts/             # Test & utility scripts
├── docs/                # Documentation
├── roadmap/             # Development roadmap
├── examples/            # Usage examples
├── Dockerfile           # Container definition
├── docker-compose.yml   # Production compose
└── requirements.txt     # Dependencies
```

### Running Tests

```bash
# Test all packages
python scripts/test_packages.py

# Test local analysis
python scripts/test_local_analysis.py

# Test with a specific repo
python scripts/test_datalake_analysis.py
```

### Recommended Models

| Model | Size | Best For | Memory |
|-------|------|----------|--------|
| `qwen2.5-coder:1.5b` | 1GB | Quick analysis, limited RAM | 2GB |
| `codellama:7b-instruct` | 4GB | Detailed analysis | 8GB |
| `deepseek-coder:6.7b` | 4GB | Complex code understanding | 8GB |

---

## 📖 Documentation

- [Development Roadmap](roadmap/PHASE_2_3_PROGRESS.md) - Current progress and future plans
- [Package Documentation](packages/README.md) - Detailed package docs
- [API Reference](packages/api/README.md) - REST/GraphQL API docs

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- 🌐 Additional LLM providers
- 📊 More analysis agents (testing, documentation)
- 🔌 IDE extensions (VS Code, JetBrains)
- 📈 Metrics and reporting dashboards
- 🔄 CI/CD integration templates

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ for developers who value privacy and code quality</strong>
</p>

<p align="center">
  <a href="https://github.com/moshesham/AI-Omniscient-Architect/issues">Report Bug</a>
  ·
  <a href="https://github.com/moshesham/AI-Omniscient-Architect/issues">Request Feature</a>
</p>
