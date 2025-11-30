# AI-Omniscient-Architect: Platform Development Plan

**Version**: 1.0  
**Date**: November 28, 2025  
**Status**: Active Development  

---

## Executive Summary

Transform AI-Omniscient-Architect into a **production-grade, local-first AI code analysis platform** that empowers developers to gain deep insights into their GitHub codebases. The platform will feature a modular architecture enabling:

- **Standalone Python packages** for agents, tools, and utilities
- **REST/GraphQL API** for programmatic access and integrations
- **Streamlit + FastAPI hybrid UI/API** for interactive and headless use
- **Plugin architecture** for custom agents and tools
- **Enterprise-ready** security, observability, and scalability

---

## 🏗️ Current Architecture Analysis

### Existing Structure
```
omniscient_architect/
├── agents/           # AI analysis agents (architecture, efficiency, reliability, alignment)
├── core/             # Core analysis engine
├── prompts/          # Prompt templates and loaders
├── tools/            # Complexity, clustering, file scanner
├── utils/            # Cache, ingestion, logging, IO
├── analysis.py       # Main AnalysisEngine
├── cli.py            # Rich CLI interface
├── web_app.py        # Streamlit web interface
├── github_client.py  # GitHub API integration
├── models.py         # Pydantic data models
├── config.py         # Configuration management
└── reporting.py      # Report generation
```

### Strengths
- Modular agent architecture with base class pattern
- Separation of prompts, tools, and utilities
- Multiple interfaces (CLI, Web, API-ready)
- GitHub integration for remote repos
- Local LLM support via Ollama

### Gaps to Address
- Tight coupling between components
- No formal API layer
- Limited plugin/extension system
- Missing observability and telemetry
- No package distribution strategy

---

## 🎯 Vision & Goals

### Primary Goals
1. **Local-First**: Run entirely on developer hardware with optional cloud features
2. **Modular Packages**: Each component publishable as standalone PyPI package
3. **API-First**: Full REST/GraphQL API for all functionality
4. **GitHub-Native**: Deep GitHub integration (repos, PRs, issues, actions)
5. **Production Quality**: Enterprise-grade reliability, security, observability

### Success Metrics
- 95%+ test coverage on core packages
- <3s cold start for typical analysis
- <100ms API response time for cached queries
- Zero critical security vulnerabilities
- 100% offline capability for core features

---

## 📦 Modular Package Architecture

### Package Hierarchy

```
omniscient-architect (meta-package)
├── omniscient-core          # Core models, config, base classes
├── omniscient-agents        # AI analysis agents
├── omniscient-tools         # Analysis tools (complexity, clustering)
├── omniscient-github        # GitHub integration
├── omniscient-api           # FastAPI REST/GraphQL server
├── omniscient-cli           # Command-line interface
├── omniscient-web           # Streamlit web UI
└── omniscient-plugins       # Plugin SDK and registry
```

### Package Details

#### 1. `omniscient-core` (Foundation)
```python
# Responsibilities:
# - Pydantic models (FileAnalysis, RepositoryInfo, AgentResponse, etc.)
# - Configuration management (AnalysisConfig, load_config)
# - Base agent class (BaseAIAgent)
# - Logging infrastructure
# - Cache abstractions

# Dependencies: pydantic, structlog, blake3
# Size: Minimal, no heavy ML dependencies
```

#### 2. `omniscient-agents` (AI Analysis)
```python
# Responsibilities:
# - ArchitectureAgent, EfficiencyAgent, ReliabilityAgent, AlignmentAgent
# - Agent registry and discovery
# - Prompt templates and loaders
# - LLM client abstraction (Ollama, OpenAI, Anthropic)

# Dependencies: omniscient-core, langchain-core, langchain-ollama
# Extension: Plugin interface for custom agents
```

#### 3. `omniscient-tools` (Analysis Utilities)
```python
# Responsibilities:
# - ComplexityAnalyzer (Lizard integration)
# - SemanticClusterer (KMeans, HDBSCAN)
# - FileScanner (pattern matching, filtering)
# - CodeParser (tree-sitter integration)

# Dependencies: omniscient-core, lizard, scikit-learn, tree-sitter
```

#### 4. `omniscient-github` (GitHub Integration)
```python
# Responsibilities:
# - GitHubClient (REST API wrapper)
# - Repository cloning and caching
# - PR/Issue analysis
# - GitHub Actions integration
# - Webhook handlers

# Dependencies: omniscient-core, PyGithub, httpx
```

#### 5. `omniscient-api` (REST/GraphQL Server)
```python
# Responsibilities:
# - FastAPI application
# - REST endpoints for analysis, repos, agents
# - GraphQL schema and resolvers
# - Authentication (API keys, OAuth)
# - Rate limiting and quotas
# - WebSocket for streaming results

# Dependencies: omniscient-core, omniscient-agents, fastapi, strawberry-graphql
```

#### 6. `omniscient-cli` (Command Line)
```python
# Responsibilities:
# - Rich terminal UI
# - Interactive and batch modes
# - Config file management
# - Output formatting (JSON, Markdown, HTML)

# Dependencies: omniscient-core, omniscient-agents, rich, click
```

#### 7. `omniscient-web` (Web Interface)
```python
# Responsibilities:
# - Streamlit application
# - Real-time analysis dashboard
# - Repository browser
# - Agent configuration UI
# - Report visualization

# Dependencies: omniscient-core, omniscient-api, streamlit, plotly
```

#### 8. `omniscient-plugins` (Extension SDK)
```python
# Responsibilities:
# - Plugin discovery and loading
# - Custom agent SDK
# - Custom tool SDK
# - Hook system for lifecycle events
# - Plugin marketplace integration

# Dependencies: omniscient-core, pluggy
```

---

## 🔌 API Architecture

### REST API Endpoints

```yaml
# Analysis
POST   /api/v1/analyze                    # Start new analysis
GET    /api/v1/analyze/{id}               # Get analysis status/results
DELETE /api/v1/analyze/{id}               # Cancel analysis
GET    /api/v1/analyze/{id}/stream        # WebSocket for live updates

# Repositories
POST   /api/v1/repos                      # Add repository
GET    /api/v1/repos                      # List repositories
GET    /api/v1/repos/{id}                 # Get repository details
DELETE /api/v1/repos/{id}                 # Remove repository
POST   /api/v1/repos/{id}/sync            # Sync from GitHub

# Agents
GET    /api/v1/agents                     # List available agents
GET    /api/v1/agents/{name}              # Get agent details
POST   /api/v1/agents/{name}/analyze      # Run specific agent

# Configuration
GET    /api/v1/config                     # Get current config
PATCH  /api/v1/config                     # Update config
GET    /api/v1/config/presets             # List presets

# Health & Metrics
GET    /api/v1/health                     # Health check
GET    /api/v1/metrics                    # Prometheus metrics
GET    /api/v1/version                    # Version info
```

### GraphQL Schema

```graphql
type Query {
  repository(id: ID!): Repository
  repositories(filter: RepoFilter): [Repository!]!
  analysis(id: ID!): Analysis
  agents: [Agent!]!
  config: Config!
}

type Mutation {
  analyzeRepository(input: AnalyzeInput!): Analysis!
  addRepository(input: AddRepoInput!): Repository!
  updateConfig(input: ConfigInput!): Config!
}

type Subscription {
  analysisProgress(id: ID!): AnalysisProgress!
}

type Repository {
  id: ID!
  url: String!
  name: String!
  branch: String!
  files: [FileAnalysis!]!
  analyses: [Analysis!]!
}

type Analysis {
  id: ID!
  status: AnalysisStatus!
  repository: Repository!
  agents: [AgentResult!]!
  startedAt: DateTime!
  completedAt: DateTime
}

type AgentResult {
  agent: Agent!
  findings: [String!]!
  confidence: Float!
  recommendations: [String!]!
}
```

---

## 🔧 Implementation Phases

### Phase 1: Package Extraction (Weeks 1-2)
**Goal**: Extract core packages without breaking existing functionality

1. **Create monorepo structure**
   ```
   packages/
   ├── core/
   ├── agents/
   ├── tools/
   ├── github/
   └── api/
   ```

2. **Extract `omniscient-core`**
   - Move models.py → packages/core/src/omniscient_core/models.py
   - Move config.py → packages/core/src/omniscient_core/config.py
   - Move base agent → packages/core/src/omniscient_core/base.py
   - Create package pyproject.toml with minimal dependencies

3. **Extract `omniscient-agents`**
   - Move agents/ → packages/agents/src/omniscient_agents/
   - Move prompts/ → packages/agents/src/omniscient_agents/prompts/
   - Depend on omniscient-core

4. **Extract `omniscient-tools`**
   - Move tools/ → packages/tools/src/omniscient_tools/
   - Move utils/cache.py → packages/tools/src/omniscient_tools/cache.py

5. **Update imports across codebase**
   - Create compatibility layer in main package
   - Ensure CLI and web app still work

### Phase 2: API Layer (Weeks 3-4)
**Goal**: Build production-ready REST/GraphQL API

1. **FastAPI application structure**
   ```python
   # packages/api/src/omniscient_api/
   ├── app.py              # FastAPI app factory
   ├── routes/
   │   ├── analyze.py      # Analysis endpoints
   │   ├── repos.py        # Repository endpoints
   │   ├── agents.py       # Agent endpoints
   │   └── health.py       # Health/metrics
   ├── graphql/
   │   ├── schema.py       # Strawberry schema
   │   └── resolvers.py    # Query/mutation resolvers
   ├── auth/
   │   ├── api_key.py      # API key auth
   │   └── oauth.py        # GitHub OAuth
   ├── middleware/
   │   ├── rate_limit.py   # Rate limiting
   │   └── logging.py      # Request logging
   └── websocket/
       └── stream.py       # Analysis streaming
   ```

2. **Authentication system**
   - API key generation and validation
   - GitHub OAuth for web UI
   - Role-based access control

3. **Rate limiting and quotas**
   - Per-user request limits
   - Analysis job quotas
   - Configurable tiers

4. **WebSocket streaming**
   - Real-time analysis progress
   - Agent output streaming
   - Error notifications

### Phase 3: Plugin System (Weeks 5-6)
**Goal**: Enable extensibility via plugins

1. **Plugin SDK**
   ```python
   from omniscient_plugins import AgentPlugin, ToolPlugin, hookimpl
   
   class SecurityAgent(AgentPlugin):
       name = "security"
       version = "1.0.0"
       
       @hookimpl
       def analyze(self, files, repo_info):
           # Custom security analysis
           pass
   ```

2. **Plugin discovery**
   - Entry point based discovery
   - Local plugin directories
   - Plugin marketplace (future)

3. **Hook system**
   ```python
   # Available hooks:
   # - pre_analysis, post_analysis
   # - pre_agent, post_agent
   # - on_file_scan, on_error
   # - on_report_generate
   ```

4. **Built-in plugin examples**
   - SecurityAgent (SAST integration)
   - DocumentationAgent (doc coverage)
   - DependencyAgent (vulnerability scanning)

### Phase 4: Observability (Weeks 7-8)
**Goal**: Production-grade monitoring and debugging

1. **Structured logging**
   ```python
   import structlog
   
   logger = structlog.get_logger()
   logger.info("analysis.started", 
       repo=repo_info.url,
       agents=selected_agents,
       config=config.dict()
   )
   ```

2. **Metrics collection**
   - Analysis duration histograms
   - Agent success/failure rates
   - LLM token usage
   - Cache hit rates

3. **Tracing**
   - OpenTelemetry integration
   - Distributed trace IDs
   - Span annotations for agents

4. **Health checks**
   - LLM connectivity
   - GitHub API status
   - Cache health
   - Disk space

### Phase 5: Performance & Scale (Weeks 9-10)
**Goal**: Optimize for large repositories

1. **Intelligent file selection**
   ```python
   class SmartFileSelector:
       def select(self, repo, budget=100):
           # Score by: complexity, churn, centrality
           # Use heatmap + clustering
           # Stay within token budget
   ```

2. **Parallel processing**
   - Async file scanning
   - Concurrent agent execution
   - Batched LLM calls

3. **Advanced caching**
   - File hash-based cache keys
   - LRU eviction policy
   - Cache warming strategies

4. **Resource management**
   - Memory limits per analysis
   - CPU throttling
   - Graceful degradation

### Phase 6: GitHub Deep Integration (Weeks 11-12)
**Goal**: Native GitHub workflow integration

1. **PR analysis**
   ```python
   # Analyze PR changes only
   await engine.analyze_pr(
       repo="owner/repo",
       pr_number=123,
       agents=["architecture", "reliability"]
   )
   ```

2. **GitHub Actions**
   ```yaml
   # .github/workflows/omniscient.yml
   - uses: omniscient-architect/analyze@v1
     with:
       agents: architecture,security
       fail-on: critical
   ```

3. **Issue integration**
   - Auto-create issues from findings
   - Link findings to existing issues
   - Track issue resolution

4. **Webhook handlers**
   - Auto-analyze on push
   - PR comment bot
   - Scheduled analysis

---

## 🔐 Security Architecture

### Authentication
- **API Keys**: HMAC-SHA256 signed tokens
- **OAuth**: GitHub OAuth 2.0 for web UI
- **JWT**: Short-lived tokens for API sessions

### Authorization
- **RBAC**: Owner, Admin, Analyst, Viewer roles
- **Resource-based**: Per-repository permissions
- **Rate limits**: Tiered by role

### Data Security
- **Encryption at rest**: AES-256 for cache files
- **Secrets masking**: Auto-detect and redact secrets in logs
- **Audit logging**: All API actions logged

### Dependency Security
- **Dependabot**: Automated dependency updates
- **SBOM generation**: Track all dependencies
- **Vulnerability scanning**: Regular scans

---

## 📊 Data Models (Extended)

```python
# packages/core/src/omniscient_core/models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentConfig(BaseModel):
    """Configuration for an individual agent."""
    name: str
    enabled: bool = True
    priority: int = 0
    timeout_seconds: int = 300
    custom_prompt: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class AnalysisJob(BaseModel):
    """Represents an analysis job."""
    id: str
    repository_id: str
    status: AnalysisStatus
    agents: List[str]
    config: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    """Complete analysis result."""
    job: AnalysisJob
    repository: RepositoryInfo
    file_analyses: List[FileAnalysis]
    agent_results: Dict[str, AgentResponse]
    summary: str
    score: float  # 0-100 overall health score
    critical_findings: List[str]
    recommendations: List[str]

class Webhook(BaseModel):
    """Webhook configuration."""
    id: str
    url: str
    events: List[str]  # ["analysis.completed", "finding.critical"]
    secret: str
    active: bool = True
```

---

## 🧪 Testing Strategy

### Test Pyramid
```
         /\
        /  \  E2E Tests (10%)
       /----\  - Full workflow tests
      /      \  - API integration tests
     /--------\
    /          \ Integration Tests (30%)
   /   ------   \ - Agent + LLM tests
  /              \ - GitHub API tests
 /----------------\
/                  \ Unit Tests (60%)
 - Model validation - Config parsing
 - Cache operations - File scanning
```

### Test Categories
```python
# pytest markers
@pytest.mark.unit          # Fast, no I/O
@pytest.mark.integration   # External services mocked
@pytest.mark.e2e          # Full system tests
@pytest.mark.slow         # >1s duration
@pytest.mark.llm          # Requires LLM
```

### Coverage Targets
- Core package: 95%+
- Agents package: 90%+
- Tools package: 90%+
- API package: 85%+

---

## 📈 Roadmap Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Package Extraction | Weeks 1-2 | omniscient-core, omniscient-agents, omniscient-tools |
| 2. API Layer | Weeks 3-4 | REST API, GraphQL, auth, WebSocket |
| 3. Plugin System | Weeks 5-6 | Plugin SDK, hooks, example plugins |
| 4. Observability | Weeks 7-8 | Metrics, tracing, dashboards |
| 5. Performance | Weeks 9-10 | Caching, parallelism, resource management |
| 6. GitHub Integration | Weeks 11-12 | PR analysis, Actions, webhooks |

---

## 🚀 Getting Started (Post-Implementation)

### Installation
```bash
# Full platform
pip install omniscient-architect

# Core only (for embedding)
pip install omniscient-core omniscient-agents

# With all extras
pip install omniscient-architect[all]
```

### Quick Start
```python
from omniscient_core import AnalysisConfig
from omniscient_agents import ArchitectureAgent, ReliabilityAgent
from omniscient_github import GitHubClient

# Configure
config = AnalysisConfig(ollama_model="codellama:7b")

# Analyze
client = GitHubClient(token="...")
repo = await client.get_repository("owner/repo")
results = await analyze(repo, agents=[ArchitectureAgent, ReliabilityAgent])

# Report
print(results.summary)
```

### API Access
```bash
# Start server
omniscient-api serve --port 8000

# Analyze via API
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"repo": "owner/repo", "agents": ["architecture"]}'
```

---

## 📝 Next Immediate Actions

1. **Create monorepo structure** with packages/ directory
2. **Extract omniscient-core** as first standalone package
3. **Set up CI/CD** for multi-package builds
4. **Implement FastAPI skeleton** with health endpoint
5. **Add OpenAPI documentation** generation
6. **Create plugin SDK** prototype

---

*This plan is designed for iterative delivery. Each phase produces shippable value while building toward the complete platform vision.*
