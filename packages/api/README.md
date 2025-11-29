# Omniscient API

REST and GraphQL API for the Omniscient Architect platform. Provides code review as a service.

## Installation

```bash
pip install omniscient-api

# With GraphQL support
pip install omniscient-api[graphql]
```

## Quick Start

### Run the API Server

```bash
# Using CLI
omniscient-api serve --host 0.0.0.0 --port 8000

# Or programmatically
python -c "from omniscient_api import create_app; import uvicorn; uvicorn.run(create_app())"
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Analyze Repository
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "repository_url": "https://github.com/owner/repo",
    "agents": ["architecture", "reliability"],
    "config": {
      "max_files": 100,
      "include_patterns": ["*.py"]
    }
  }'
```

#### Analyze Local Files
```bash
curl -X POST http://localhost:8000/api/v1/analyze/files \
  -H "Content-Type: multipart/form-data" \
  -F "files=@src/main.py" \
  -F "files=@src/utils.py"
```

#### Get Analysis Status
```bash
curl http://localhost:8000/api/v1/analysis/{analysis_id}
```

#### List Available Agents
```bash
curl http://localhost:8000/api/v1/agents
```

## Configuration

### Environment Variables

- `OMNISCIENT_API_HOST`: API host (default: "0.0.0.0")
- `OMNISCIENT_API_PORT`: API port (default: 8000)
- `OMNISCIENT_API_WORKERS`: Number of workers (default: 1)
- `GITHUB_TOKEN`: GitHub personal access token
- `OLLAMA_HOST`: Ollama server URL (default: "http://localhost:11434")

### Config File

```yaml
# config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins:
    - "http://localhost:3000"
    - "https://your-frontend.com"

analysis:
  max_file_size: 100000
  default_agents:
    - architecture
    - reliability
    - efficiency

llm:
  provider: ollama
  model: llama3.2:latest
```

## Python Client

```python
from omniscient_api.client import OmniscientClient

client = OmniscientClient(base_url="http://localhost:8000")

# Analyze a repository
result = await client.analyze_repository(
    "https://github.com/owner/repo",
    agents=["architecture", "reliability"],
)

print(f"Analysis ID: {result.analysis_id}")
print(f"Status: {result.status}")

# Wait for completion
result = await client.wait_for_analysis(result.analysis_id)

for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
```

## GraphQL (Optional)

With the `graphql` extra installed:

```bash
pip install omniscient-api[graphql]
```

Access GraphQL playground at: `http://localhost:8000/graphql`

### Example Query

```graphql
query {
  analyzeRepository(url: "https://github.com/owner/repo") {
    analysisId
    status
    findings {
      agentName
      severity
      title
      description
    }
  }
}
```

## Docker

```bash
# Build
docker build -t omniscient-api .

# Run
docker run -p 8000:8000 \
  -e GITHUB_TOKEN=ghp_... \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  omniscient-api
```

## API Response Models

### AnalysisResponse
```json
{
  "analysis_id": "uuid",
  "status": "pending|running|completed|failed",
  "repository_url": "https://github.com/owner/repo",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z",
  "findings": [...],
  "summary": {...},
  "metrics": {...}
}
```

### Finding
```json
{
  "agent_name": "architecture",
  "severity": "high|medium|low|info",
  "category": "design_pattern",
  "title": "Circular Dependency Detected",
  "description": "...",
  "file_path": "src/module.py",
  "line_start": 10,
  "line_end": 25,
  "suggestions": [...]
}
```

## License

MIT License
