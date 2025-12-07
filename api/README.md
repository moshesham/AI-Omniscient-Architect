# API Deployment

This directory contains the Vercel serverless deployment configuration for the Omniscient Architect API.

## Structure

- `index.py` - Entry point for Vercel serverless functions
- `requirements.txt` - Minimal dependencies for serverless deployment

## Deployment

The API is automatically deployed to Vercel when you push to the repository.

### Environment Variables

Set these in your Vercel project settings:

- `OMNISCIENT_DEBUG` - Set to "false" for production
- `GITHUB_TOKEN` - (Optional) GitHub API token for repository analysis
- `OLLAMA_HOST` - (Optional) Ollama API endpoint if using external LLM

## Local Testing

```bash
# Install dependencies
pip install -r api/requirements.txt
pip install -e packages/core
pip install -e packages/llm
pip install -e packages/agents
pip install -e packages/tools
pip install -e packages/github
pip install -e packages/api

# Run the API
uvicorn api.index:app --reload --port 8000
```

## Limitations

The serverless deployment has the following limitations:

- No RAG/vector database support (requires persistent storage)
- No local Ollama LLM (use external API providers)
- Cold start latency on first request

For full functionality with RAG and local LLM, deploy on a server or use Docker.

## API Documentation

Once deployed, visit:
- `/docs` - Swagger UI documentation
- `/redoc` - ReDoc documentation
- `/health` - Health check endpoint
