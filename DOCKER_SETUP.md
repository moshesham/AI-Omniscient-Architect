# Docker Setup and Deployment Guide

## Overview

This guide covers the improved Docker setup for the Omniscient Architect project, including optimizations, best practices, and deployment strategies.

## What's New

### Code Improvements

1. **Resource Management**
   - Added `ResourceManager` context manager for proper database and LLM connection handling
   - Automatic cleanup of resources (prevents connection leaks)
   - Improved error handling with detailed traceback logging

2. **Better Error Handling**
   - Try-catch blocks with informative error messages
   - Graceful degradation when learning features are unavailable
   - Connection pool management

3. **Performance Optimizations**
   - Connection pooling for database operations
   - Reusable resource managers
   - Async context managers for efficient resource usage

### Docker Improvements

1. **Multi-Stage Dockerfile**
   - Smaller final image size
   - Better layer caching
   - Separate build and runtime stages
   - Non-root user for security

2. **Enhanced docker-compose.yml**
   - Health checks for all services
   - Resource limits and reservations
   - Proper networking with dedicated bridge network
   - Database initialization on first run
   - Better dependency management

3. **Database Initialization**
   - Automated schema creation via init-db.sql
   - All required extensions (vector, pg_trgm)
   - Proper indexes for performance
   - Learning system tables pre-created

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB RAM available
- 10GB free disk space

### First Time Setup

1. **Copy environment file**
   ```powershell
   Copy-Item env.example .env
   ```

2. **Edit .env file** (optional)
   Update passwords and configuration as needed.

3. **Start all services**
   ```powershell
   docker-compose up -d
   ```

4. **Wait for services to be healthy**
   ```powershell
   docker-compose ps
   ```

5. **Pull required models** (if not auto-pulled)
   ```powershell
   docker exec omniscient-architect-ollama ollama pull nomic-embed-text
   docker exec omniscient-architect-ollama ollama pull qwen2.5-coder:1.5b
   ```

6. **Access the application**
   - Web UI: http://localhost:8501
   - Ollama API: http://localhost:11434
   - PostgreSQL: localhost:5432

### Using PowerShell Management Script

```powershell
# Start all services
.\\docker-manage.ps1 -Action start

# Stop all services
.\\docker-manage.ps1 -Action stop

# Restart services
.\\docker-manage.ps1 -Action restart

# View logs
.\\docker-manage.ps1 -Action logs -FollowLogs

# Check status
.\\docker-manage.ps1 -Action status

# Rebuild images
.\\docker-manage.ps1 -Action build

# Clean up everything
.\\docker-manage.ps1 -Action clean
```

## Architecture

### Service Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Omniscient Architect                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │   Streamlit  │   │   Ollama     │   │  PostgreSQL │ │
│  │   Web App    │──▶│   LLM/Embed  │   │  + pgvector │ │
│  │   :8501      │   │   :11434     │   │  :5432      │ │
│  └──────────────┘   └──────────────┘   └─────────────┘ │
│         │                   │                   │        │
│         └───────────────────┴───────────────────┘        │
│                     Network Bridge                       │
│                   (172.28.0.0/16)                        │
└─────────────────────────────────────────────────────────┘

Volumes:
- postgres_data: Database persistence
- ollama_data: Model storage
- knowledge_data: Uploaded documents
```

### Network Configuration

- **Bridge Network**: `omniscient-network` (172.28.0.0/16)
- **Service Discovery**: Automatic DNS resolution
- **Inter-service Communication**: Via container names

### Resource Allocation

| Service    | Memory Limit | Memory Reserved | CPU  |
|------------|--------------|-----------------|------|
| PostgreSQL | 1GB          | 512MB           | Auto |
| Ollama     | 4GB          | 2GB             | Auto |
| App        | 2GB          | 512MB           | Auto |

**Total**: ~7GB RAM minimum recommended

## Database Schema

The database is automatically initialized with:

### Core RAG Tables
- `rag.documents` - Source documents
- `rag.chunks` - Document chunks with embeddings
- `rag.knowledge_questions` - Test questions
- `rag.knowledge_scores` - Quality metrics

### Learning System Tables
- `rag.learned_facts` - Extracted knowledge
- `rag.reasoning_chains` - Reasoning patterns
- `rag.query_refinements` - Query improvements
- `rag.user_feedback` - User interactions

### Indexes
- Vector similarity (IVFFlat)
- Full-text search (GIN)
- Foreign keys and references

## Health Checks

### Automated Health Monitoring

All services include health checks:

```yaml
PostgreSQL:
  - Checks: pg_isready
  - Interval: 10s
  - Timeout: 5s
  - Retries: 5

Ollama:
  - Checks: API availability
  - Interval: 30s
  - Timeout: 10s
  - Retries: 3

App:
  - Checks: Streamlit health endpoint
  - Interval: 30s
  - Timeout: 10s
  - Retries: 3
```

### Manual Health Check

Run the health check script:

```powershell
python scripts\\health-check.py
```

Output includes:
- Service status
- Response times
- Database statistics
- Available models
- Extension status

Export to JSON:
```powershell
python scripts\\health-check.py --json health-report.json
```

## Configuration

### Environment Variables

Key variables in `.env`:

```bash
# Database
POSTGRES_PASSWORD=localdev
DATABASE_URL=postgresql://omniscient:localdev@postgres:5432/omniscient

# LLM
OLLAMA_HOST=http://ollama:11434
EMBEDDING_MODEL=nomic-embed-text

# RAG
CHUNK_SIZE=512
HYBRID_ALPHA=0.5
TOP_K=5

# Learning
MAX_LEARNED_FACTS=5
MIN_FACT_CONFIDENCE=0.3
```

### Volume Persistence

Data is persisted in Docker volumes:

```powershell
# Backup volumes
docker run --rm -v omniscient-architect_postgres_data:/data -v ${PWD}:/backup alpine tar czf /backup/postgres-backup.tar.gz /data

# Restore volumes
docker run --rm -v omniscient-architect_postgres_data:/data -v ${PWD}:/backup alpine tar xzf /backup/postgres-backup.tar.gz -C /
```

## Development vs Production

### Development Setup

Use `docker-compose.dev.yml`:

```powershell
docker-compose -f docker-compose.dev.yml up
```

Features:
- Hot reload enabled
- Source code mounted
- Debug logging
- All ports exposed

### Production Setup

Use default `docker-compose.yml`:

```powershell
docker-compose up -d
```

Features:
- No source mounts (immutable)
- Minimal image
- Security hardened
- Resource limits enforced

## Troubleshooting

### Services won't start

1. Check Docker resources:
   ```powershell
   docker system df
   docker system prune
   ```

2. Check service logs:
   ```powershell
   docker-compose logs postgres
   docker-compose logs ollama
   docker-compose logs app
   ```

### Database connection errors

1. Verify PostgreSQL is healthy:
   ```powershell
   docker-compose ps postgres
   ```

2. Check connection string in `.env`

3. Test direct connection:
   ```powershell
   docker exec -it omniscient-architect-postgres psql -U omniscient -d omniscient
   ```

### Ollama model issues

1. List available models:
   ```powershell
   docker exec omniscient-architect-ollama ollama list
   ```

2. Pull missing models:
   ```powershell
   docker exec omniscient-architect-ollama ollama pull nomic-embed-text
   ```

3. Check Ollama logs:
   ```powershell
   docker-compose logs ollama
   ```

### Application errors

1. View application logs:
   ```powershell
   docker-compose logs app
   ```

2. Restart application:
   ```powershell
   docker-compose restart app
   ```

3. Check health endpoint:
   ```powershell
   curl http://localhost:8501/_stcore/health
   ```

## Maintenance

### Update Images

```powershell
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up -d --build
```

### Clean Up

```powershell
# Remove stopped containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Complete cleanup
docker system prune -a --volumes
```

### Backup Database

```powershell
# Create backup
docker exec omniscient-architect-postgres pg_dump -U omniscient omniscient > backup.sql

# Restore backup
cat backup.sql | docker exec -i omniscient-architect-postgres psql -U omniscient omniscient
```

## Performance Tuning

### PostgreSQL

Edit `.env`:
```bash
POSTGRES_SHARED_BUFFERS=512MB      # 25% of RAM
POSTGRES_EFFECTIVE_CACHE_SIZE=2GB  # 50-75% of RAM
POSTGRES_WORK_MEM=32MB             # RAM / max_connections
```

### Ollama

For GPU support, modify `docker-compose.yml`:
```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Application

Increase memory limits in `docker-compose.yml`:
```yaml
app:
  deploy:
    resources:
      limits:
        memory: 4G
```

## Security Best Practices

1. **Change default passwords** in `.env`
2. **Use secrets** for production (Docker Swarm/Kubernetes)
3. **Enable SSL/TLS** for external access
4. **Restrict network access** with firewall rules
5. **Regular updates** of base images
6. **Scan images** for vulnerabilities

```powershell
# Scan for vulnerabilities
docker scan omniscient-architect-app:latest
```

## Monitoring and Observability

### Log Collection

```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app

# With timestamps
docker-compose logs -f --timestamps

# Tail last 100 lines
docker-compose logs --tail=100
```

### Metrics (Future)

Integration points for:
- Prometheus for metrics
- Grafana for visualization
- Jaeger for tracing
- ELK stack for logs

## Next Steps

1. **Set up monitoring** - Add Prometheus/Grafana
2. **Configure backups** - Automated daily backups
3. **Add CI/CD** - GitHub Actions for deployment
4. **Implement caching** - Redis for session/query caching
5. **Load balancing** - Multiple app instances

## Support

- GitHub Issues: https://github.com/moshesham/AI-Omniscient-Architect/issues
- Documentation: See README.md and LEARNING_SYSTEM.md
- Health Check: `python scripts/health-check.py`
