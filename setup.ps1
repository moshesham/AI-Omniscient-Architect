# Quick Setup Script for Omniscient Architect
# This script automates the initial setup process

param(
    [Parameter(Mandatory=$false)]
    [switch]$SkipModelDownload,
    
    [Parameter(Mandatory=$false)]
    [switch]$DevMode
)

$ErrorActionPreference = "Stop"

Write-Host "🏗️  Omniscient Architect - Quick Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "✓ Checking prerequisites..." -ForegroundColor Yellow

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Host "  ✓ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker-compose --version
    Write-Host "  ✓ Docker Compose: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker Compose not found." -ForegroundColor Red
    exit 1
}

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "✓ Creating .env file from template..." -ForegroundColor Yellow
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-Host "  ✓ .env file created. Please review and update if needed." -ForegroundColor Green
    } else {
        Write-Host "  ✗ env.example not found. Please create .env manually." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  ✓ .env file already exists" -ForegroundColor Green
}

# Determine which compose file to use
$composeFile = if ($DevMode) { "docker-compose.dev.yml" } else { "docker-compose.yml" }
$composeMode = if ($DevMode) { "Development" } else { "Production" }

Write-Host ""
Write-Host "✓ Starting services in $composeMode mode..." -ForegroundColor Yellow
Write-Host "  Using: $composeFile" -ForegroundColor Gray

# Start services
docker-compose -f $composeFile up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to start services" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Waiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Check service health
$services = @("postgres", "ollama", "app")
$allHealthy = $true

foreach ($service in $services) {
    $health = docker inspect --format='{{.State.Health.Status}}' "omniscient-architect-$service" 2>$null
    
    if ($health -eq "healthy" -or $health -eq "") {
        Write-Host "  ✓ $service is running" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $service status: $health" -ForegroundColor Yellow
        $allHealthy = $false
    }
}

# Download models if not skipped
if (-not $SkipModelDownload) {
    Write-Host ""
    Write-Host "✓ Checking and downloading AI models..." -ForegroundColor Yellow
    
    # Check if embedding model exists
    $embedExists = docker exec omniscient-architect-ollama ollama list 2>$null | Select-String "nomic-embed-text"
    
    if (-not $embedExists) {
        Write-Host "  → Downloading embedding model (nomic-embed-text)..." -ForegroundColor Gray
        docker exec omniscient-architect-ollama ollama pull nomic-embed-text
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Embedding model downloaded" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Failed to download embedding model" -ForegroundColor Red
        }
    } else {
        Write-Host "  ✓ Embedding model already available" -ForegroundColor Green
    }
    
    # Check if default LLM model exists
    $llmExists = docker exec omniscient-architect-ollama ollama list 2>$null | Select-String "qwen2.5-coder:1.5b"
    
    if (-not $llmExists) {
        Write-Host "  → Downloading LLM model (qwen2.5-coder:1.5b)..." -ForegroundColor Gray
        Write-Host "    This may take several minutes..." -ForegroundColor Gray
        docker exec omniscient-architect-ollama ollama pull qwen2.5-coder:1.5b
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ LLM model downloaded" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ Failed to download LLM model (can download later)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ✓ LLM model already available" -ForegroundColor Green
    }
}

# Run health check if available
Write-Host ""
Write-Host "✓ Running health check..." -ForegroundColor Yellow

if (Test-Path "scripts\health-check.py") {
    python scripts\health-check.py 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ All services healthy!" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Some services may not be fully ready yet" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ℹ Health check script not found, skipping..." -ForegroundColor Gray
}

# Final instructions
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Access Points:" -ForegroundColor Cyan
Write-Host "  • Web UI:        http://localhost:8501" -ForegroundColor White
Write-Host "  • Ollama API:    http://localhost:11434" -ForegroundColor White
Write-Host "  • PostgreSQL:    localhost:5432" -ForegroundColor White
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Open http://localhost:8501 in your browser" -ForegroundColor White
Write-Host "  2. Go to 'Knowledge Base' tab" -ForegroundColor White
Write-Host "  3. Upload documents to start learning" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Management Commands:" -ForegroundColor Cyan
Write-Host "  • View logs:     docker-compose -f $composeFile logs -f" -ForegroundColor White
Write-Host "  • Stop services: docker-compose -f $composeFile down" -ForegroundColor White
Write-Host "  • Restart:       docker-compose -f $composeFile restart" -ForegroundColor White
Write-Host "  • Health check:  python scripts\health-check.py" -ForegroundColor White
Write-Host ""
Write-Host "📖 Documentation:" -ForegroundColor Cyan
Write-Host "  • Setup Guide:     DOCKER_SETUP.md" -ForegroundColor White
Write-Host "  • Learning System: LEARNING_SYSTEM.md" -ForegroundColor White
Write-Host "  • Quick Start:     LEARNING_QUICKSTART.md" -ForegroundColor White
Write-Host ""

if (-not $allHealthy) {
    Write-Host "⚠️  Note: Some services may still be initializing." -ForegroundColor Yellow
    Write-Host "   Wait 1-2 minutes and check logs if issues persist." -ForegroundColor Yellow
    Write-Host ""
}
