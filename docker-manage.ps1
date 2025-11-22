# Omniscient Architect Docker Management Script
# This script helps manage the Docker deployment of the Omniscient Architect application

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "stop", "restart", "build", "logs", "status", "clean")]
    [string]$Action = "status",

    [Parameter(Mandatory=$false)]
    [switch]$FollowLogs
)

$ErrorActionPreference = "Stop"

function Write-Header {
    Write-Host "🏗️ Omniscient Architect - Docker Management" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
}

function Start-Services {
    Write-Host "🚀 Starting Omniscient Architect services..." -ForegroundColor Green
    docker-compose up -d

    Write-Host "⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10

    Write-Host "✅ Services started! Access the application at: http://localhost:8501" -ForegroundColor Green
    Write-Host "📊 Ollama API available at: http://localhost:11434" -ForegroundColor Green
}

function Stop-Services {
    Write-Host "🛑 Stopping Omniscient Architect services..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "✅ Services stopped" -ForegroundColor Green
}

function Restart-Services {
    Write-Host "🔄 Restarting Omniscient Architect services..." -ForegroundColor Yellow
    docker-compose restart
    Write-Host "✅ Services restarted" -ForegroundColor Green
}

function Build-Services {
    Write-Host "🔨 Building Omniscient Architect services..." -ForegroundColor Yellow
    docker-compose build --no-cache
    Write-Host "✅ Services built" -ForegroundColor Green
}

function Show-Logs {
    if ($FollowLogs) {
        Write-Host "📋 Showing live logs (Ctrl+C to stop)..." -ForegroundColor Yellow
        docker-compose logs -f
    } else {
        Write-Host "📋 Showing recent logs..." -ForegroundColor Yellow
        docker-compose logs
    }
}

function Show-Status {
    Write-Host "📊 Service Status:" -ForegroundColor Yellow
    docker-compose ps

    Write-Host "`n🔍 Container Health:" -ForegroundColor Yellow
    $containers = docker-compose ps --format "table {{.Name}}\t{{.Status}}"
    Write-Host $containers
}

function Clean-Services {
    Write-Host "🧹 Cleaning up Omniscient Architect services..." -ForegroundColor Red
    Write-Host "This will remove all containers, volumes, and networks!" -ForegroundColor Red
    $confirmation = Read-Host "Are you sure? (y/N)"
    if ($confirmation -eq "y" -or $confirmation -eq "Y") {
        docker-compose down -v --remove-orphans
        Write-Host "✅ Cleanup completed" -ForegroundColor Green
    } else {
        Write-Host "❌ Cleanup cancelled" -ForegroundColor Yellow
    }
}

# Main execution
Write-Header

try {
    switch ($Action) {
        "start" { Start-Services }
        "stop" { Stop-Services }
        "restart" { Restart-Services }
        "build" { Build-Services }
        "logs" { Show-Logs }
        "status" { Show-Status }
        "clean" { Clean-Services }
        default {
            Write-Host "Usage: .\docker-manage.ps1 -Action <action>" -ForegroundColor Yellow
            Write-Host "Actions: start, stop, restart, build, logs, status, clean" -ForegroundColor Yellow
            Write-Host "Example: .\docker-manage.ps1 -Action start" -ForegroundColor Yellow
            Write-Host "Example: .\docker-manage.ps1 -Action logs -FollowLogs" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}