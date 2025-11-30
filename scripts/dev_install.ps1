#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Install all Omniscient Architect packages in editable (development) mode.

.DESCRIPTION
    This script installs all sub-packages from the packages/ directory in editable mode,
    allowing you to develop and test changes without reinstalling.

.PARAMETER Minimal
    Install only core and llm packages (minimal setup for testing).

.PARAMETER WithDev
    Include development dependencies (pytest, black, etc.).

.EXAMPLE
    .\scripts\dev_install.ps1
    # Installs all packages in editable mode

.EXAMPLE
    .\scripts\dev_install.ps1 -Minimal -WithDev
    # Installs core+llm packages with dev dependencies
#>

param(
    [switch]$Minimal,
    [switch]$WithDev
)

$ErrorActionPreference = "Stop"

# Get the repository root (parent of scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PackagesDir = Join-Path $RepoRoot "packages"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Omniscient Architect - Dev Install" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Define package install order (dependencies first)
if ($Minimal) {
    $Packages = @("core", "llm")
    Write-Host "Installing MINIMAL packages: core, llm" -ForegroundColor Yellow
} else {
    $Packages = @("core", "llm", "tools", "agents", "github", "api", "rag")
    Write-Host "Installing ALL packages" -ForegroundColor Green
}

Write-Host ""

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "WARNING: No virtual environment detected!" -ForegroundColor Yellow
    Write-Host "Consider activating a venv first: .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Aborted." -ForegroundColor Red
        exit 1
    }
}

# Install each package in editable mode
foreach ($pkg in $Packages) {
    $pkgPath = Join-Path $PackagesDir $pkg
    
    if (Test-Path $pkgPath) {
        Write-Host "Installing $pkg..." -ForegroundColor Cyan
        
        $extras = ""
        if ($WithDev) {
            $extras = "[dev]"
        }
        
        # Special handling for packages with extra dependencies
        if ($pkg -eq "llm") {
            $extras = "[ollama]"
            if ($WithDev) { $extras = "[ollama,dev]" }
        }
        elseif ($pkg -eq "rag") {
            $extras = "[ast]"
            if ($WithDev) { $extras = "[ast,dev]" }
        }
        
        $installCmd = "pip install -e `"$pkgPath$extras`""
        Write-Host "  > $installCmd" -ForegroundColor DarkGray
        
        Invoke-Expression $installCmd
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install $pkg" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "  ✓ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Skipping $pkg (not found at $pkgPath)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "✓ Development install complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor White
pip list | Select-String "omniscient"
Write-Host ""
Write-Host "You can now import packages:" -ForegroundColor White
Write-Host "  from omniscient_core import FileAnalysis" -ForegroundColor DarkGray
Write-Host "  from omniscient_llm import OllamaProvider" -ForegroundColor DarkGray
Write-Host "  from omniscient_rag import RAGPipeline" -ForegroundColor DarkGray
