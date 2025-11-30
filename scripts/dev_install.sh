#!/usr/bin/env bash
#
# Install all Omniscient Architect packages in editable (development) mode.
#
# Usage:
#   ./scripts/dev_install.sh           # Install all packages
#   ./scripts/dev_install.sh --minimal # Install only core+llm
#   ./scripts/dev_install.sh --dev     # Include dev dependencies
#

set -e

# Parse arguments
MINIMAL=false
WITH_DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal|-m)
            MINIMAL=true
            shift
            ;;
        --dev|-d)
            WITH_DEV=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--minimal] [--dev]"
            exit 1
            ;;
    esac
done

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGES_DIR="$REPO_ROOT/packages"

echo "======================================"
echo "Omniscient Architect - Dev Install"
echo "======================================"
echo ""

# Define packages to install
if [ "$MINIMAL" = true ]; then
    PACKAGES=("core" "llm")
    echo "Installing MINIMAL packages: core, llm"
else
    PACKAGES=("core" "llm" "tools" "agents" "github" "api" "rag")
    echo "Installing ALL packages"
fi

echo ""

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected!"
    echo "Consider activating a venv first: source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Install each package
for pkg in "${PACKAGES[@]}"; do
    pkg_path="$PACKAGES_DIR/$pkg"
    
    if [ -d "$pkg_path" ]; then
        echo "Installing $pkg..."
        
        extras=""
        if [ "$WITH_DEV" = true ]; then
            extras="[dev]"
        fi
        
        # Special handling for packages with extra dependencies
        case $pkg in
            llm)
                extras="[ollama]"
                if [ "$WITH_DEV" = true ]; then extras="[ollama,dev]"; fi
                ;;
            rag)
                extras="[ast]"
                if [ "$WITH_DEV" = true ]; then extras="[ast,dev]"; fi
                ;;
        esac
        
        cmd="pip install -e \"$pkg_path$extras\""
        echo "  > $cmd"
        
        eval "$cmd"
        
        echo "  ✓ $pkg installed"
    else
        echo "  ⚠ Skipping $pkg (not found at $pkg_path)"
    fi
done

echo ""
echo "======================================"
echo "✓ Development install complete!"
echo "======================================"
echo ""
echo "Installed packages:"
pip list | grep -i omniscient || true
echo ""
echo "You can now import packages:"
echo "  from omniscient_core import FileAnalysis"
echo "  from omniscient_llm import OllamaProvider"
echo "  from omniscient_rag import RAGPipeline"
