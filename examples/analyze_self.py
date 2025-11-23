#!/usr/bin/env python3
"""
Example: Analyze the Omniscient Architect repository itself

This script demonstrates how to use the Omniscient Architect tool
to analyze its own codebase. It performs a comprehensive AI-powered
code review of the repository that contains the tool.

Requirements:
- Python 3.11+
- Ollama running with codellama:7b-instruct model
- All dependencies installed

Usage:
    python examples/analyze_self.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omniscient_architect.cli import sync_main

async def main():
    """Run the analysis on the Omniscient Architect repo."""
    print("🚀 Starting analysis of the Omniscient Architect repository...")
    print("This may take a few minutes depending on your hardware.")

    # Simulate CLI arguments for analyzing this repo
    # In a real scenario, you might want to analyze a different repo
    sys.argv = [
        "omniscient_architect_ai.py",
        "https://github.com/moshesham/AI-Omniscient-Architect",
        "--objective",
        "Analyze this AI-powered code review and analysis tool for software architecture, efficiency, reliability, and alignment with its goals",
        "--output",
        "examples/analysis_report.md",
        "--depth",
        "standard"
    ]

    # Run the CLI
    try:
        sync_main()
        print("✅ Analysis completed successfully!")
        print("📄 Report saved to: examples/analysis_report.md")
        print("\nYou can now:")
        print("- Open the report in your editor")
        print("- View it in the web interface at http://localhost:8501")
        print("- Modify the objective or repo URL for different analyses")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)