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

import subprocess
import sys
from pathlib import Path


def main():
    """Run the analysis on the Omniscient Architect repo."""
    print("🚀 Starting analysis of the Omniscient Architect repository...")
    print("This may take a few minutes depending on your hardware.")

    # Run the CLI as a subprocess to avoid event loop conflicts
    cmd = [
        sys.executable,
        "omniscient_architect_ai.py",
        ".",
        "--objective",
        (
            "Analyze this AI-powered code review and analysis tool for software "
            "architecture, efficiency, reliability, and alignment with its goals"
        ),
        "--output",
        "examples/analysis_report.md",
        "--depth",
        "standard",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            print("✅ Analysis completed successfully!")
            print("📄 Report saved to: examples/analysis_report.md")
            print("\nYou can now:")
            print("- Open the report in your editor")
            print("- View it in the web interface at http://localhost:8501")
            print("- Modify the objective or repo URL for different analyses")
            return 0
        else:
            print(f"❌ Analysis failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return 1
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
