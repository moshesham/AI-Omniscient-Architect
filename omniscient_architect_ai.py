#!/usr/bin/env python3
"""
Omniscient Architect - AI-powered Code Review System

This is the main entry point for the AI-driven code review system.
For backward compatibility, this script provides the same CLI interface
as the original version, but now powered by AI agents.
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

if __name__ == '__main__':
    # Import after adjusting sys.path to avoid E402 in lint
    from omniscient_architect.cli import sync_main
    sync_main()
