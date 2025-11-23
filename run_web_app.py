#!/usr/bin/env python3
"""Run the Omniscient Architect web application."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit web application."""
    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent

        # Add src to path for imports
        sys.path.insert(0, str(script_dir / "src"))

        # Run streamlit
        env = os.environ.copy()
        env['PYTHONPATH'] = str(script_dir / "src")
        cmd = [sys.executable, "-m", "streamlit", "run", str(script_dir / "src" / "omniscient_architect" / "web_app.py")]
        subprocess.run(cmd, cwd=script_dir, env=env)

    except KeyboardInterrupt:
        print("\n👋 Web application stopped by user")
    except Exception as e:
        print(f"❌ Error running web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()