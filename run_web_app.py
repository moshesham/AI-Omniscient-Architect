#!/usr/bin/env python3
"""Run the Omniscient Architect web application."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit web application."""
    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent

        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(script_dir / "web_app.py")]
        subprocess.run(cmd, cwd=script_dir)

    except KeyboardInterrupt:
        print("\n👋 Web application stopped by user")
    except Exception as e:
        print(f"❌ Error running web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()