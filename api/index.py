"""Vercel serverless function entry point for FastAPI."""

import sys
import os
from pathlib import Path

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "llm", "agents", "tools", "github", "api"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): sys.path.insert(0, str(_path))

# Set minimal config for serverless environment
os.environ.setdefault("OMNISCIENT_DEBUG", "false")
os.environ.setdefault("OMNISCIENT_API_HOST", "0.0.0.0")
os.environ.setdefault("OMNISCIENT_API_PORT", "8000")

try:
    # Import and create the FastAPI app
    from omniscient_api.app import create_app
    
    # Create the app instance for Vercel
    app = create_app()
    
    # This is the handler that Vercel will call
    handler = app
    
except Exception as e:
    # Fallback minimal app if packages fail to load
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/")
    def read_root():
        return {
            "error": "API initialization failed",
            "detail": str(e),
            "message": "Please check package installation"
        }
    
    handler = app

