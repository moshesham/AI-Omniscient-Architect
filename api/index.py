"""Vercel serverless function entry point for FastAPI."""

import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Set minimal config for serverless environment
os.environ.setdefault("OMNISCIENT_DEBUG", "false")
os.environ.setdefault("OMNISCIENT_API_HOST", "0.0.0.0")
os.environ.setdefault("OMNISCIENT_API_PORT", "8000")

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "llm", "agents", "tools", "github", "api"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): 
        sys.path.insert(0, str(_path))

# Initialize app variable
app = None
initialization_error = None

try:
    # Import and create the FastAPI app
    from omniscient_api.app import create_app
    
    # Create the app instance for Vercel
    app = create_app()
    
except Exception as e:
    # Store error for debugging
    initialization_error = str(e)
    import traceback
    error_traceback = traceback.format_exc()
    
    # Create fallback minimal app if packages fail to load
    app = FastAPI(
        title="Omniscient Architect API",
        description="AI-powered code architecture analysis (Initialization Error)",
        version="0.2.0"
    )
    
    @app.get("/")
    async def read_root():
        return {
            "name": "Omniscient Architect API",
            "status": "degraded",
            "error": "API initialization failed",
            "detail": initialization_error,
            "traceback": error_traceback,
            "message": "Please check logs and package installation",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health")
    async def health():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": initialization_error,
                "checks": {
                    "api": False,
                    "packages": False
                }
            }
        )

# This is the handler that Vercel will call
handler = app

