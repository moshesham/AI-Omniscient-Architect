"""Vercel serverless function entry point for FastAPI."""

import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum

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

# Initialize variables
# We use _app to avoid Vercel's native detection which can be flaky
_app = None
handler = None
initialization_error = None

try:
    # Import and create the FastAPI app
    from omniscient_api.app import create_app
    
    # Create the app instance
    _app = create_app()
    
except Exception as e:
    # Store error for debugging
    initialization_error = str(e)
    import traceback
    error_traceback = traceback.format_exc()
    
    # Create fallback minimal app
    _app = FastAPI(
        title="Omniscient Architect API",
        description="AI-powered code architecture analysis (Initialization Error)",
        version="0.2.0"
    )
    
    @_app.get("/")
    async def read_root():
        return {
            "name": "Omniscient Architect API",
            "status": "degraded",
            "error": "API initialization failed",
            "details": initialization_error,
            "traceback": error_traceback.split("\n")
        }
    
    @_app.get("/health")
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

# Create the handler for Vercel (using Mangum adapter)
handler = Mangum(_app)

