"""Omniscient API - REST and GraphQL API for code review.

This package provides a production-ready API server for the Omniscient
Architect platform, enabling code review as a service.

Features:
- REST API with FastAPI
- GraphQL support (optional)
- Async analysis processing
- Rate limiting and authentication
"""

from omniscient_api.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    Finding,
    AnalysisSummary,
)
from omniscient_api.config import APIConfig

# Lazy import for app to avoid multipart dependency at import time
def create_app(*args, **kwargs):
    """Create FastAPI application.
    
    This is a lazy loader to avoid requiring python-multipart at import time.
    """
    from omniscient_api.app import create_app as _create_app
    return _create_app(*args, **kwargs)

__all__ = [
    # App factory
    "create_app",
    # Models
    "AnalysisRequest",
    "AnalysisResponse",
    "AnalysisStatus",
    "Finding",
    "AnalysisSummary",
    # Config
    "APIConfig",
]

__version__ = "0.1.0"
