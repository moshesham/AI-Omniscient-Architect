"""FastAPI application factory."""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omniscient_core.logging import get_logger
from .config import APIConfig, load_api_config
from .routes import router as api_router
from .routes.rag import router as rag_router
from .models import HealthResponse
from .database import init_db

logger = get_logger(__name__)

# Track startup time
_startup_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _startup_time
    _startup_time = time.time()
    
    logger.info("Starting Omniscient API server")
    
    # Initialize database
    await init_db()
    
    # Initialize RAG Pipeline
    try:
        from omniscient_rag.pipeline import RAGPipeline
        from omniscient_llm.providers.ollama import OllamaProvider
        
        config: APIConfig = app.state.config
        
        # Initialize LLM provider
        llm_provider = OllamaProvider(
            base_url=config.get_llm_base_url(),
            model=config.llm.model,
        )
        
        # Initialize RAG
        # Use memory:// if no DATABASE_URL provided or if explicitly requested
        import os
        db_url = os.getenv("DATABASE_URL", "memory://")
        
        pipeline = RAGPipeline.create(
            db_url=db_url,
            embed_fn=llm_provider,
            llm_fn=llm_provider.generate,
        )
        await pipeline.initialize()
        app.state.rag_pipeline = pipeline
        logger.info(f"RAG Pipeline initialized with {db_url}")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        app.state.rag_pipeline = None
    
    yield
    
    # Cleanup
    if app.state.rag_pipeline:
        await app.state.rag_pipeline.close()
    
    logger.info("Shutting down Omniscient API server")


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        config: API configuration (loads from environment if not provided)
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = load_api_config()
    
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Store config in app state
    app.state.config = config
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allow_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
    )
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=config.version,
            uptime_seconds=time.time() - _startup_time if _startup_time else 0,
            checks={
                "api": True,
                # TODO: Add more checks (LLM, GitHub, etc.)
            }
        )
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": config.title,
            "version": config.version,
            "docs": "/docs",
            "health": "/health",
            "api": config.api_prefix,
        }
    
    # Include API routes
    app.include_router(api_router, prefix=config.api_prefix)
    app.include_router(rag_router, prefix=config.api_prefix)
    
    logger.info(f"API configured at {config.host}:{config.port}{config.api_prefix}")
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    config_path: Optional[str] = None,
):
    """Run the API server.
    
    Args:
        host: Server host
        port: Server port
        workers: Number of worker processes
        reload: Enable auto-reload for development
        config_path: Path to config file
    """
    import uvicorn
    
    config = load_api_config(config_path)
    config.host = host
    config.port = port
    config.workers = workers
    
    uvicorn.run(
        "omniscient_api.app:create_app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=reload,
        factory=True,
    )
