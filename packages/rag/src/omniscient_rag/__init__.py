"""Omniscient RAG - Retrieval-Augmented Generation infrastructure.

This package provides:
- Pluggable chunking strategies (Fixed, Semantic, AST-based)
- PostgreSQL + pgvector storage
- Hybrid search (vector + BM25)
- Knowledge metrics tracking
- Persistent learning across sessions
- Embedding caching for performance
"""

from .pipeline import RAGPipeline, RAGConfig
from .chunkers import ChunkerFactory
from .models import Document, Chunk, RetrievalResult, KnowledgeScore, ChunkingStrategy

# Embedding cache (always available)
from .embedding_cache import (
    EmbeddingCache,
    CachedEmbedder,
    CacheStats,
    get_embedding_cache,
    cached_embed,
    cached_embed_batch,
)

# Learning module (optional - requires all dependencies)
try:
    from .learning import (
        KnowledgeMemory,
        ContextInjector,
        FeedbackLearner,
        LearnedFact,
        ReasoningChain,
        FeedbackType,
    )
    HAS_LEARNING = True
except ImportError:
    HAS_LEARNING = False

__all__ = [
    # Core
    "RAGPipeline",
    "RAGConfig",
    "ChunkerFactory",
    "ChunkingStrategy",
    "Document",
    "Chunk",
    "RetrievalResult",
    "KnowledgeScore",
    # Embedding cache
    "EmbeddingCache",
    "CachedEmbedder",
    "CacheStats",
    "get_embedding_cache",
    "cached_embed",
    "cached_embed_batch",
    # Learning (if available)
    "KnowledgeMemory",
    "ContextInjector", 
    "FeedbackLearner",
    "LearnedFact",
    "ReasoningChain",
    "FeedbackType",
    "HAS_LEARNING",
]

__version__ = "0.1.0"
