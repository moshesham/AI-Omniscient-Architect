"""Storage backends for RAG system.

Provides:
- PostgresVectorStore: PostgreSQL + pgvector storage
"""

from .postgres import PostgresVectorStore, DatabaseConfig

__all__ = [
    "PostgresVectorStore",
    "DatabaseConfig",
]
