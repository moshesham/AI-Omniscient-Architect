from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from ..models import Chunk, RetrievalResult, Document

class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the store (connections, schemas, etc)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        pass

    @abstractmethod
    async def ingest_document(self, document: Document, chunks: List[Chunk]) -> None:
        """Ingest a document and its chunks."""
        pass

    @abstractmethod
    async def search_vectors(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """Search using vector similarity."""
        pass

    @abstractmethod
    async def search_hybrid(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """Search using hybrid (vector + keyword) strategy."""
        pass
