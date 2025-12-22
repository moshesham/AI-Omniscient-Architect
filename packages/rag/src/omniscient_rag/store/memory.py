import math
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from ..models import Chunk, RetrievalResult, Document
from .base import VectorStore

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class InMemoryVectorStore(VectorStore):
    """In-memory vector store for testing and development."""

    def __init__(self):
        self.documents: Dict[UUID, Document] = {}
        self.chunks: List[Chunk] = []
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def ingest_document(self, document: Document, chunks: List[Chunk]) -> None:
        self.documents[document.id] = document
        self.chunks.extend(chunks)

    async def search_vectors(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        results = []
        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            if score >= min_score:
                results.append(RetrievalResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def search_hybrid(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        # For in-memory, we'll just use vector search for now as BM25 is complex to implement from scratch
        # A real implementation would use rank_bm25 or similar
        return await self.search_vectors(query_embedding, top_k)
