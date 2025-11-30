"""Hybrid search combining vector similarity and BM25."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Any
from uuid import UUID

from ..models import Chunk, RetrievalResult
from ..store.postgres import PostgresVectorStore


@dataclass
class SearchConfig:
    """Configuration for hybrid search.
    
    Attributes:
        top_k: Number of results to return
        alpha: Weight for vector vs BM25 (0=BM25 only, 1=vector only)
        vector_top_k: Number of vector results to fetch before fusion
        bm25_top_k: Number of BM25 results to fetch before fusion
        rrf_k: RRF constant (higher = less aggressive rank compression)
        min_score: Minimum combined score threshold
    """
    top_k: int = 5
    alpha: float = 0.5
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    min_score: float = 0.0


class HybridSearcher:
    """Hybrid search combining vector similarity and BM25 full-text search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both
    search methods into a unified result set.
    
    RRF Formula: score = Σ 1 / (k + rank_i)
    
    Where k is a constant (default 60) and rank_i is the rank in each list.
    
    Example:
        >>> searcher = HybridSearcher(store, embed_fn)
        >>> results = await searcher.search("spark configuration", top_k=5)
        >>> for result in results:
        ...     print(f"{result.source}: {result.combined_score:.3f}")
    """
    
    def __init__(
        self,
        store: PostgresVectorStore,
        embed_fn: Callable[[str], Any],  # async function returning List[float]
        config: Optional[SearchConfig] = None,
    ):
        """Initialize hybrid searcher.
        
        Args:
            store: PostgreSQL vector store
            embed_fn: Async function to embed query text
            config: Search configuration
        """
        self.store = store
        self.embed_fn = embed_fn
        self.config = config or SearchConfig()
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid search.
        
        Args:
            query: Search query text
            top_k: Override default top_k
            alpha: Override default alpha (vector weight)
            document_ids: Filter by specific documents
            
        Returns:
            List of RetrievalResult sorted by combined score
        """
        top_k = top_k or self.config.top_k
        alpha = alpha if alpha is not None else self.config.alpha
        
        # Get results from both search methods
        vector_results = await self._vector_search(query, document_ids)
        bm25_results = await self._bm25_search(query, document_ids)
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            alpha=alpha,
        )
        
        # Filter and limit
        filtered = [r for r in combined if r.combined_score >= self.config.min_score]
        return filtered[:top_k]
    
    async def search_vector_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[RetrievalResult]:
        """Search using vector similarity only.
        
        Args:
            query: Search query
            top_k: Number of results
            document_ids: Filter by documents
            
        Returns:
            List of results sorted by vector similarity
        """
        top_k = top_k or self.config.top_k
        results = await self._vector_search(query, document_ids)
        
        return [
            RetrievalResult(
                chunk=chunk,
                vector_score=score,
                bm25_score=0.0,
                combined_score=score,
                rank=i + 1,
            )
            for i, (chunk, score) in enumerate(results[:top_k])
        ]
    
    async def search_bm25_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[RetrievalResult]:
        """Search using BM25 full-text only.
        
        Args:
            query: Search query
            top_k: Number of results
            document_ids: Filter by documents
            
        Returns:
            List of results sorted by BM25 score
        """
        top_k = top_k or self.config.top_k
        results = await self._bm25_search(query, document_ids)
        
        return [
            RetrievalResult(
                chunk=chunk,
                vector_score=0.0,
                bm25_score=score,
                combined_score=score,
                rank=i + 1,
            )
            for i, (chunk, score) in enumerate(results[:top_k])
        ]
    
    async def _vector_search(
        self,
        query: str,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[tuple]:
        """Perform vector similarity search."""
        # Generate query embedding
        query_embedding = await self.embed_fn(query)
        
        # Search vector store
        results = await self.store.search_vectors(
            query_embedding=query_embedding,
            top_k=self.config.vector_top_k,
            document_ids=document_ids,
        )
        
        return results
    
    async def _bm25_search(
        self,
        query: str,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[tuple]:
        """Perform BM25 full-text search."""
        results = await self.store.search_fulltext(
            query=query,
            top_k=self.config.bm25_top_k,
            document_ids=document_ids,
        )
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[tuple],
        bm25_results: List[tuple],
        alpha: float = 0.5,
    ) -> List[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion.
        
        RRF Score = α * (1/(k + vector_rank)) + (1-α) * (1/(k + bm25_rank))
        
        Args:
            vector_results: List of (chunk, score) from vector search
            bm25_results: List of (chunk, score) from BM25 search
            alpha: Weight for vector results (0-1)
            
        Returns:
            Combined and sorted results
        """
        k = self.config.rrf_k
        
        # Build chunk ID to result mapping
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        
        # Process vector results
        for rank, (chunk, score) in enumerate(vector_results, start=1):
            chunk_id = str(chunk.id)
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "vector_score": score,
                    "vector_rank": rank,
                    "bm25_score": 0.0,
                    "bm25_rank": None,
                }
            else:
                chunk_scores[chunk_id]["vector_score"] = score
                chunk_scores[chunk_id]["vector_rank"] = rank
        
        # Process BM25 results
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            chunk_id = str(chunk.id)
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "vector_rank": None,
                    "bm25_score": score,
                    "bm25_rank": rank,
                }
            else:
                chunk_scores[chunk_id]["bm25_score"] = score
                chunk_scores[chunk_id]["bm25_rank"] = rank
        
        # Calculate RRF scores
        results = []
        for chunk_id, data in chunk_scores.items():
            # RRF contribution from each list
            vector_rrf = 0.0
            bm25_rrf = 0.0
            
            if data["vector_rank"] is not None:
                vector_rrf = 1.0 / (k + data["vector_rank"])
            
            if data["bm25_rank"] is not None:
                bm25_rrf = 1.0 / (k + data["bm25_rank"])
            
            # Weighted combination
            combined_score = alpha * vector_rrf + (1 - alpha) * bm25_rrf
            
            results.append(RetrievalResult(
                chunk=data["chunk"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=combined_score,
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign final ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def explain_ranking(self, result: RetrievalResult) -> str:
        """Generate human-readable explanation of ranking.
        
        Args:
            result: Retrieval result to explain
            
        Returns:
            Explanation string
        """
        alpha = self.config.alpha
        
        explanation = f"Rank #{result.rank} (score: {result.combined_score:.4f})\n"
        explanation += f"  Vector similarity: {result.vector_score:.4f} (weight: {alpha:.0%})\n"
        explanation += f"  BM25 relevance: {result.bm25_score:.4f} (weight: {1-alpha:.0%})\n"
        explanation += f"  Source: {result.source}\n"
        
        return explanation
