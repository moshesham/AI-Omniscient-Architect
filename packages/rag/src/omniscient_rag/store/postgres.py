"""PostgreSQL + pgvector storage for RAG system."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import UUID

from omniscient_core import optional_import

HAS_POSTGRES, _ = optional_import("psycopg")
if HAS_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    from pgvector.psycopg import register_vector_async

from ..models import Document, Chunk, KnowledgeQuestion, KnowledgeScore


def _ensure_uuid(value: Union[str, UUID]) -> UUID:
    """Convert value to UUID if it's a string."""
    return value if isinstance(value, UUID) else UUID(str(value))


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration.
    
    Attributes:
        connection_string: PostgreSQL connection URL
        pool_size: Connection pool size
        embedding_dimensions: Vector dimensions (768 for nomic-embed-text)
    """
    connection_string: str = "postgresql://omniscient:localdev@localhost:5432/omniscient"
    pool_size: int = 5
    embedding_dimensions: int = 768
    schema: str = "rag"
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        import os
        return cls(
            connection_string=os.getenv(
                "DATABASE_URL",
                "postgresql://omniscient:localdev@localhost:5432/omniscient"
            ),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
        )


class PostgresVectorStore:
    """PostgreSQL + pgvector storage backend.
    
    Provides:
    - Vector storage with pgvector extension
    - Full-text search with tsvector/tsquery (BM25-like ranking)
    - Document and chunk management
    - Knowledge metrics storage
    
    Example:
        >>> store = PostgresVectorStore(config)
        >>> await store.initialize()
        >>> await store.insert_chunks(chunks)
        >>> results = await store.search_vectors(query_embedding, top_k=5)
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize store.
        
        Args:
            config: Database configuration
        """
        if not HAS_POSTGRES:
            raise ImportError(
                "psycopg and pgvector are required. "
                "Install with: pip install 'psycopg[binary]' pgvector"
            )
        
        self.config = config or DatabaseConfig.from_env()
        self._pool: Optional[AsyncConnectionPool] = None
    
    async def initialize(self) -> None:
        """Initialize database connection pool and schema."""
        self._pool = AsyncConnectionPool(
            self.config.connection_string,
            min_size=1,
            max_size=self.config.pool_size,
            open=False,
        )
        await self._pool.open()
        
        # Create schema and extensions first
        async with self._pool.connection() as conn:
            await self._create_extensions(conn)
        
        # Register pgvector type (must be done after extension is created)
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            await self._create_schema(conn)
    
    async def _create_extensions(self, conn) -> None:
        """Create required PostgreSQL extensions."""
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def _create_schema(self, conn) -> None:
        """Create database schema if not exists."""
        schema = self.config.schema
        dim = self.config.embedding_dimensions
        
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Documents table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.documents (
                id UUID PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Chunks table with vector embedding and tsvector for full-text search
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.chunks (
                id UUID PRIMARY KEY,
                document_id UUID REFERENCES {schema}.documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                embedding vector({dim}),
                metadata JSONB DEFAULT '{{}}',
                start_char INTEGER DEFAULT 0,
                end_char INTEGER DEFAULT 0,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
            ON {schema}.chunks 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS chunks_tsv_idx 
            ON {schema}.chunks 
            USING GIN (tsv)
        """)
        
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS chunks_document_id_idx 
            ON {schema}.chunks (document_id)
        """)
        
        # Knowledge questions table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.knowledge_questions (
                id UUID PRIMARY KEY,
                document_id UUID REFERENCES {schema}.documents(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                expected_answer TEXT NOT NULL,
                topic TEXT DEFAULT 'general',
                difficulty TEXT DEFAULT 'medium',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Knowledge scores table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.knowledge_scores (
                id UUID PRIMARY KEY,
                retrieval_precision FLOAT NOT NULL,
                answer_accuracy FLOAT NOT NULL,
                coverage_ratio FLOAT NOT NULL,
                questions_evaluated INTEGER DEFAULT 0,
                details JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        await conn.commit()
    
    # =========================================================================
    # Document Operations
    # =========================================================================
    
    async def insert_document(self, document: Document) -> UUID:
        """Insert a document.
        
        Args:
            document: Document to insert
            
        Returns:
            Document ID
        """
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            await conn.execute(
                f"""
                INSERT INTO {schema}.documents (id, source, content, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                (
                    str(document.id),
                    document.source,
                    document.content,
                    json.dumps(document.metadata),
                    document.created_at,
                )
            )
            await conn.commit()
        
        return document.id
    
    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get a document by ID."""
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT * FROM {schema}.documents WHERE id = %s",
                    (str(document_id),)
                )
                row = await cur.fetchone()
                
                if not row:
                    return None
                
                return Document(
                    id=_ensure_uuid(row["id"]),
                    content=row["content"],
                    source=row["source"],
                    metadata=row["metadata"] or {},
                    created_at=row["created_at"],
                )
    
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and its chunks."""
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            result = await conn.execute(
                f"DELETE FROM {schema}.documents WHERE id = %s",
                (str(document_id),)
            )
            await conn.commit()
            return result.rowcount > 0
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """List all documents."""
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT * FROM {schema}.documents 
                    ORDER BY created_at DESC 
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset)
                )
                rows = await cur.fetchall()
                
                return [
                    Document(
                        id=_ensure_uuid(row["id"]),
                        content=row["content"],
                        source=row["source"],
                        metadata=row["metadata"] or {},
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
    
    # =========================================================================
    # Chunk Operations
    # =========================================================================
    
    async def insert_chunks(self, chunks: List[Chunk]) -> int:
        """Insert multiple chunks.
        
        Args:
            chunks: Chunks to insert (must have embeddings)
            
        Returns:
            Number of chunks inserted
        """
        if not chunks:
            return 0
        
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            
            # Use COPY for bulk insert
            async with conn.cursor() as cur:
                for chunk in chunks:
                    embedding = chunk.embedding if chunk.embedding else [0.0] * self.config.embedding_dimensions
                    
                    await cur.execute(
                        f"""
                        INSERT INTO {schema}.chunks 
                        (id, document_id, content, embedding, metadata, start_char, end_char)
                        VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            str(chunk.id),
                            str(chunk.document_id),
                            chunk.content,
                            embedding,
                            json.dumps(chunk.metadata),
                            chunk.start_char,
                            chunk.end_char,
                        )
                    )
            
            await conn.commit()
        
        return len(chunks)
    
    async def get_chunks_for_document(self, document_id: UUID) -> List[Chunk]:
        """Get all chunks for a document."""
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT * FROM {schema}.chunks 
                    WHERE document_id = %s 
                    ORDER BY start_char
                    """,
                    (str(document_id),)
                )
                rows = await cur.fetchall()
                
                return [
                    Chunk(
                        id=_ensure_uuid(row["id"]),
                        document_id=_ensure_uuid(row["document_id"]),
                        content=row["content"],
                        embedding=list(row["embedding"]) if row["embedding"] is not None else None,
                        metadata=row["metadata"] or {},
                        start_char=row["start_char"],
                        end_char=row["end_char"],
                    )
                    for row in rows
                ]
    
    # =========================================================================
    # Vector Search
    # =========================================================================
    
    async def search_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            document_ids: Optional filter by document IDs
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            
            async with conn.cursor(row_factory=dict_row) as cur:
                if document_ids:
                    doc_filter = f"AND document_id = ANY(%s::uuid[])"
                    params = (query_embedding, [str(d) for d in document_ids], top_k)
                else:
                    doc_filter = ""
                    params = (query_embedding, top_k)
                
                await cur.execute(
                    f"""
                    SELECT *, 1 - (embedding <=> %s::vector) AS similarity
                    FROM {schema}.chunks
                    WHERE embedding IS NOT NULL {doc_filter}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """.replace("%s::uuid[]", "%s::uuid[]" if document_ids else ""),
                    (query_embedding, *([str(d) for d in document_ids] if document_ids else []), query_embedding, top_k)
                    if document_ids else (query_embedding, query_embedding, top_k)
                )
                
                rows = await cur.fetchall()
                
                return [
                    (
                        Chunk(
                            id=_ensure_uuid(row["id"]),
                            document_id=_ensure_uuid(row["document_id"]),
                            content=row["content"],
                            embedding=list(row["embedding"]) if row["embedding"] is not None else None,
                            metadata=row["metadata"] or {},
                            start_char=row["start_char"],
                            end_char=row["end_char"],
                        ),
                        row["similarity"],
                    )
                    for row in rows
                ]
    
    # =========================================================================
    # Full-Text Search (BM25-like)
    # =========================================================================
    
    async def search_fulltext(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search using PostgreSQL full-text search.
        
        Uses ts_rank for BM25-like scoring.
        
        Args:
            query: Search query text
            top_k: Number of results
            document_ids: Optional filter by document IDs
            
        Returns:
            List of (chunk, score) tuples
        """
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                doc_filter = ""
                params: tuple = (query, top_k)
                
                if document_ids:
                    doc_filter = "AND document_id = ANY(%s::uuid[])"
                    params = (query, [str(d) for d in document_ids], top_k)
                
                await cur.execute(
                    f"""
                    SELECT *, ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS rank
                    FROM {schema}.chunks
                    WHERE tsv @@ plainto_tsquery('english', %s) {doc_filter}
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, query, *([str(d) for d in document_ids] if document_ids else []), top_k)
                    if document_ids else (query, query, top_k)
                )
                
                rows = await cur.fetchall()
                
                return [
                    (
                        Chunk(
                            id=_ensure_uuid(row["id"]),
                            document_id=_ensure_uuid(row["document_id"]),
                            content=row["content"],
                            metadata=row["metadata"] or {},
                            start_char=row["start_char"],
                            end_char=row["end_char"],
                        ),
                        row["rank"],
                    )
                    for row in rows
                ]
    
    # =========================================================================
    # Knowledge Questions
    # =========================================================================
    
    async def insert_questions(self, questions: List[KnowledgeQuestion]) -> int:
        """Insert knowledge test questions."""
        if not questions:
            return 0
        
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for q in questions:
                    await cur.execute(
                        f"""
                        INSERT INTO {schema}.knowledge_questions
                        (id, document_id, question, expected_answer, topic, difficulty, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            str(q.id),
                            str(q.document_id),
                            q.question,
                            q.expected_answer,
                            q.topic,
                            q.difficulty,
                            q.created_at,
                        )
                    )
            await conn.commit()
        
        return len(questions)
    
    async def get_questions(
        self,
        document_id: Optional[UUID] = None,
        topic: Optional[str] = None,
        limit: int = 50,
    ) -> List[KnowledgeQuestion]:
        """Get knowledge questions."""
        schema = self.config.schema
        
        conditions = []
        params: List[Any] = []
        
        if document_id:
            conditions.append("document_id = %s")
            params.append(str(document_id))
        
        if topic:
            conditions.append("topic = %s")
            params.append(topic)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT * FROM {schema}.knowledge_questions
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    tuple(params)
                )
                rows = await cur.fetchall()
                
                return [
                    KnowledgeQuestion(
                        id=_ensure_uuid(row["id"]),
                        document_id=_ensure_uuid(row["document_id"]),
                        question=row["question"],
                        expected_answer=row["expected_answer"],
                        topic=row["topic"],
                        difficulty=row["difficulty"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
    
    # =========================================================================
    # Knowledge Scores
    # =========================================================================
    
    async def insert_score(self, score: KnowledgeScore) -> UUID:
        """Insert a knowledge score."""
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            await conn.execute(
                f"""
                INSERT INTO {schema}.knowledge_scores
                (id, retrieval_precision, answer_accuracy, coverage_ratio, 
                 questions_evaluated, details, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(score.id),
                    score.retrieval_precision,
                    score.answer_accuracy,
                    score.coverage_ratio,
                    score.questions_evaluated,
                    json.dumps(score.details),
                    score.timestamp,
                )
            )
            await conn.commit()
        
        return score.id
    
    async def get_score_history(self, limit: int = 30) -> List[KnowledgeScore]:
        """Get recent knowledge scores."""
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT * FROM {schema}.knowledge_scores
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                rows = await cur.fetchall()
                
                return [
                    KnowledgeScore(
                        id=_ensure_uuid(row["id"]),
                        retrieval_precision=row["retrieval_precision"],
                        answer_accuracy=row["answer_accuracy"],
                        coverage_ratio=row["coverage_ratio"],
                        questions_evaluated=row["questions_evaluated"],
                        details=row["details"] or {},
                        timestamp=row["created_at"],
                    )
                    for row in rows
                ]
    
    async def get_latest_score(self) -> Optional[KnowledgeScore]:
        """Get the most recent knowledge score."""
        history = await self.get_score_history(limit=1)
        return history[0] if history else None
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # Count documents
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.documents")
                doc_count = (await cur.fetchone())["count"]
                
                # Count chunks
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.chunks")
                chunk_count = (await cur.fetchone())["count"]
                
                # Count chunks with embeddings
                await cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.chunks WHERE embedding IS NOT NULL"
                )
                embedded_count = (await cur.fetchone())["count"]
                
                # Count questions
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.knowledge_questions")
                question_count = (await cur.fetchone())["count"]
                
                # Count scores
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.knowledge_scores")
                score_count = (await cur.fetchone())["count"]
                
                return {
                    "documents": doc_count,
                    "chunks": chunk_count,
                    "chunks_with_embeddings": embedded_count,
                    "knowledge_questions": question_count,
                    "knowledge_scores": score_count,
                    "embedding_coverage": embedded_count / chunk_count if chunk_count > 0 else 0,
                }
    
    async def clear_all(self) -> Dict[str, int]:
        """Clear all data from the store.
        
        Returns:
            Dictionary with counts of deleted items
        """
        schema = self.config.schema
        deleted = {}
        
        async with self._pool.connection() as conn:
            # Delete in order respecting foreign keys
            result = await conn.execute(f"DELETE FROM {schema}.knowledge_scores")
            deleted["knowledge_scores"] = result.rowcount if hasattr(result, 'rowcount') else 0
            
            result = await conn.execute(f"DELETE FROM {schema}.knowledge_questions")
            deleted["knowledge_questions"] = result.rowcount if hasattr(result, 'rowcount') else 0
            
            result = await conn.execute(f"DELETE FROM {schema}.chunks")
            deleted["chunks"] = result.rowcount if hasattr(result, 'rowcount') else 0
            
            result = await conn.execute(f"DELETE FROM {schema}.documents")
            deleted["documents"] = result.rowcount if hasattr(result, 'rowcount') else 0
        
        return deleted
