"""RAG Pipeline - Main orchestrator for the RAG system."""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, AsyncIterator
from uuid import UUID

from omniscient_core import AsyncContextMixin

from .models import Document, Chunk, RAGConfig, RetrievalResult, KnowledgeScore
from .chunkers import ChunkerFactory, BaseChunker
from .store import PostgresVectorStore, DatabaseConfig
from .search import HybridSearcher, SearchConfig
from .metrics import KnowledgeScorer, QuestionGenerator, ScoringConfig


class RAGPipeline(AsyncContextMixin):
    """Main RAG pipeline orchestrator.
    
    Coordinates all RAG components:
    - Document ingestion with configurable chunking
    - Embedding generation via Ollama
    - Storage in PostgreSQL + pgvector
    - Hybrid search (vector + BM25)
    - Knowledge metrics tracking
    
    Example:
        >>> pipeline = RAGPipeline.create(
        ...     db_url="postgresql://...",
        ...     embed_fn=ollama_provider.embed,
        ...     llm_fn=ollama_provider.generate,
        ... )
        >>> await pipeline.initialize()
        >>> 
        >>> # Ingest documents
        >>> await pipeline.ingest_directory("/path/to/docs")
        >>> 
        >>> # Query
        >>> results = await pipeline.query("How to configure Spark?")
        >>> 
        >>> # Evaluate knowledge
        >>> score = await pipeline.evaluate()
    """
    
    def __init__(
        self,
        store: PostgresVectorStore,
        chunker: BaseChunker,
        embed_fn: Callable[[str], Any],
        llm_fn: Optional[Callable[[str], Any]] = None,
        config: Optional[RAGConfig] = None,
    ):
        """Initialize RAG pipeline.
        
        Args:
            store: PostgreSQL vector store
            chunker: Document chunker
            embed_fn: Async function for generating embeddings
            llm_fn: Optional LLM function for question generation
            config: RAG configuration
        """
        self.store = store
        self.chunker = chunker
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.config = config or RAGConfig()
        
        # Initialize components
        self.searcher = HybridSearcher(
            store=store,
            embed_fn=embed_fn,
            config=SearchConfig(
                top_k=self.config.top_k,
                alpha=self.config.hybrid_alpha,
            ),
        )
        
        self.scorer: Optional[KnowledgeScorer] = None
        self.question_generator: Optional[QuestionGenerator] = None
        
        if llm_fn:
            self.question_generator = QuestionGenerator(
                llm_fn=llm_fn,
                questions_per_doc=self.config.questions_per_document,
            )
            self.scorer = KnowledgeScorer(
                store=store,
                searcher=self.searcher,
                llm_fn=llm_fn,
            )
    
    @classmethod
    def create(
        cls,
        db_url: str,
        embed_fn: Callable[[str], Any],
        llm_fn: Optional[Callable[[str], Any]] = None,
        chunking_strategy: str = "auto",
        chunk_size: int = 512,
        chunk_overlap: float = 0.1,
        **kwargs,
    ) -> "RAGPipeline":
        """Factory method to create a RAG pipeline.
        
        Args:
            db_url: PostgreSQL connection URL
            embed_fn: Async embedding function
            llm_fn: Optional async LLM function
            chunking_strategy: Chunking strategy name
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap ratio
            **kwargs: Additional RAGConfig options
            
        Returns:
            Configured RAGPipeline instance
        """
        store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
        chunker = ChunkerFactory.create(chunking_strategy, chunk_size, chunk_overlap)
        config = RAGConfig(
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
        
        return cls(
            store=store,
            chunker=chunker,
            embed_fn=embed_fn,
            llm_fn=llm_fn,
            config=config,
        )
    
    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        await self.store.initialize()
    
    async def close(self) -> None:
        """Close database connection."""
        await self.store.close()
    
    # =========================================================================
    # Ingestion
    # =========================================================================
    
    async def ingest_document(
        self,
        document: Document,
        generate_questions: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Ingest a single document.
        
        Args:
            document: Document to ingest
            generate_questions: Override auto-question generation
            
        Returns:
            Ingestion result with chunk count and stats
        """
        generate_q = generate_questions if generate_questions is not None else self.config.auto_generate_questions
        
        # Store document
        await self.store.insert_document(document)
        
        # Chunk document
        chunks = self.chunker.chunk(document)
        
        # Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = await self._embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Store chunks
        await self.store.insert_chunks(chunks)
        
        # Generate test questions (on ingestion)
        questions_generated = 0
        if generate_q and self.question_generator:
            questions = await self.question_generator.generate(document)
            await self.store.insert_questions(questions)
            questions_generated = len(questions)
        
        return {
            "document_id": str(document.id),
            "source": document.source,
            "chunks_created": len(chunks),
            "questions_generated": questions_generated,
        }
    
    async def ingest_text(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest text content as a document.
        
        Args:
            content: Text content
            source: Source identifier
            metadata: Optional metadata
            
        Returns:
            Ingestion result
        """
        document = Document(
            content=content,
            source=source,
            metadata=metadata or {},
        )
        return await self.ingest_document(document)
    
    async def ingest_file(
        self,
        file_path: Path,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """Ingest a file.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Returns:
            Ingestion result
        """
        path = Path(file_path)
        content = path.read_text(encoding=encoding)
        
        document = Document(
            content=content,
            source=str(path),
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
            },
        )
        return await self.ingest_document(document)
    
    async def ingest_directory(
        self,
        directory: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Ingest all matching files from a directory.
        
        Args:
            directory: Directory path
            patterns: Glob patterns to match (default: common doc/code extensions)
            recursive: Search subdirectories
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Summary with total documents and chunks
        """
        directory = Path(directory)
        
        if patterns is None:
            patterns = ["*.md", "*.txt", "*.rst", "*.py", "*.js", "*.ts", "*.yaml", "*.json"]
        
        # Collect files
        files = []
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        
        # Filter out hidden files and common excludes
        files = [
            f for f in files
            if not any(part.startswith('.') for part in f.parts)
            and 'node_modules' not in str(f)
            and '__pycache__' not in str(f)
        ]
        
        results = []
        total = len(files)
        
        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, file_path.name)
            
            try:
                result = await self.ingest_file(file_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "source": str(file_path),
                    "error": str(e),
                    "chunks_created": 0,
                })
        
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        return {
            "total_files": total,
            "successful": len(successful),
            "failed": len(failed),
            "total_chunks": sum(r.get("chunks_created", 0) for r in successful),
            "total_questions": sum(r.get("questions_generated", 0) for r in successful),
            "failures": failed,
        }
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        # Check if embed_fn supports batch
        if hasattr(self.embed_fn, 'embed_batch'):
            return await self.embed_fn.embed_batch(texts)
        
        # Otherwise, process one at a time
        for text in texts:
            embedding = await self.embed_fn(text)
            embeddings.append(embedding)
        
        return embeddings
    
    # =========================================================================
    # Query / Search
    # =========================================================================
    
    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[RetrievalResult]:
        """Query the knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Hybrid search alpha (0=BM25, 1=vector)
            document_ids: Filter by specific documents
            
        Returns:
            List of retrieval results with scores
        """
        return await self.searcher.search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            document_ids=document_ids,
        )
    
    async def query_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """Query and return formatted context string.
        
        Useful for direct insertion into LLM prompts.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Formatted context string with sources
        """
        results = await self.query(query, top_k=top_k)
        
        context_parts = []
        for result in results:
            source = result.source
            content = result.chunk.content
            context_parts.append(f"[Source: {source}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    # =========================================================================
    # Knowledge Metrics
    # =========================================================================
    
    async def evaluate(
        self,
        document_id: Optional[UUID] = None,
        topic: Optional[str] = None,
    ) -> KnowledgeScore:
        """Run on-demand knowledge evaluation.
        
        Args:
            document_id: Evaluate specific document
            topic: Evaluate specific topic
            
        Returns:
            KnowledgeScore with metrics
        """
        if not self.scorer:
            raise RuntimeError("Scorer not available - LLM function required")
        
        return await self.scorer.evaluate(
            document_id=document_id,
            topic=topic,
        )
    
    async def get_score_history(self, limit: int = 30) -> List[KnowledgeScore]:
        """Get knowledge score history.
        
        Args:
            limit: Number of scores to return
            
        Returns:
            List of historical scores
        """
        return await self.store.get_score_history(limit=limit)
    
    async def get_latest_score(self) -> Optional[KnowledgeScore]:
        """Get the most recent knowledge score."""
        return await self.store.get_latest_score()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        stats = await self.store.get_stats()
        
        # Add latest score
        latest = await self.get_latest_score()
        if latest:
            stats["latest_score"] = {
                "overall": latest.overall_score,
                "precision": latest.retrieval_precision,
                "accuracy": latest.answer_accuracy,
                "timestamp": latest.timestamp.isoformat(),
            }
        
        return stats
    
    # =========================================================================
    # Management
    # =========================================================================
    
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and all its chunks."""
        return await self.store.delete_document(document_id)
    
    async def list_documents(self, limit: int = 100) -> List[Document]:
        """List all documents."""
        return await self.store.list_documents(limit=limit)
    
    async def clear_all(self) -> Dict[str, int]:
        """Clear all data (use with caution).
        
        Returns:
            Count of deleted items
        """
        docs = await self.list_documents(limit=10000)
        deleted = 0
        for doc in docs:
            if await self.delete_document(doc.id):
                deleted += 1
        
        return {"documents_deleted": deleted}
