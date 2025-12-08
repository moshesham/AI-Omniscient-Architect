"""Data models for RAG system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    SEMANTIC = "semantic"
    AST = "ast"
    AUTO = "auto"


@dataclass
class Document:
    """A document to be ingested into the RAG system.
    
    Attributes:
        id: Unique document identifier
        content: Full document text
        source: File path or URL
        metadata: Additional metadata (language, title, etc.)
        created_at: Ingestion timestamp
    """
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def language(self) -> Optional[str]:
        """Get document language from metadata."""
        return self.metadata.get("language")
    
    @property
    def file_extension(self) -> Optional[str]:
        """Extract file extension from source."""
        if "." in self.source:
            return self.source.rsplit(".", 1)[-1].lower()
        return None


@dataclass
class Chunk:
    """A chunk of text extracted from a document.
    
    Attributes:
        id: Unique chunk identifier
        document_id: Parent document ID
        content: Chunk text content
        embedding: Vector embedding (768-dim for nomic-embed-text)
        metadata: Chunk-specific metadata (position, type, etc.)
        start_char: Start character offset in original document
        end_char: End character offset in original document
    """
    content: str
    document_id: UUID
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    id: UUID = field(default_factory=uuid4)
    start_char: int = 0
    end_char: int = 0
    
    @property
    def token_count(self) -> int:
        """Approximate token count (1 token ≈ 4 chars)."""
        return len(self.content) // 4
    
    @property
    def chunk_type(self) -> str:
        """Get chunk type from metadata (e.g., 'function', 'class', 'paragraph')."""
        return self.metadata.get("type", "text")


@dataclass
class RetrievalResult:
    """Result from hybrid search retrieval.
    
    Attributes:
        chunk: The retrieved chunk
        vector_score: Cosine similarity score (0-1)
        bm25_score: BM25 relevance score
        combined_score: RRF-combined score
        rank: Final ranking position
        source: Original document source path
    """
    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0
    
    @property
    def source(self) -> str:
        """Get source from chunk metadata."""
        return self.chunk.metadata.get("source", "unknown")
    
    def to_context_string(self) -> str:
        """Format as context string for LLM prompt."""
        return f"[Source: {self.source}]\n{self.chunk.content}"


@dataclass
class KnowledgeQuestion:
    """Auto-generated test question for knowledge evaluation.
    
    Attributes:
        id: Unique question identifier
        document_id: Source document ID
        question: The test question
        expected_answer: Expected answer (for scoring)
        topic: Topic/category of the question
        difficulty: Question difficulty (easy, medium, hard)
    """
    question: str
    expected_answer: str
    document_id: UUID
    topic: str = "general"
    difficulty: str = "medium"
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class KnowledgeScore:
    """Knowledge evaluation metrics.
    
    Attributes:
        id: Unique score identifier
        retrieval_precision: Relevant chunks / total retrieved (0-1)
        answer_accuracy: Answer quality score (0-100)
        coverage_ratio: % of knowledge base used (0-1)
        questions_evaluated: Number of test questions used
        timestamp: When evaluation was performed
        details: Detailed per-question results
    """
    retrieval_precision: float
    answer_accuracy: float
    coverage_ratio: float
    questions_evaluated: int = 0
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        """Compute overall knowledge score (0-100)."""
        # Weighted average: 40% accuracy, 35% precision, 25% coverage
        return (
            self.answer_accuracy * 0.4 +
            self.retrieval_precision * 100 * 0.35 +
            self.coverage_ratio * 100 * 0.25
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "retrieval_precision": self.retrieval_precision,
            "answer_accuracy": self.answer_accuracy,
            "coverage_ratio": self.coverage_ratio,
            "questions_evaluated": self.questions_evaluated,
            "overall_score": self.overall_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline.
    
    Attributes:
        chunking_strategy: Which chunking strategy to use
        chunk_size: Target chunk size in tokens (for fixed chunking)
        chunk_overlap: Overlap between chunks (0-1 ratio)
        embedding_model: Ollama model for embeddings
        top_k: Number of results to retrieve
        hybrid_alpha: Weight for vector vs BM25 (0=BM25 only, 1=vector only)
        auto_generate_questions: Generate test questions on ingestion
        questions_per_document: Number of questions to generate per doc
    """
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    chunk_size: int = 512
    chunk_overlap: float = 0.1
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    # Embedding robustness controls
    embedding_timeout: float = 30.0
    embedding_max_retries: int = 2
    embedding_batch_serial: bool = True
    top_k: int = 5
    hybrid_alpha: float = 0.5
    auto_generate_questions: bool = True
    questions_per_document: int = 3
    
    # Database settings
    db_connection_string: Optional[str] = None
    db_pool_size: int = 5
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.chunk_overlap < 1:
            raise ValueError("chunk_overlap must be between 0 and 1")
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValueError("hybrid_alpha must be between 0 and 1")
        if self.chunk_size < 64:
            raise ValueError("chunk_size must be at least 64 tokens")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.embedding_timeout <= 0:
            raise ValueError("embedding_timeout must be positive")
        if self.embedding_max_retries < 1:
            raise ValueError("embedding_max_retries must be at least 1")
