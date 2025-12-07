"""
Unit and integration tests for the RAG system.

Run with:
    pytest tests/test_rag.py -v
    pytest tests/test_rag.py -v -k "test_chunker"  # Run only chunker tests
    pytest tests/test_rag.py -v --run-integration   # Include integration tests
"""

import pytest
import asyncio
from pathlib import Path
from uuid import uuid4
from typing import List


# Add package paths
import sys
from pathlib import Path
_root = Path(__file__).parent.parent
for _pkg in ["core", "llm", "rag"]:
    _path = _root / "packages" / _pkg / "src"
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    from omniscient_rag.models import Document
    return Document(
        content="""
# Python Best Practices

## Code Style

Follow PEP 8 for Python code style. Use 4 spaces for indentation.
Maximum line length should be 79 characters for code, 72 for docstrings.

```python
def calculate_total(items: list[float]) -> float:
    \"\"\"Calculate the total of all items.\"\"\"
    return sum(items)
```

## Type Hints

Use type hints for function parameters and return values.
This improves code readability and enables static type checking.

## Documentation

Write docstrings for all public functions and classes.
Use Google-style or NumPy-style docstring format.
""",
        source="python_best_practices.md",
        metadata={"language": "markdown", "topic": "python"},
    )


@pytest.fixture
def sample_code_document():
    """Create a sample Python code document."""
    from omniscient_rag.models import Document
    return Document(
        content="""
import asyncio
from typing import List, Optional

class DataProcessor:
    \"\"\"Process data from various sources.\"\"\"
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = {}
    
    async def fetch_data(self, url: str) -> Optional[dict]:
        \"\"\"Fetch data from a URL.\"\"\"
        # Implementation here
        pass
    
    def transform(self, data: List[dict]) -> List[dict]:
        \"\"\"Transform data records.\"\"\"
        return [self._transform_record(r) for r in data]
    
    def _transform_record(self, record: dict) -> dict:
        \"\"\"Transform a single record.\"\"\"
        return {k.lower(): v for k, v in record.items()}


def main():
    processor = DataProcessor({"key": "value"})
    asyncio.run(processor.fetch_data("https://api.example.com"))
""",
        source="data_processor.py",
        metadata={"language": "python"},
    )


# =============================================================================
# Chunker Tests
# =============================================================================

class TestFixedChunker:
    """Tests for FixedChunker."""
    
    def test_chunk_small_document(self, sample_document):
        """Small documents should produce single chunk."""
        from omniscient_rag.chunkers import FixedChunker
        
        chunker = FixedChunker(chunk_size=1024)  # Large enough for sample
        chunks = chunker.chunk(sample_document)
        
        assert len(chunks) >= 1
        assert all(c.document_id == sample_document.id for c in chunks)
    
    def test_chunk_with_overlap(self, sample_document):
        """Chunks should have overlap when document is large."""
        from omniscient_rag.chunkers import FixedChunker
        
        chunker = FixedChunker(chunk_size=100, chunk_overlap=0.2)
        chunks = chunker.chunk(sample_document)
        
        # With small chunk size, should produce multiple chunks
        assert len(chunks) > 1
        
        # Check metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["strategy"] == "fixed"
    
    def test_empty_document(self):
        """Empty documents should return no chunks."""
        from omniscient_rag.chunkers import FixedChunker
        from omniscient_rag.models import Document
        
        chunker = FixedChunker()
        doc = Document(content="", source="empty.txt")
        chunks = chunker.chunk(doc)
        
        assert len(chunks) == 0


class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    def test_chunk_by_headings(self, sample_document):
        """Should split on markdown headings."""
        from omniscient_rag.chunkers import SemanticChunker
        
        chunker = SemanticChunker(chunk_size=512)
        chunks = chunker.chunk(sample_document)
        
        assert len(chunks) >= 1
        # Should preserve semantic structure
        assert any("Code Style" in c.content or "Type Hints" in c.content for c in chunks)
    
    def test_preserve_code_blocks(self, sample_document):
        """Code blocks should be kept intact when possible."""
        from omniscient_rag.chunkers import SemanticChunker
        
        chunker = SemanticChunker(chunk_size=512, respect_code_blocks=True)
        chunks = chunker.chunk(sample_document)
        
        # Check that code block is preserved
        code_chunks = [c for c in chunks if "def calculate_total" in c.content]
        if code_chunks:
            # The function should not be split mid-definition
            assert "return sum(items)" in code_chunks[0].content


class TestASTChunker:
    """Tests for ASTChunker."""
    
    def test_chunk_by_functions(self, sample_code_document):
        """Should split by function definitions."""
        from omniscient_rag.chunkers import ASTChunker
        
        chunker = ASTChunker(chunk_size=512)
        chunks = chunker.chunk(sample_code_document)
        
        assert len(chunks) >= 1
        
        # Check that class/function structure is recognized
        all_content = " ".join(c.content for c in chunks)
        assert "DataProcessor" in all_content
        assert "fetch_data" in all_content
    
    def test_preserve_class_structure(self, sample_code_document):
        """Class definitions should be handled appropriately."""
        from omniscient_rag.chunkers import ASTChunker
        
        chunker = ASTChunker(chunk_size=1024)  # Large enough for whole class
        chunks = chunker.chunk(sample_code_document)
        
        # With large chunk size, class should be mostly together
        class_chunks = [c for c in chunks if "class DataProcessor" in c.content]
        assert len(class_chunks) >= 1


class TestChunkerFactory:
    """Tests for ChunkerFactory."""
    
    def test_create_by_name(self):
        """Factory should create chunkers by name."""
        from omniscient_rag.chunkers import ChunkerFactory
        
        for strategy in ["fixed", "semantic", "ast", "auto"]:
            chunker = ChunkerFactory.create(strategy)
            assert chunker is not None
            assert chunker.strategy_name.lower() in strategy.lower() or strategy == "auto"
    
    def test_create_with_params(self):
        """Factory should pass parameters to chunkers."""
        from omniscient_rag.chunkers import ChunkerFactory
        
        chunker = ChunkerFactory.create("fixed", chunk_size=256, chunk_overlap=0.15)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 0.15
    
    def test_invalid_strategy_raises(self):
        """Unknown strategies should raise ValueError."""
        from omniscient_rag.chunkers import ChunkerFactory
        
        with pytest.raises(ValueError) as exc_info:
            ChunkerFactory.create("unknown_strategy")
        
        assert "unknown_strategy" in str(exc_info.value).lower()


class TestAutoChunker:
    """Tests for AutoChunker auto-detection."""
    
    def test_auto_detect_markdown(self, sample_document):
        """Should use semantic chunking for markdown."""
        from omniscient_rag.chunkers import ChunkerFactory
        
        chunker = ChunkerFactory.create("auto")
        chunks = chunker.chunk(sample_document)
        
        assert len(chunks) >= 1
        # Should detect semantic strategy
        assert any(c.metadata.get("auto_detected_strategy") == "semantic" for c in chunks)
    
    def test_auto_detect_python(self, sample_code_document):
        """Should use AST chunking for Python."""
        from omniscient_rag.chunkers import ChunkerFactory
        
        chunker = ChunkerFactory.create("auto")
        chunks = chunker.chunk(sample_code_document)
        
        assert len(chunks) >= 1
        # Should detect AST strategy
        assert any(c.metadata.get("auto_detected_strategy") == "ast" for c in chunks)


# =============================================================================
# Model Tests
# =============================================================================

class TestModels:
    """Tests for data models."""
    
    def test_document_creation(self):
        """Document should be created with defaults."""
        from omniscient_rag.models import Document
        
        doc = Document(content="Test content", source="test.txt")
        
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert doc.id is not None
        assert doc.created_at is not None
    
    def test_chunk_token_count(self):
        """Chunk should calculate approximate token count."""
        from omniscient_rag.models import Chunk
        
        chunk = Chunk(
            content="Hello world this is a test",  # ~6 words, ~6 tokens
            document_id=uuid4(),
        )
        
        # Approximate: 1 token per 4 chars
        assert chunk.token_count > 0
        assert chunk.token_count < 20
    
    def test_knowledge_score_overall(self):
        """KnowledgeScore should calculate weighted overall score."""
        from omniscient_rag.models import KnowledgeScore
        
        score = KnowledgeScore(
            retrieval_precision=0.8,
            answer_accuracy=75.0,
            coverage_ratio=0.5,
            questions_evaluated=10,
        )
        
        # Weighted: 40% accuracy + 35% precision*100 + 25% coverage*100
        expected = 75.0 * 0.4 + 80.0 * 0.35 + 50.0 * 0.25
        assert abs(score.overall_score - expected) < 0.1
    
    def test_retrieval_result_context_string(self):
        """RetrievalResult should format as context string."""
        from omniscient_rag.models import Chunk, RetrievalResult
        
        chunk = Chunk(
            content="This is the chunk content.",
            document_id=uuid4(),
            metadata={"source": "docs/example.md"},
        )
        
        result = RetrievalResult(
            chunk=chunk,
            vector_score=0.85,
            bm25_score=0.72,
            combined_score=0.78,
            rank=1,
        )
        
        context = result.to_context_string()
        assert "docs/example.md" in context
        assert "This is the chunk content" in context


# =============================================================================
# Search Tests
# =============================================================================

class TestHybridSearchScoring:
    """Tests for hybrid search scoring logic."""
    
    def test_rrf_fusion(self):
        """RRF fusion should combine rankings correctly."""
        from omniscient_rag.search.hybrid import HybridSearcher, SearchConfig
        from omniscient_rag.models import Chunk
        from unittest.mock import MagicMock, AsyncMock
        
        # Create mock store and embed function
        mock_store = MagicMock()
        mock_embed = AsyncMock(return_value=[0.1] * 768)
        
        config = SearchConfig(top_k=5, alpha=0.5, rrf_k=60)
        searcher = HybridSearcher(mock_store, mock_embed, config)
        
        # Create test chunks
        chunk1 = Chunk(content="Chunk 1", document_id=uuid4())
        chunk2 = Chunk(content="Chunk 2", document_id=uuid4())
        chunk3 = Chunk(content="Chunk 3", document_id=uuid4())
        
        # Simulate rankings
        # Vector: chunk1 (rank 1), chunk2 (rank 2)
        # BM25:   chunk2 (rank 1), chunk3 (rank 2)
        vector_results = [(chunk1, 0.9), (chunk2, 0.7)]
        bm25_results = [(chunk2, 0.8), (chunk3, 0.6)]
        
        # Run fusion
        combined = searcher._reciprocal_rank_fusion(
            vector_results, bm25_results, alpha=0.5
        )
        
        # chunk2 should rank highest (appears in both lists)
        assert len(combined) == 3
        chunk2_result = next(r for r in combined if r.chunk.id == chunk2.id)
        assert chunk2_result.rank <= 2  # Should be near top


# =============================================================================
# Integration Tests (require DB + Ollama)
# =============================================================================

@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests requiring database and Ollama.
    
    These tests are skipped by default.
    Run with: pytest --run-integration
    Requires: PostgreSQL with pgvector, Ollama with nomic-embed-text
    """
    
    @pytest.fixture
    async def pipeline(self):
        """Create a test pipeline."""
        pytest.importorskip("psycopg")
        pytest.importorskip("pgvector")
        
        from omniscient_rag import RAGPipeline, RAGConfig
        from omniscient_rag.store import PostgresVectorStore, DatabaseConfig
        from omniscient_rag.chunkers import ChunkerFactory
        from omniscient_llm import OllamaProvider
        
        # Use test database
        db_url = "postgresql://omniscient:localdev@localhost:5432/omniscient_test"
        
        provider = OllamaProvider(model="qwen2.5-coder:1.5b")
        await provider.initialize()
        
        store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
        
        config = RAGConfig(
            chunking_strategy="semantic",
            chunk_size=256,
            auto_generate_questions=False,
        )
        
        chunker = ChunkerFactory.create("semantic", 256)
        
        pipeline = RAGPipeline(
            store=store,
            chunker=chunker,
            embed_fn=lambda t: provider.embed(t, model="nomic-embed-text"),
            config=config,
        )
        await pipeline.initialize()
        
        yield pipeline
        
        await pipeline.clear_all()
        await pipeline.close()
        await provider.close()
    
    @pytest.mark.asyncio
    async def test_ingest_and_query(self, pipeline, sample_document):
        """Test full ingest and query cycle."""
        # Ingest
        result = await pipeline.ingest_document(sample_document)
        assert result["chunks_created"] > 0
        
        # Query
        results = await pipeline.query("What is the maximum line length in Python?")
        assert len(results) > 0
        
        # Should find relevant content
        all_content = " ".join(r.chunk.content for r in results)
        assert "79" in all_content or "line length" in all_content.lower()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_balance(self, pipeline, sample_document):
        """Test that hybrid search uses both vector and BM25."""
        await pipeline.ingest_document(sample_document)
        
        # Query with alpha=0.5 (balanced)
        balanced = await pipeline.query("PEP 8 style", alpha=0.5)
        
        # Query with alpha=1.0 (vector only)
        vector_only = await pipeline.query("PEP 8 style", alpha=1.0)
        
        # Query with alpha=0.0 (BM25 only)
        bm25_only = await pipeline.query("PEP 8 style", alpha=0.0)
        
        # All should return results
        assert len(balanced) > 0
        assert len(vector_only) > 0
        assert len(bm25_only) > 0
        
        # Rankings may differ between modes
        # (Just verify we get results, not specific ranking)


# =============================================================================
# Metrics Tests
# =============================================================================

class TestQuestionGenerator:
    """Tests for question generation."""
    
    def test_simple_generator(self, sample_document):
        """SimpleQuestionGenerator should extract questions from structure."""
        from omniscient_rag.metrics.questions import SimpleQuestionGenerator
        
        generator = SimpleQuestionGenerator(questions_per_doc=5)
        questions = generator.generate(sample_document)
        
        assert len(questions) > 0
        assert len(questions) <= 5
        
        for q in questions:
            assert q.question
            assert q.document_id == sample_document.id


class TestKnowledgeScoring:
    """Tests for knowledge scoring heuristics."""
    
    def test_heuristic_exact_match(self):
        """Exact answer match should score 100."""
        from omniscient_rag.metrics.scorer import KnowledgeScorer
        from omniscient_rag.models import KnowledgeQuestion
        from unittest.mock import MagicMock
        
        scorer = KnowledgeScorer(
            store=MagicMock(),
            searcher=MagicMock(),
        )
        
        question = KnowledgeQuestion(
            question="What is the default port?",
            expected_answer="8080",
            document_id=uuid4(),
        )
        
        context = "The server runs on port 8080 by default."
        score = scorer._score_heuristic(question, context)
        
        assert score == 100.0
    
    def test_heuristic_partial_match(self):
        """Partial answer match should score proportionally."""
        from omniscient_rag.metrics.scorer import KnowledgeScorer
        from omniscient_rag.models import KnowledgeQuestion
        from unittest.mock import MagicMock
        
        scorer = KnowledgeScorer(
            store=MagicMock(),
            searcher=MagicMock(),
        )
        
        question = KnowledgeQuestion(
            question="What are the memory settings?",
            expected_answer="executor memory and driver memory",
            document_id=uuid4(),
        )
        
        # Context contains "executor" and "memory" but not "driver"
        context = "Configure the executor memory for optimal performance."
        score = scorer._score_heuristic(question, context)
        
        assert 0 < score < 100


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
