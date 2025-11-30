"""
Comprehensive End-to-End RAG System Test & Validation

This script tests all RAG features:
1. Document ingestion with different chunking strategies
2. Embedding generation and storage
3. Hybrid search (vector + BM25)
4. Knowledge persistence and learning
5. Performance benchmarking

Run with:
    python scripts/test_e2e_rag.py
"""

import asyncio
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from uuid import uuid4

# Add package paths
project_root = Path(__file__).parent.parent
for pkg in ["core", "llm", "rag"]:
    src_path = project_root / "packages" / pkg / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))


@dataclass
class BenchmarkResult:
    """Result from a benchmark operation."""
    operation: str
    duration_ms: float
    items_processed: int = 1
    success: bool = True
    error: str = ""
    
    @property
    def throughput(self) -> float:
        """Items per second."""
        if self.duration_ms == 0:
            return 0
        return (self.items_processed / self.duration_ms) * 1000


@dataclass 
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.results:
            return {}
        
        durations = [r.duration_ms for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        return {
            "total_tests": len(self.results),
            "passed": len(self.results) - len(failed),
            "failed": len(failed),
            "total_duration_ms": sum(durations),
            "avg_duration_ms": statistics.mean(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
        }


# Sample test documents
TEST_DOCUMENTS = [
    {
        "source": "spark_memory.md",
        "content": """# Apache Spark Memory Management

## Overview
Spark uses both on-heap and off-heap memory for execution and storage.

## Executor Memory
The `spark.executor.memory` setting controls the heap size per executor.
Default value is 1g. Recommended: 4-8g for most workloads.

```python
spark = SparkSession.builder \\
    .config("spark.executor.memory", "4g") \\
    .config("spark.executor.memoryOverhead", "1g") \\
    .getOrCreate()
```

## Memory Fractions
- spark.memory.fraction: 0.6 (default)
- spark.memory.storageFraction: 0.5 (default)

## Tuning Tips
1. Monitor memory usage via Spark UI
2. Increase memoryOverhead for Python/R workloads
3. Use off-heap memory for large datasets
"""
    },
    {
        "source": "spark_shuffle.md", 
        "content": """# Spark Shuffle Optimization

## What is Shuffle?
Shuffle redistributes data across partitions during wide transformations
like groupBy, join, and repartition.

## Key Settings
- spark.sql.shuffle.partitions: 200 (default)
- spark.shuffle.compress: true
- spark.shuffle.spill.compress: true

## Optimizing Shuffle
### Reduce Shuffle Size
```python
# Use broadcast for small tables
from pyspark.sql.functions import broadcast
df.join(broadcast(small_df), "key")
```

### Partition Tuning
Aim for 128MB-200MB per partition.
Too many partitions = overhead
Too few = memory pressure

## Common Issues
- Skewed data causing OOM
- Excessive network I/O
- Disk spill slowdowns
"""
    },
    {
        "source": "kafka_consumers.md",
        "content": """# Kafka Consumer Configuration

## Consumer Groups
Consumers in the same group share partitions.
Each partition is consumed by exactly one consumer in a group.

## Key Configurations
```properties
bootstrap.servers=localhost:9092
group.id=my-consumer-group
auto.offset.reset=earliest
enable.auto.commit=false
max.poll.records=500
```

## Offset Management
- auto.commit: Simple but may lose messages
- manual commit: More control, use commitSync() or commitAsync()

## Performance Tuning
1. Increase max.poll.records for throughput
2. Use multiple consumers per group
3. Monitor consumer lag
4. Tune fetch.min.bytes and fetch.max.wait.ms
"""
    },
    {
        "source": "python_async.py",
        "content": '''"""
Async Python Patterns

This module demonstrates async/await patterns.
"""

import asyncio
from typing import List, Any

class AsyncDataProcessor:
    """Process data asynchronously."""
    
    def __init__(self, concurrency: int = 10):
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
    
    async def process_item(self, item: Any) -> Any:
        """Process a single item."""
        async with self.semaphore:
            await asyncio.sleep(0.1)  # Simulate I/O
            return item * 2
    
    async def process_batch(self, items: List[Any]) -> List[Any]:
        """Process items in parallel with limited concurrency."""
        tasks = [self.process_item(item) for item in items]
        return await asyncio.gather(*tasks)


async def main():
    processor = AsyncDataProcessor(concurrency=5)
    results = await processor.process_batch(range(100))
    print(f"Processed {len(results)} items")


if __name__ == "__main__":
    asyncio.run(main())
'''
    },
]

# Ground truth Q&A for validation
GROUND_TRUTH_QA = [
    {
        "question": "What is the default Spark executor memory?",
        "expected_terms": ["1g", "default"],
        "expected_source": "spark_memory.md",
    },
    {
        "question": "How do I broadcast a small table in Spark?",
        "expected_terms": ["broadcast", "join", "small"],
        "expected_source": "spark_shuffle.md",
    },
    {
        "question": "What is the default number of shuffle partitions?",
        "expected_terms": ["200", "partitions"],
        "expected_source": "spark_shuffle.md",
    },
    {
        "question": "How does Kafka consumer group work?",
        "expected_terms": ["partition", "consumer", "group"],
        "expected_source": "kafka_consumers.md",
    },
    {
        "question": "How to limit async concurrency in Python?",
        "expected_terms": ["semaphore", "concurrency"],
        "expected_source": "python_async.py",
    },
]


class E2ETestRunner:
    """End-to-end test runner for RAG system."""
    
    def __init__(self, db_url: str = "postgresql://omniscient:localdev@localhost:5432/omniscient"):
        self.db_url = db_url
        self.store = None
        self.provider = None
        self.memory = None
        self.suites: Dict[str, TestSuite] = {}
        self.bottlenecks: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize all components."""
        from omniscient_rag.store import PostgresVectorStore, DatabaseConfig
        from omniscient_rag.learning import KnowledgeMemory
        from omniscient_llm import OllamaProvider
        
        print("🔧 Initializing test environment...")
        
        # Initialize store
        start = time.perf_counter()
        self.store = PostgresVectorStore(DatabaseConfig(connection_string=self.db_url))
        await self.store.initialize()
        init_time = (time.perf_counter() - start) * 1000
        print(f"  ✓ PostgreSQL store initialized ({init_time:.1f}ms)")
        
        # Initialize Ollama
        start = time.perf_counter()
        self.provider = OllamaProvider(model="qwen2.5-coder:1.5b")
        await self.provider.initialize()
        init_time = (time.perf_counter() - start) * 1000
        print(f"  ✓ Ollama provider initialized ({init_time:.1f}ms)")
        
        # Initialize learning
        start = time.perf_counter()
        self.memory = KnowledgeMemory(self.store, self._embed)
        await self.memory.initialize()
        init_time = (time.perf_counter() - start) * 1000
        print(f"  ✓ Knowledge memory initialized ({init_time:.1f}ms)")
        
        print()
    
    async def _embed(self, text: str) -> List[float]:
        """Generate embedding."""
        return await self.provider.embed(text, model="nomic-embed-text")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.provider:
            await self.provider.close()
        if self.store:
            await self.store.close()
    
    def _record(self, suite_name: str, result: BenchmarkResult):
        """Record a benchmark result."""
        if suite_name not in self.suites:
            self.suites[suite_name] = TestSuite(name=suite_name)
        self.suites[suite_name].add(result)
        
        # Track potential bottleneck
        if result.duration_ms > 1000:  # > 1 second
            self.bottlenecks.append({
                "suite": suite_name,
                "operation": result.operation,
                "duration_ms": result.duration_ms,
            })
    
    # =========================================================================
    # Test Suites
    # =========================================================================
    
    async def test_embedding_generation(self) -> TestSuite:
        """Test embedding generation performance."""
        print("📊 Testing Embedding Generation...")
        suite = TestSuite(name="embedding")
        
        test_texts = [
            "Short text",
            "Medium length text that contains more words and context for embedding",
            "A much longer piece of text that simulates a realistic document chunk. " * 10,
        ]
        
        for i, text in enumerate(test_texts):
            start = time.perf_counter()
            try:
                embedding = await self._embed(text)
                duration = (time.perf_counter() - start) * 1000
                
                # Validate embedding
                assert len(embedding) == 768, f"Expected 768 dims, got {len(embedding)}"
                assert all(isinstance(x, float) for x in embedding)
                
                result = BenchmarkResult(
                    operation=f"embed_{len(text)}_chars",
                    duration_ms=duration,
                    items_processed=1,
                )
                print(f"  ✓ Embedded {len(text)} chars in {duration:.1f}ms")
            except Exception as e:
                result = BenchmarkResult(
                    operation=f"embed_{len(text)}_chars",
                    duration_ms=0,
                    success=False,
                    error=str(e),
                )
                print(f"  ✗ Failed: {e}")
            
            suite.add(result)
            self._record("embedding", result)
        
        # Batch embedding test
        start = time.perf_counter()
        try:
            batch_texts = ["Text number " + str(i) for i in range(10)]
            embeddings = await self.provider.embed_batch(batch_texts, model="nomic-embed-text")
            duration = (time.perf_counter() - start) * 1000
            
            result = BenchmarkResult(
                operation="batch_embed_10",
                duration_ms=duration,
                items_processed=10,
            )
            print(f"  ✓ Batch embedded 10 texts in {duration:.1f}ms ({result.throughput:.1f}/sec)")
        except Exception as e:
            result = BenchmarkResult(
                operation="batch_embed_10",
                duration_ms=0,
                success=False,
                error=str(e),
            )
            print(f"  ✗ Batch embed failed: {e}")
        
        suite.add(result)
        self._record("embedding", result)
        
        print()
        return suite
    
    async def test_chunking_strategies(self) -> TestSuite:
        """Test different chunking strategies."""
        print("📊 Testing Chunking Strategies...")
        suite = TestSuite(name="chunking")
        
        from omniscient_rag.chunkers import ChunkerFactory
        from omniscient_rag.models import Document
        
        strategies = ["fixed", "semantic", "ast", "auto"]
        
        for doc_data in TEST_DOCUMENTS[:2]:  # Test with 2 docs
            doc = Document(
                content=doc_data["content"],
                source=doc_data["source"],
            )
            
            for strategy in strategies:
                start = time.perf_counter()
                try:
                    chunker = ChunkerFactory.create(strategy, chunk_size=256)
                    chunks = chunker.chunk(doc)
                    duration = (time.perf_counter() - start) * 1000
                    
                    result = BenchmarkResult(
                        operation=f"{strategy}_{doc.source}",
                        duration_ms=duration,
                        items_processed=len(chunks),
                    )
                    print(f"  ✓ {strategy:8} on {doc.source}: {len(chunks)} chunks in {duration:.1f}ms")
                except Exception as e:
                    result = BenchmarkResult(
                        operation=f"{strategy}_{doc.source}",
                        duration_ms=0,
                        success=False,
                        error=str(e),
                    )
                    print(f"  ✗ {strategy} on {doc.source}: {e}")
                
                suite.add(result)
                self._record("chunking", result)
        
        print()
        return suite
    
    async def test_document_ingestion(self) -> TestSuite:
        """Test document ingestion pipeline."""
        print("📊 Testing Document Ingestion...")
        suite = TestSuite(name="ingestion")
        
        from omniscient_rag.models import Document
        from omniscient_rag.chunkers import ChunkerFactory
        
        # Clear existing data
        await self.store.clear_all()
        
        chunker = ChunkerFactory.create("semantic", chunk_size=256)
        
        for doc_data in TEST_DOCUMENTS:
            doc = Document(
                content=doc_data["content"],
                source=doc_data["source"],
            )
            
            # Chunk
            start = time.perf_counter()
            chunks = chunker.chunk(doc)
            chunk_time = (time.perf_counter() - start) * 1000
            
            # Embed each chunk
            embed_start = time.perf_counter()
            for chunk in chunks:
                chunk.embedding = await self._embed(chunk.content)
            embed_time = (time.perf_counter() - embed_start) * 1000
            
            # Store document
            store_start = time.perf_counter()
            await self.store.insert_document(doc)
            doc_store_time = (time.perf_counter() - store_start) * 1000
            
            # Store chunks
            chunk_store_start = time.perf_counter()
            await self.store.insert_chunks(chunks)
            chunk_store_time = (time.perf_counter() - chunk_store_start) * 1000
            
            total_time = (time.perf_counter() - start) * 1000
            
            result = BenchmarkResult(
                operation=f"ingest_{doc.source}",
                duration_ms=total_time,
                items_processed=len(chunks),
            )
            
            print(f"  ✓ {doc.source}: {len(chunks)} chunks")
            print(f"    → Chunk: {chunk_time:.1f}ms, Embed: {embed_time:.1f}ms, Store: {doc_store_time + chunk_store_time:.1f}ms")
            
            suite.add(result)
            self._record("ingestion", result)
            
            # Track embedding as potential bottleneck
            if embed_time > 500:
                self.bottlenecks.append({
                    "suite": "ingestion",
                    "operation": f"embedding_{doc.source}",
                    "duration_ms": embed_time,
                    "recommendation": "Consider batch embedding or caching",
                })
        
        print()
        return suite
    
    async def test_vector_search(self) -> TestSuite:
        """Test vector similarity search."""
        print("📊 Testing Vector Search...")
        suite = TestSuite(name="vector_search")
        
        queries = [q["question"] for q in GROUND_TRUTH_QA]
        
        for query in queries:
            # Embed query
            start = time.perf_counter()
            query_embedding = await self._embed(query)
            embed_time = (time.perf_counter() - start) * 1000
            
            # Search
            search_start = time.perf_counter()
            results = await self.store.search_vectors(query_embedding, top_k=5)
            search_time = (time.perf_counter() - search_start) * 1000
            
            total_time = embed_time + search_time
            
            result = BenchmarkResult(
                operation=f"vector_search",
                duration_ms=total_time,
                items_processed=len(results),
            )
            
            print(f"  ✓ '{query[:40]}...'")
            print(f"    → Embed: {embed_time:.1f}ms, Search: {search_time:.1f}ms, Results: {len(results)}")
            
            suite.add(result)
            self._record("vector_search", result)
        
        print()
        return suite
    
    async def test_fulltext_search(self) -> TestSuite:
        """Test full-text (BM25) search."""
        print("📊 Testing Full-Text Search...")
        suite = TestSuite(name="fulltext_search")
        
        queries = [q["question"] for q in GROUND_TRUTH_QA]
        
        for query in queries:
            start = time.perf_counter()
            try:
                results = await self.store.search_fulltext(query, top_k=5)
                duration = (time.perf_counter() - start) * 1000
                
                result = BenchmarkResult(
                    operation="fulltext_search",
                    duration_ms=duration,
                    items_processed=len(results),
                )
                print(f"  ✓ '{query[:40]}...' → {len(results)} results in {duration:.1f}ms")
            except Exception as e:
                result = BenchmarkResult(
                    operation="fulltext_search",
                    duration_ms=0,
                    success=False,
                    error=str(e),
                )
                print(f"  ✗ '{query[:40]}...': {e}")
            
            suite.add(result)
            self._record("fulltext_search", result)
        
        print()
        return suite
    
    async def test_hybrid_search(self) -> TestSuite:
        """Test hybrid search with RRF."""
        print("📊 Testing Hybrid Search (Vector + BM25)...")
        suite = TestSuite(name="hybrid_search")
        
        from omniscient_rag.search.hybrid import HybridSearcher, SearchConfig
        
        config = SearchConfig(top_k=5, alpha=0.5)
        searcher = HybridSearcher(self.store, self._embed, config)
        
        for qa in GROUND_TRUTH_QA:
            query = qa["question"]
            expected_source = qa["expected_source"]
            expected_terms = qa["expected_terms"]
            
            start = time.perf_counter()
            try:
                results = await searcher.search(query, top_k=5)
                duration = (time.perf_counter() - start) * 1000
                
                # Validate results
                found_source = False
                found_terms = 0
                
                for r in results:
                    source = r.chunk.metadata.get("source", r.chunk.metadata.get("document_source", ""))
                    if expected_source in str(source) or expected_source in r.chunk.content:
                        found_source = True
                    
                    content_lower = r.chunk.content.lower()
                    for term in expected_terms:
                        if term.lower() in content_lower:
                            found_terms += 1
                
                precision = found_terms / len(expected_terms) if expected_terms else 0
                
                result = BenchmarkResult(
                    operation=f"hybrid_search",
                    duration_ms=duration,
                    items_processed=len(results),
                )
                
                status = "✓" if found_source or precision > 0.5 else "⚠"
                print(f"  {status} '{query[:40]}...'")
                print(f"    → {duration:.1f}ms, {len(results)} results, precision: {precision:.0%}")
                
            except Exception as e:
                result = BenchmarkResult(
                    operation="hybrid_search",
                    duration_ms=0,
                    success=False,
                    error=str(e),
                )
                print(f"  ✗ '{query[:40]}...': {e}")
            
            suite.add(result)
            self._record("hybrid_search", result)
        
        print()
        return suite
    
    async def test_knowledge_learning(self) -> TestSuite:
        """Test knowledge learning and retrieval."""
        print("📊 Testing Knowledge Learning...")
        suite = TestSuite(name="learning")
        
        from omniscient_rag.learning import (
            LearnedFact, ReasoningChain, FeedbackLearner, FeedbackType
        )
        
        learner = FeedbackLearner(self.memory)
        
        # Test fact storage
        start = time.perf_counter()
        fact = LearnedFact(
            topic="Spark",
            fact="Default executor memory is 1g",
            source_question="What is default executor memory?",
            source_answer="The default is 1g",
            confidence=0.9,
        )
        await self.memory.store_fact(fact)
        duration = (time.perf_counter() - start) * 1000
        
        result = BenchmarkResult(operation="store_fact", duration_ms=duration)
        print(f"  ✓ Stored fact in {duration:.1f}ms")
        suite.add(result)
        self._record("learning", result)
        
        # Test fact retrieval
        start = time.perf_counter()
        facts = await self.memory.get_relevant_facts("Spark memory configuration", top_k=5)
        duration = (time.perf_counter() - start) * 1000
        
        result = BenchmarkResult(
            operation="retrieve_facts",
            duration_ms=duration,
            items_processed=len(facts),
        )
        print(f"  ✓ Retrieved {len(facts)} facts in {duration:.1f}ms")
        suite.add(result)
        self._record("learning", result)
        
        # Test reasoning chain storage
        start = time.perf_counter()
        chain = ReasoningChain(
            question="How to optimize Spark shuffle?",
            steps=["Identify shuffle operations", "Check partition count", "Tune settings"],
            final_answer="Reduce shuffle partitions and use broadcast joins",
            was_correct=True,
            feedback_score=0.9,
        )
        await self.memory.store_reasoning_chain(chain)
        duration = (time.perf_counter() - start) * 1000
        
        result = BenchmarkResult(operation="store_chain", duration_ms=duration)
        print(f"  ✓ Stored reasoning chain in {duration:.1f}ms")
        suite.add(result)
        self._record("learning", result)
        
        # Test feedback processing
        start = time.perf_counter()
        learned = await learner.process_feedback(
            question="How do I set Spark memory?",
            answer="Use spark.executor.memory setting",
            rating=0.8,
            feedback_type=FeedbackType.POSITIVE,
        )
        duration = (time.perf_counter() - start) * 1000
        
        result = BenchmarkResult(
            operation="process_feedback",
            duration_ms=duration,
            items_processed=learned.get("facts_extracted", 0),
        )
        print(f"  ✓ Processed feedback in {duration:.1f}ms (extracted {learned.get('facts_extracted', 0)} facts)")
        suite.add(result)
        self._record("learning", result)
        
        print()
        return suite
    
    async def test_context_injection(self) -> TestSuite:
        """Test context injection for prompts."""
        print("📊 Testing Context Injection...")
        suite = TestSuite(name="context_injection")
        
        from omniscient_rag.learning import ContextInjector, InjectionConfig
        
        config = InjectionConfig(max_facts=5, max_reasoning_chains=2)
        injector = ContextInjector(self.memory, config)
        
        queries = [
            "How do I configure Spark executor memory?",
            "What are Kafka consumer best practices?",
            "How to write async Python code?",
        ]
        
        for query in queries:
            start = time.perf_counter()
            try:
                context = await injector.build_context(query)
                duration = (time.perf_counter() - start) * 1000
                
                result = BenchmarkResult(
                    operation="build_context",
                    duration_ms=duration,
                    items_processed=len(context) if context else 0,
                )
                print(f"  ✓ '{query[:40]}...' → {len(context)} chars in {duration:.1f}ms")
            except Exception as e:
                result = BenchmarkResult(
                    operation="build_context",
                    duration_ms=0,
                    success=False,
                    error=str(e),
                )
                print(f"  ✗ '{query[:40]}...': {e}")
            
            suite.add(result)
            self._record("context_injection", result)
        
        print()
        return suite
    
    async def test_concurrent_operations(self) -> TestSuite:
        """Test concurrent search operations."""
        print("📊 Testing Concurrent Operations...")
        suite = TestSuite(name="concurrency")
        
        from omniscient_rag.search.hybrid import HybridSearcher, SearchConfig
        
        config = SearchConfig(top_k=3, alpha=0.5)
        searcher = HybridSearcher(self.store, self._embed, config)
        
        queries = [q["question"] for q in GROUND_TRUTH_QA]
        
        # Sequential baseline
        start = time.perf_counter()
        for query in queries:
            await searcher.search(query, top_k=3)
        sequential_time = (time.perf_counter() - start) * 1000
        
        result = BenchmarkResult(
            operation="sequential_5_queries",
            duration_ms=sequential_time,
            items_processed=len(queries),
        )
        print(f"  ✓ Sequential 5 queries: {sequential_time:.1f}ms ({result.throughput:.1f}/sec)")
        suite.add(result)
        self._record("concurrency", result)
        
        # Concurrent
        start = time.perf_counter()
        tasks = [searcher.search(q, top_k=3) for q in queries]
        await asyncio.gather(*tasks)
        concurrent_time = (time.perf_counter() - start) * 1000
        
        result = BenchmarkResult(
            operation="concurrent_5_queries",
            duration_ms=concurrent_time,
            items_processed=len(queries),
        )
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
        print(f"  ✓ Concurrent 5 queries: {concurrent_time:.1f}ms ({result.throughput:.1f}/sec, {speedup:.1f}x speedup)")
        suite.add(result)
        self._record("concurrency", result)
        
        print()
        return suite
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def print_summary(self):
        """Print test summary."""
        print("=" * 70)
        print("📋 TEST SUMMARY")
        print("=" * 70)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for name, suite in self.suites.items():
            summary = suite.summary()
            total_tests += summary.get("total_tests", 0)
            total_passed += summary.get("passed", 0)
            total_failed += summary.get("failed", 0)
            
            status = "✓" if summary.get("failed", 0) == 0 else "✗"
            print(f"\n{status} {name}:")
            print(f"   Tests: {summary.get('total_tests', 0)} ({summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed)")
            print(f"   Avg Duration: {summary.get('avg_duration_ms', 0):.1f}ms")
            print(f"   Total Time: {summary.get('total_duration_ms', 0):.1f}ms")
        
        print("\n" + "-" * 70)
        print(f"TOTAL: {total_tests} tests ({total_passed} passed, {total_failed} failed)")
        
        # Bottlenecks
        if self.bottlenecks:
            print("\n" + "=" * 70)
            print("⚠️  BOTTLENECKS DETECTED")
            print("=" * 70)
            
            # Sort by duration
            sorted_bottlenecks = sorted(self.bottlenecks, key=lambda x: x["duration_ms"], reverse=True)
            
            for b in sorted_bottlenecks[:5]:
                print(f"\n  🔴 {b['suite']}/{b['operation']}: {b['duration_ms']:.0f}ms")
                if "recommendation" in b:
                    print(f"     → {b['recommendation']}")
        
        print()
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on test results."""
        recommendations = []
        
        # Analyze embedding performance
        if "embedding" in self.suites:
            embed_summary = self.suites["embedding"].summary()
            avg_embed = embed_summary.get("avg_duration_ms", 0)
            if avg_embed > 100:
                recommendations.append(
                    f"⚡ Embedding is slow ({avg_embed:.0f}ms avg). Consider:\n"
                    "   - Use batch embedding (embed_batch) instead of individual calls\n"
                    "   - Cache embeddings for repeated queries\n"
                    "   - Use a faster/smaller embedding model"
                )
        
        # Analyze search performance
        if "hybrid_search" in self.suites:
            search_summary = self.suites["hybrid_search"].summary()
            avg_search = search_summary.get("avg_duration_ms", 0)
            if avg_search > 500:
                recommendations.append(
                    f"⚡ Hybrid search is slow ({avg_search:.0f}ms avg). Consider:\n"
                    "   - Create proper indexes on embedding column\n"
                    "   - Reduce top_k for initial retrieval\n"
                    "   - Pre-filter by document IDs if possible"
                )
        
        # Analyze concurrency
        if "concurrency" in self.suites:
            results = self.suites["concurrency"].results
            if len(results) >= 2:
                seq = results[0].duration_ms
                conc = results[1].duration_ms
                if seq > 0 and conc > 0:
                    speedup = seq / conc
                    if speedup < 1.5:
                        recommendations.append(
                            f"⚡ Low concurrency benefit ({speedup:.1f}x). Consider:\n"
                            "   - Increase connection pool size\n"
                            "   - Use connection pooling for Ollama\n"
                            "   - Check for shared resource bottlenecks"
                        )
        
        # Analyze ingestion
        if "ingestion" in self.suites:
            ing_summary = self.suites["ingestion"].summary()
            avg_ing = ing_summary.get("avg_duration_ms", 0)
            if avg_ing > 2000:
                recommendations.append(
                    f"⚡ Ingestion is slow ({avg_ing:.0f}ms avg). Consider:\n"
                    "   - Batch insert chunks instead of one-by-one\n"
                    "   - Use COPY command for bulk inserts\n"
                    "   - Pre-generate embeddings in parallel"
                )
        
        return recommendations


async def main():
    """Run all tests."""
    print("=" * 70)
    print("🧪 RAG SYSTEM END-TO-END TEST & VALIDATION")
    print("=" * 70)
    print()
    
    runner = E2ETestRunner()
    
    try:
        await runner.initialize()
        
        # Run all test suites
        await runner.test_embedding_generation()
        await runner.test_chunking_strategies()
        await runner.test_document_ingestion()
        await runner.test_vector_search()
        await runner.test_fulltext_search()
        await runner.test_hybrid_search()
        await runner.test_knowledge_learning()
        await runner.test_context_injection()
        await runner.test_concurrent_operations()
        
        # Print summary
        runner.print_summary()
        
        # Get recommendations
        recommendations = runner.get_optimization_recommendations()
        if recommendations:
            print("=" * 70)
            print("💡 OPTIMIZATION RECOMMENDATIONS")
            print("=" * 70)
            for rec in recommendations:
                print(f"\n{rec}")
        
        print("\n" + "=" * 70)
        print("✅ End-to-end testing complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
