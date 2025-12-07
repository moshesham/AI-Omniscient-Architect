"""
RAG Knowledge Improvement Testing Script

This script tests the RAG system's ability to:
1. Ingest documentation and generate embeddings
2. Create test questions from content
3. Measure retrieval precision and answer quality
4. Track knowledge improvement over multiple iterations

Usage:
    python scripts/test_rag_improvement.py --content-dir ./test_docs
    python scripts/test_rag_improvement.py --use-sample  # Uses built-in sample content
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "llm", "rag"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): sys.path.insert(0, str(_path))

# Sample documentation for testing (Spark configuration example)
SAMPLE_SPARK_DOCS = """
# Apache Spark Configuration Guide

## Executor Configuration

### Memory Settings

Spark executors are the worker processes that run tasks. Proper memory configuration is critical for performance.

**spark.executor.memory**: Sets the amount of memory per executor. Default is 1g. 
For data-intensive workloads, use 4g-8g. Maximum depends on your cluster's node memory.

```python
spark = SparkSession.builder \\
    .config("spark.executor.memory", "8g") \\
    .config("spark.executor.memoryOverhead", "1g") \\
    .getOrCreate()
```

**spark.executor.memoryOverhead**: Additional memory for JVM overhead, interned strings, and other native overheads. 
Default is max(384MB, 0.10 * spark.executor.memory). Increase for PySpark or R workloads.

### Core Settings

**spark.executor.cores**: Number of cores per executor. Default is 1 on YARN, all available on standalone.
Best practice: Use 4-5 cores per executor to balance parallelism and memory overhead.

**spark.executor.instances**: Number of executors to launch. With dynamic allocation disabled, 
this sets a fixed number. Calculate based on: (total_cores - driver_cores) / executor_cores.

## Driver Configuration

**spark.driver.memory**: Memory for the driver process. Default is 1g.
Increase for operations that collect data to the driver (collect(), toPandas()).

**spark.driver.cores**: Number of cores for the driver. Default is 1.

## Shuffle Configuration

**spark.sql.shuffle.partitions**: Number of partitions for shuffles. Default is 200.
For large datasets: set to 2-3x the number of executor cores.
For small datasets: reduce to avoid overhead (e.g., 20-50).

**spark.shuffle.compress**: Whether to compress shuffle output. Default is true.
Uses spark.io.compression.codec (default: lz4).

## Dynamic Allocation

Enable dynamic allocation to automatically scale executors based on workload:

```python
spark = SparkSession.builder \\
    .config("spark.dynamicAllocation.enabled", "true") \\
    .config("spark.dynamicAllocation.minExecutors", "2") \\
    .config("spark.dynamicAllocation.maxExecutors", "20") \\
    .config("spark.dynamicAllocation.executorIdleTimeout", "60s") \\
    .getOrCreate()
```

## Performance Tuning Tips

1. **Broadcast joins**: For small tables (<10MB), use broadcast to avoid shuffle.
2. **Caching**: Use `.cache()` or `.persist()` for reused DataFrames.
3. **Partitioning**: Repartition data by join keys to minimize shuffle.
4. **Serialization**: Use Kryo serialization for better performance.

```python
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```
"""

SAMPLE_SPARK_ADVANCED = """
# Advanced Spark Optimization

## Memory Management Deep Dive

### Unified Memory Model (Spark 1.6+)

Spark uses a unified memory model where execution and storage share the same region:

- **Execution Memory**: Used for shuffles, joins, sorts, aggregations
- **Storage Memory**: Used for caching and propagating internal data

**spark.memory.fraction**: Fraction of heap for execution and storage (default: 0.6).
The remaining 0.4 is for user data structures and internal metadata.

**spark.memory.storageFraction**: Fraction of unified memory for storage (default: 0.5).
Execution can borrow from storage if storage is idle.

### Off-Heap Memory

For large datasets, off-heap memory reduces GC overhead:

```python
spark = SparkSession.builder \\
    .config("spark.memory.offHeap.enabled", "true") \\
    .config("spark.memory.offHeap.size", "4g") \\
    .getOrCreate()
```

## Query Optimization

### Adaptive Query Execution (AQE)

Spark 3.0+ includes AQE for runtime query optimization:

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

**Coalesce Partitions**: Automatically reduces partition count after shuffle.
**Skew Join**: Splits skewed partitions to balance work across executors.

### Join Strategies

1. **Broadcast Hash Join**: Best for small tables (< broadcast threshold)
2. **Sort Merge Join**: Default for large tables, requires sorting
3. **Shuffle Hash Join**: Alternative when one side fits in memory

Control with hints:
```python
df1.join(df2.hint("broadcast"), "key")  # Force broadcast
df1.join(df2.hint("merge"), "key")      # Force sort-merge
```

## Monitoring and Debugging

### Spark UI Metrics

Key metrics to monitor:
- **Shuffle Read/Write**: High values indicate potential optimization opportunities
- **Task Duration Distribution**: Skew shows uneven data distribution  
- **GC Time**: High GC indicates memory pressure

### Event Logging

Enable event logging for post-hoc analysis:
```python
spark.conf.set("spark.eventLog.enabled", "true")
spark.conf.set("spark.eventLog.dir", "hdfs:///spark-logs")
```
"""

# Ground truth Q&A for evaluation
GROUND_TRUTH_QA = [
    {
        "question": "What is the default value for spark.executor.memory?",
        "answer": "1g",
        "topic": "executor_config",
        "difficulty": "easy"
    },
    {
        "question": "How many cores per executor is recommended as best practice?",
        "answer": "4-5 cores",
        "topic": "executor_config",
        "difficulty": "medium"
    },
    {
        "question": "What is the default value for spark.sql.shuffle.partitions?",
        "answer": "200",
        "topic": "shuffle",
        "difficulty": "easy"
    },
    {
        "question": "What is spark.memory.fraction default value?",
        "answer": "0.6",
        "topic": "memory",
        "difficulty": "medium"
    },
    {
        "question": "What configuration enables Adaptive Query Execution?",
        "answer": "spark.sql.adaptive.enabled",
        "topic": "optimization",
        "difficulty": "medium"
    },
    {
        "question": "What serializer is recommended for better Spark performance?",
        "answer": "Kryo serialization or KryoSerializer",
        "topic": "performance",
        "difficulty": "medium"
    },
    {
        "question": "What is the size threshold for using broadcast joins?",
        "answer": "10MB or less",
        "topic": "joins",
        "difficulty": "medium"
    },
    {
        "question": "What configuration enables off-heap memory in Spark?",
        "answer": "spark.memory.offHeap.enabled",
        "topic": "memory",
        "difficulty": "hard"
    },
]


@dataclass
class TestResult:
    """Result from a single test iteration."""
    iteration: int
    timestamp: datetime
    documents_ingested: int
    chunks_created: int
    questions_tested: int
    retrieval_precision: float
    answer_accuracy: float
    coverage_ratio: float
    overall_score: float
    per_question_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass  
class ImprovementReport:
    """Report showing knowledge improvement over iterations."""
    total_iterations: int
    initial_score: float
    final_score: float
    improvement_percentage: float
    results: List[TestResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "improvement_percentage": self.improvement_percentage,
            "results": [
                {
                    "iteration": r.iteration,
                    "overall_score": r.overall_score,
                    "retrieval_precision": r.retrieval_precision,
                    "answer_accuracy": r.answer_accuracy,
                }
                for r in self.results
            ]
        }


class RAGImprovementTester:
    """Tests RAG system knowledge improvement over multiple iterations."""
    
    def __init__(
        self,
        db_url: str = "postgresql://omniscient:localdev@localhost:5432/omniscient",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "qwen2.5-coder:1.5b",
    ):
        self.db_url = db_url
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.provider = None
        self.pipeline = None
        self.results: List[TestResult] = []
    
    async def initialize(self):
        """Initialize connections."""
        from omniscient_llm import OllamaProvider
        from omniscient_rag import RAGPipeline, RAGConfig
        from omniscient_rag.store import PostgresVectorStore, DatabaseConfig
        from omniscient_rag.chunkers import ChunkerFactory
        
        print("🔧 Initializing RAG test environment...")
        
        # Initialize Ollama provider
        self.provider = OllamaProvider(
            model=self.llm_model,
            base_url=self.ollama_url,
        )
        await self.provider.initialize()
        
        # Check if embedding model is available
        print(f"  → Checking embedding model: {self.embedding_model}")
        try:
            test_embed = await self.provider.embed("test", model=self.embedding_model)
            print(f"  ✓ Embedding model ready (dim={len(test_embed)})")
        except Exception as e:
            print(f"  ✗ Embedding model not available: {e}")
            print(f"  → Pulling {self.embedding_model}...")
            await self.provider.pull_model(self.embedding_model)
        
        # Initialize store
        store = PostgresVectorStore(DatabaseConfig(connection_string=self.db_url))
        await store.initialize()
        
        # Create pipeline
        config = RAGConfig(
            chunking_strategy="semantic",
            chunk_size=512,
            chunk_overlap=0.1,
            embedding_model=self.embedding_model,
            top_k=5,
            hybrid_alpha=0.5,
            auto_generate_questions=False,  # We'll use ground truth
        )
        
        chunker = ChunkerFactory.create("semantic", 512, 0.1)
        
        self.pipeline = RAGPipeline(
            store=store,
            chunker=chunker,
            embed_fn=lambda text: self.provider.embed(text, model=self.embedding_model),
            llm_fn=self._llm_generate,
            config=config,
        )
        await self.pipeline.initialize()
        
        print("  ✓ RAG pipeline initialized")
    
    async def _llm_generate(self, prompt: str) -> str:
        """Generate text using LLM."""
        from omniscient_llm.models import GenerationRequest
        response = await self.provider.generate(GenerationRequest(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3,
        ))
        return response.content
    
    async def cleanup(self):
        """Clean up resources."""
        if self.pipeline:
            await self.pipeline.close()
        if self.provider:
            await self.provider.close()
    
    async def clear_knowledge_base(self):
        """Clear all existing data."""
        print("🗑️  Clearing existing knowledge base...")
        await self.pipeline.clear_all()
        print("  ✓ Knowledge base cleared")
    
    async def ingest_content(self, content: str, source: str) -> Dict[str, Any]:
        """Ingest content into the knowledge base."""
        from omniscient_rag.models import Document
        
        print(f"📥 Ingesting: {source}")
        
        doc = Document(
            content=content,
            source=source,
            metadata={"type": "documentation", "topic": "spark"},
        )
        
        result = await self.pipeline.ingest_document(doc, generate_questions=False)
        print(f"  ✓ Created {result['chunks_created']} chunks")
        
        return result
    
    async def evaluate_retrieval(
        self,
        questions: List[Dict[str, Any]],
    ) -> TestResult:
        """Evaluate retrieval quality against ground truth questions."""
        from omniscient_rag.search import HybridSearcher, SearchConfig
        
        print("📊 Evaluating retrieval quality...")
        
        searcher = self.pipeline.searcher
        results_per_question = []
        
        total_precision = 0.0
        total_accuracy = 0.0
        chunks_used = set()
        
        for q in questions:
            question = q["question"]
            expected_answer = q["answer"]
            
            # Search for relevant chunks
            results = await searcher.search(question, top_k=5)
            
            # Calculate precision: do retrieved chunks contain the answer?
            relevant_count = 0
            for r in results:
                content_lower = r.chunk.content.lower()
                answer_lower = expected_answer.lower()
                
                # Check if any key terms from answer appear in chunk
                answer_terms = [t for t in answer_lower.split() if len(t) > 2]
                matches = sum(1 for t in answer_terms if t in content_lower)
                
                if matches >= len(answer_terms) * 0.5:  # At least 50% of terms
                    relevant_count += 1
                    chunks_used.add(str(r.chunk.id))
            
            precision = relevant_count / len(results) if results else 0
            
            # Calculate accuracy using heuristic (keyword matching)
            combined_context = " ".join([r.chunk.content for r in results])
            accuracy = self._calculate_answer_accuracy(expected_answer, combined_context)
            
            total_precision += precision
            total_accuracy += accuracy
            
            results_per_question.append({
                "question": question,
                "expected_answer": expected_answer,
                "precision": precision,
                "accuracy": accuracy,
                "chunks_retrieved": len(results),
                "top_chunk_score": results[0].combined_score if results else 0,
            })
        
        # Get stats
        stats = await self.pipeline.get_stats()
        total_chunks = stats.get("chunks", 1)
        coverage = len(chunks_used) / total_chunks if total_chunks > 0 else 0
        
        avg_precision = total_precision / len(questions)
        avg_accuracy = total_accuracy / len(questions)
        
        # Calculate overall score (weighted)
        overall_score = (
            avg_accuracy * 0.4 +
            avg_precision * 100 * 0.35 +
            coverage * 100 * 0.25
        )
        
        result = TestResult(
            iteration=len(self.results) + 1,
            timestamp=datetime.now(),
            documents_ingested=stats.get("documents", 0),
            chunks_created=stats.get("chunks", 0),
            questions_tested=len(questions),
            retrieval_precision=avg_precision,
            answer_accuracy=avg_accuracy,
            coverage_ratio=coverage,
            overall_score=overall_score,
            per_question_results=results_per_question,
        )
        
        self.results.append(result)
        
        print(f"  → Precision: {avg_precision:.1%}")
        print(f"  → Accuracy: {avg_accuracy:.1f}")
        print(f"  → Coverage: {coverage:.1%}")
        print(f"  → Overall Score: {overall_score:.1f}")
        
        return result
    
    def _calculate_answer_accuracy(self, expected: str, context: str) -> float:
        """Calculate how well context contains the expected answer."""
        if not context:
            return 0.0
        
        expected_lower = expected.lower()
        context_lower = context.lower()
        
        # Direct substring match
        if expected_lower in context_lower:
            return 100.0
        
        # Token matching
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "or", "and", "for", "to", "of"}
        expected_tokens = [t for t in expected_lower.split() if t not in stop_words and len(t) > 2]
        
        if not expected_tokens:
            return 50.0  # Neutral if no meaningful tokens
        
        matches = sum(1 for t in expected_tokens if t in context_lower)
        return min(100.0, (matches / len(expected_tokens)) * 100)
    
    async def run_improvement_test(
        self,
        content_batches: List[tuple],  # (content, source) tuples
        ground_truth: List[Dict[str, Any]],
        iterations: int = 3,
    ) -> ImprovementReport:
        """
        Run iterative improvement test.
        
        Each iteration:
        1. Ingest a batch of content
        2. Evaluate retrieval quality
        3. Track improvement
        """
        print("\n" + "="*60)
        print("🧪 RAG KNOWLEDGE IMPROVEMENT TEST")
        print("="*60 + "\n")
        
        await self.clear_knowledge_base()
        
        batch_size = len(content_batches) // iterations
        
        for i in range(iterations):
            print(f"\n📍 ITERATION {i + 1}/{iterations}")
            print("-" * 40)
            
            # Ingest this iteration's batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < iterations - 1 else len(content_batches)
            
            for content, source in content_batches[start_idx:end_idx]:
                await self.ingest_content(content, source)
            
            # Evaluate
            await self.evaluate_retrieval(ground_truth)
        
        # Generate report
        initial_score = self.results[0].overall_score if self.results else 0
        final_score = self.results[-1].overall_score if self.results else 0
        improvement = ((final_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        
        report = ImprovementReport(
            total_iterations=len(self.results),
            initial_score=initial_score,
            final_score=final_score,
            improvement_percentage=improvement,
            results=self.results,
        )
        
        return report
    
    def print_report(self, report: ImprovementReport):
        """Print a formatted improvement report."""
        print("\n" + "="*60)
        print("📈 IMPROVEMENT REPORT")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"   Iterations: {report.total_iterations}")
        print(f"   Initial Score: {report.initial_score:.1f}")
        print(f"   Final Score: {report.final_score:.1f}")
        print(f"   Improvement: {report.improvement_percentage:+.1f}%")
        
        print(f"\n📉 Score Progression:")
        for r in report.results:
            bar_len = int(r.overall_score / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"   Iter {r.iteration}: [{bar}] {r.overall_score:.1f}")
        
        print(f"\n📋 Per-Iteration Details:")
        for r in report.results:
            print(f"\n   Iteration {r.iteration}:")
            print(f"      Documents: {r.documents_ingested}")
            print(f"      Chunks: {r.chunks_created}")
            print(f"      Precision: {r.retrieval_precision:.1%}")
            print(f"      Accuracy: {r.answer_accuracy:.1f}")
            print(f"      Coverage: {r.coverage_ratio:.1%}")
        
        print("\n" + "="*60 + "\n")


async def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test RAG knowledge improvement")
    parser.add_argument("--content-dir", type=str, help="Directory with documentation to ingest")
    parser.add_argument("--use-sample", action="store_true", help="Use built-in Spark sample docs")
    parser.add_argument("--db-url", type=str, default="postgresql://omniscient:localdev@localhost:5432/omniscient")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--iterations", type=int, default=3, help="Number of test iterations")
    parser.add_argument("--output", type=str, help="Save report to JSON file")
    
    args = parser.parse_args()
    
    if not args.content_dir and not args.use_sample:
        print("Using sample Spark documentation (use --content-dir for custom content)")
        args.use_sample = True
    
    tester = RAGImprovementTester(
        db_url=args.db_url,
        ollama_url=args.ollama_url,
    )
    
    try:
        await tester.initialize()
        
        # Prepare content batches
        if args.use_sample:
            content_batches = [
                (SAMPLE_SPARK_DOCS, "spark_config_guide.md"),
                (SAMPLE_SPARK_ADVANCED, "spark_advanced_optimization.md"),
            ]
            ground_truth = GROUND_TRUTH_QA
        else:
            # Load from directory
            content_dir = Path(args.content_dir)
            content_batches = []
            
            for file_path in content_dir.rglob("*"):
                if file_path.suffix in [".md", ".txt", ".rst"]:
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        content_batches.append((content, str(file_path.name)))
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
            
            if not content_batches:
                print(f"No documentation files found in {content_dir}")
                return
            
            # For custom content, generate questions or use a default set
            ground_truth = GROUND_TRUTH_QA  # Could be extended to auto-generate
        
        # Run test
        report = await tester.run_improvement_test(
            content_batches=content_batches,
            ground_truth=ground_truth,
            iterations=min(args.iterations, len(content_batches)),
        )
        
        # Print report
        tester.print_report(report)
        
        # Save if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            print(f"📁 Report saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    import sys
    import selectors
    
    # Windows compatibility: psycopg requires SelectorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
