# Omniscient RAG

RAG (Retrieval-Augmented Generation) infrastructure for Omniscient Architect.

## Features

- **Pluggable Chunking Strategies**: Fixed, Semantic, AST-based
- **PostgreSQL + pgvector**: Unified storage for vectors and metadata
- **Hybrid Search**: Vector similarity + BM25 with Reciprocal Rank Fusion
- **Knowledge Metrics**: Track retrieval quality and model understanding

## Installation

```bash
pip install omniscient-rag

# For AST-based chunking (Python/JavaScript parsing)
pip install omniscient-rag[ast]
```

## Quick Start

```python
from omniscient_rag import RAGPipeline, ChunkerFactory
from omniscient_rag.store import PostgresVectorStore
from omniscient_rag.search import HybridSearcher

# Initialize store
store = PostgresVectorStore(connection_string="postgresql://...")

# Create chunker (user-selectable strategy)
chunker = ChunkerFactory.create("semantic")  # or "fixed", "ast", "auto"

# Build pipeline
pipeline = RAGPipeline(store=store, chunker=chunker)

# Ingest documents
await pipeline.ingest("/path/to/docs")

# Query with hybrid search
results = await pipeline.query("How to configure Spark executors?")
```

## Chunking Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `fixed` | General text | Token-based with configurable size/overlap |
| `semantic` | Documentation | Split by headings, paragraphs, code blocks |
| `ast` | Source code | Parse by function/class boundaries |
| `auto` | Mixed content | Auto-detect based on file extension |
