"""Simple RAG test with minimal data."""

import asyncio
from pathlib import Path
from omniscient_llm.providers.ollama import OllamaProvider
from omniscient_rag.pipeline import RAGPipeline

DB_URL = "postgresql://omniscient:localdev@localhost:5432/omniscient"


async def main():
    print("Simple RAG Test\n")
    
    # Embedding provider
    embed_provider = OllamaProvider(model="nomic-embed-text", timeout=120.0)
    await embed_provider.initialize()

    # LLM provider for question generation
    llm_provider = OllamaProvider(model="qwen2.5-coder:1.5b", timeout=120.0)
    await llm_provider.initialize()

    # Create pipeline
    pipeline = RAGPipeline.create(
        db_url=DB_URL,
        embed_fn=embed_provider,
        llm_fn=llm_provider.generate_text,  # Explicitly pass generate_text from LLM provider
        chunking_strategy="semantic",
        chunk_size=200,
        chunk_overlap=0.1,
        config_overrides={
            "embedding_timeout": 120.0,
            "embedding_max_retries": 3,
            "similarity_threshold": 0.3,
        },
    )

    await pipeline.initialize()

    # Clear database
    print("Clearing database...")
    await pipeline.store.clear_all()

    # Create simple test document
    test_text = """
    Machine learning is a subset of artificial intelligence.
    It involves training models on data to make predictions.
    Neural networks are a popular machine learning technique.
    """
    
    print(f"\nIngesting test document...")
    result = await pipeline.ingest_text(test_text, source="test.txt")
    print(f"Result: {result.get('chunks_created', 0)} chunks, {result.get('questions_generated', 0)} questions")
    
    # Check stats
    stats = await pipeline.store.get_stats()
    print(f"\nDatabase stats: {stats}")
    
    # Query
    if result.get('chunks_created', 0) > 0:
        print("\nQuerying...")
        query_results = await pipeline.query("What is machine learning?", top_k=3)
        print(f"Found {len(query_results)} results")
        if query_results:
            print(f"Top result: {query_results[0].chunk.content[:100]}...")

    await embed_provider.close()
    await llm_provider.close()
    await pipeline.close()
    print("\nTest complete!")


if __name__ == "__main__":
    # Windows fix for psycopg async
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
