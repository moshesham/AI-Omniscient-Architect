import sys
from pathlib import Path
import asyncio
import traceback

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "agents", "tools", "github", "api", "llm", "rag"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): sys.path.insert(0, str(_path))

from omniscient_llm import OllamaProvider
from omniscient_rag import RAGPipeline

async def main():
    db_url = "postgresql://omniscient:localdev@localhost:5432/omniscient"
    
    print(f"Connecting to DB: {db_url}")

    # Initialize provider
    print("Initializing Ollama...")
    provider = OllamaProvider(model="nomic-embed-text") 
    await provider.initialize()

    # Use factory to create pipeline
    pipeline = RAGPipeline.create(
        db_url=db_url,
        embed_fn=provider,
        chunking_strategy="semantic",
        chunk_size=512,
        config_overrides={
            "embedding_timeout": 30.0,
            "embedding_batch_serial": True,
        },
    )

    print("Initializing Pipeline...")
    await pipeline.initialize()

    # Test Query
    print("\nTesting Query...")
    query = "What are the key concepts in data science?"
    try:
        results = await pipeline.query(query, top_k=3)
        print(f"Query: {query}")
        if not results:
            print("No results found.")
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Score: {res.combined_score}")
            content = res.chunk.content if hasattr(res, 'chunk') else res.content
            print(f"Content: {content[:200]}...")
            print(f"Source: {res.source}")
    except Exception as e:
        print(f"Query failed: {e}")
        traceback.print_exc()

    await pipeline.close()
    await provider.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
