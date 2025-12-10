import sys
from pathlib import Path
import asyncio
import traceback

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "agents", "tools", "github", "api", "llm", "rag"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists():
        sys.path.insert(0, str(_path))

from omniscient_llm import OllamaProvider
from omniscient_rag import RAGPipeline

async def main():
    import sys
    
    db_url = "postgresql://omniscient:localdev@localhost:5432/omniscient"
    
    # Get target path from command line or prompt user
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        print("Usage: python evaluate_rag_folder.py <target_folder_path>")
        print("Example: python evaluate_rag_folder.py C:\\Users\\Moshe\\PycharmProjects\\Data-Science-Analytical-Handbook")
        return

    if not target_path.exists():
        print(f"Error: Path does not exist: {target_path}")
        return

    print(f"Connecting to DB: {db_url}")
    print(f"Using files from: {target_path}")

    # Embedding provider (reuse as generator via generate_text)
    embed_provider = OllamaProvider(model="nomic-embed-text", timeout=120.0)  # Increased timeout
    await embed_provider.initialize()

    # Use the factory to auto-wire embed/generate_text and config overrides
    pipeline = RAGPipeline.create(
        db_url=db_url,
        embed_fn=embed_provider,  # provider instance is accepted
        llm_fn=None,  # will auto-use generate_text from provider
        chunking_strategy="semantic",
        chunk_size=512,
        chunk_overlap=0.1,
        config_overrides={
            "embedding_timeout": 120.0,  # Increased for slower Ollama responses
            "embedding_max_retries": 3,
            "embedding_batch_serial": True,
            "questions_per_document": 2,
        },
    )

    await pipeline.initialize()

    # Clear old data to avoid pollution from zero-vector embeddings
    print("\nClearing old data...")
    stats_before = await pipeline.store.get_stats()
    print(f"Before: {stats_before}")
    deleted = await pipeline.store.clear_all()
    print(f"Deleted: {deleted}")
    stats_after = await pipeline.store.get_stats()
    print(f"After clear: {stats_after}")

    # Use heuristic scoring (faster than LLM-based scoring)
    if pipeline.scorer:
        pipeline.scorer.config.use_llm_scoring = False

    # Collect ALL files from the target folder
    files = []
    for ext in ("*.md", "*.txt", "*.py"):
        files.extend(list(target_path.rglob(ext)))

    if not files:
        print("No files found in the target folder.")
        await pipeline.close()
        await embed_provider.close()
        return

    print(f"Ingesting {len(files)} files with real embeddings and auto-generated questions...")

    for i, p in enumerate(files):
        try:
            print(f"[{i+1}/{len(files)}] Ingesting {p.name}...")
            result = await pipeline.ingest_file(p)
            print(f"  Chunks: {result['chunks_created']}, Questions: {result['questions_generated']}")
        except Exception as e:
            print(f"Failed to ingest {p}: {e}")
            traceback.print_exc()

    # Run evaluation
    try:
        print("Running evaluation...")
        score = await pipeline.evaluate()
        print("Evaluation complete:")
        print(score.to_dict())
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()

    await pipeline.close()
    await embed_provider.close()

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
