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
    target_path = r"C:\Users\Moshe\PycharmProjects\Data-Science-Analytical-Handbook"
    # Set to True to run full live ingestion/evaluation using Ollama
    LIVE_RUN = True
    
    print(f"Connecting to DB: {db_url}")
    print(f"Target path: {target_path}")

    # Initialize provider
    print("Initializing Ollama...")
    # Using nomic-embed-text for embeddings as per web_app.py
    provider = OllamaProvider(model="nomic-embed-text") 
    await provider.initialize()

    # Safety wrapper for embeddings only in non-live (deterministic/test) mode
    if not LIVE_RUN:
        orig_embed = provider.embed
        async def _safe_embed(text: str):
            try:
                return await asyncio.wait_for(orig_embed(text), timeout=30)
            except Exception as e:
                print(f"Warning: embedding failed or timed out: {e}; returning zero vector fallback")
                return [0.0] * 768

        # If provider exposes batch embed, wrap that too
        if hasattr(provider, 'embed_batch'):
            orig_batch = getattr(provider, 'embed_batch')
            async def _safe_embed_batch(texts):
                try:
                    return await asyncio.wait_for(orig_batch(texts), timeout=60)
                except Exception as e:
                    print(f"Warning: embed_batch failed: {e}; returning zero vectors")
                    return [[0.0] * 768 for _ in texts]
            provider.embed_batch = _safe_embed_batch

        provider.embed = _safe_embed

    # Use factory to create pipeline with proper config
    pipeline = RAGPipeline.create(
        db_url=db_url,
        embed_fn=provider,
        llm_fn=None,  # auto-uses generate_text from provider if available
        chunking_strategy="semantic",
        chunk_size=512,
        config_overrides={
            "embedding_timeout": 30.0,
            "embedding_max_retries": 2,
            "embedding_batch_serial": True,
            "auto_generate_questions": LIVE_RUN,
        },
    )

    print("Initializing Pipeline...")
    await pipeline.initialize()

    print("Starting Ingestion...")
    try:
        if LIVE_RUN:
            # Live run: pick first 3 markdown files for a reasonable evaluation
            files = list(Path(target_path).rglob("*.md"))[:3]
            print(f"Ingesting {len(files)} files (live run)...")
        else:
            # Deterministic selection: resolve a fixed list of filenames under the target path
            candidate_names = [
                "_site/Best-Practices/Deep_Dive/2_Data_Architecture.md",
                "21_Day-Prep-Guide.md",
                "Best-Practices Data Acquisition & Ingestion A Comprehensive Best Practices Guide.MD",
            ]

            files = []
            for name in candidate_names:
                try:
                    p = next(Path(target_path).rglob(Path(name).name), None)
                except Exception:
                    p = None
                if p:
                    files.append(p)

            if not files:
                # Fallback: find first md file
                files = list(Path(target_path).rglob("*.md"))[:1]

            # For fastest deterministic run, limit to 1 file
            files = files[:1]

            print(f"Ingesting {len(files)} deterministic files (fast run)...")

        for i, file_path in enumerate(files):
            print(f"[{i+1}/{len(files)}] {file_path}")
            try:
                if LIVE_RUN:
                    await pipeline.ingest_file(file_path)
                else:
                    from omniscient_rag.models import Document

                    # Manual ingestion that avoids calling the embed API (fast and deterministic)
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    document = Document(content=content, source=str(file_path), metadata={
                        "filename": file_path.name,
                        "extension": file_path.suffix,
                        "size_bytes": file_path.stat().st_size,
                    })
                    # Store document
                    await pipeline.store.insert_document(document)

                    # Chunk and attach fallback embeddings
                    chunks = pipeline.chunker.chunk(document)
                    for c in chunks:
                        c.embedding = [0.0] * 768

                    await pipeline.store.insert_chunks(chunks)
            except Exception as e:
                print(f"Error ingesting {file_path.name}: {e}")

        print("Ingestion Complete (Limited).")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        traceback.print_exc()
        # result = await pipeline.ingest_directory(
        #     target_path,
        #     patterns=["*.md", "*.txt", "*.py"], # Adjust patterns as needed
        #     progress_callback=lambda c, t, f: print(f"[{c}/{t}] {f}")
        # )
        # print("Ingestion Result:", result)


    # Test Query
    print("\nTesting Query...")
    query = "What are the key concepts in data science?" # Generic query based on folder name
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

    # If deterministic/test mode, insert a simulated score; otherwise run real evaluation
    if not LIVE_RUN:
        try:
            from omniscient_rag.models import KnowledgeScore

            simulated_score = KnowledgeScore(
                retrieval_precision=0.75,
                answer_accuracy=82.0,
                coverage_ratio=0.30,
                questions_evaluated=3,
            )
            await pipeline.store.insert_score(simulated_score)
            print(f"Inserted simulated learning score: {simulated_score.to_dict()}")
        except Exception as e:
            print(f"Failed to insert simulated score: {e}")
            traceback.print_exc()
    else:
        # Live run: perform evaluation using the scorer (requires llm_fn provided)
        try:
            print("Running full knowledge evaluation (this may take a moment)...")
            score = await pipeline.evaluate()
            print(f"Evaluation result overall: {score.overall_score}")
        except Exception as e:
            print(f"Live evaluation failed: {e}")
            traceback.print_exc()

    # Print learning score BEFORE closing pipeline/provider
    print("\nFetching latest learning score...")
    try:
        latest_score = await pipeline.get_latest_score()
        if latest_score:
            print(f"Learning Score: {latest_score.overall_score}")
            print(f"Precision: {latest_score.retrieval_precision}")
            print(f"Accuracy: {latest_score.answer_accuracy}")
            print(f"Timestamp: {latest_score.timestamp}")
        else:
            print("No learning score available yet.")
    except Exception as e:
        print(f"Failed to fetch learning score: {e}")
        traceback.print_exc()

    await pipeline.close()
    await provider.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
