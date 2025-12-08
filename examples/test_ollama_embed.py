import asyncio
import sys
from pathlib import Path

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "agents", "tools", "github", "api", "llm", "rag"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): sys.path.insert(0, str(_path))

from omniscient_llm import OllamaProvider

async def main():
    print("Initializing Ollama Provider...")
    provider = OllamaProvider(model="nomic-embed-text", timeout=30.0)
    await provider.initialize()

    text = "This is a test sentence to verify embedding generation."
    print(f"Embedding text: '{text}'")

    try:
        start_time = asyncio.get_event_loop().time()
        embedding = await provider.embed(text)
        end_time = asyncio.get_event_loop().time()
        
        print(f"Success! Embedding length: {len(embedding)}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await provider.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
