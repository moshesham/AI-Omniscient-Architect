import asyncio
import sys
from pathlib import Path

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "agents", "tools", "github", "api", "llm", "rag"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): sys.path.insert(0, str(_path))

from omniscient_rag.store import PostgresVectorStore, DatabaseConfig

async def main():
    db_url = "postgresql://omniscient:localdev@localhost:5432/omniscient"
    store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
    await store.initialize()
    try:
        stats = await store.get_stats()
        print("Store stats:", stats)

        scores = await store.get_score_history(limit=5)
        print("Recent scores:")
        for s in scores:
            print(s.to_dict())

        # Try to get questions
        try:
            questions = await store.get_questions(limit=10)
            print(f"Loaded {len(questions)} knowledge questions")
            for q in questions[:5]:
                print({"id": str(q.id), "question": q.question, "expected": q.expected_answer})
        except Exception as e:
            print("Failed to load questions:", e)
    finally:
        await store.close()

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
