"""
Example: Knowledge Persistence Across Ollama Sessions

This script demonstrates how learned knowledge persists and improves
the model's responses over time, even when Ollama restarts.

The key insight is that while Ollama doesn't retain state between runs,
we create a "memory layer" in PostgreSQL that:
1. Stores successful Q&A pairs
2. Records reasoning patterns that worked
3. Learns query refinements
4. Injects this knowledge into future prompts

Run this example:
    python scripts/demo_knowledge_persistence.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add package paths
project_root = Path(__file__).parent.parent
for pkg in ["core", "llm", "rag"]:
    src_path = project_root / "packages" / pkg / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))


async def main():
    """Demonstrate knowledge persistence across sessions."""
    
    print("=" * 60)
    print("🧠 KNOWLEDGE PERSISTENCE DEMO")
    print("=" * 60)
    print()
    
    # --- Setup ---
    from omniscient_rag.store import PostgresVectorStore, DatabaseConfig
    from omniscient_rag.learning import (
        KnowledgeMemory,
        ContextInjector,
        FeedbackLearner,
        LearnedFact,
        ReasoningChain,
        FeedbackType,
        InjectionConfig,
    )
    from omniscient_llm import OllamaProvider
    
    # Initialize components
    db_url = "postgresql://omniscient:localdev@localhost:5432/omniscient"
    store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
    await store.initialize()
    
    provider = OllamaProvider(model="qwen2.5-coder:1.5b")
    await provider.initialize()
    
    async def embed_fn(text: str) -> list:
        return await provider.embed(text, model="nomic-embed-text")
    
    async def llm_fn(prompt: str) -> str:
        from omniscient_llm.models import GenerationRequest
        request = GenerationRequest(messages=[{"role": "user", "content": prompt}])
        response = await provider.generate(request)
        return response.content
    
    # Initialize learning components
    memory = KnowledgeMemory(store, embed_fn)
    await memory.initialize()
    
    injector = ContextInjector(memory, InjectionConfig(
        max_facts=5,
        max_reasoning_chains=2,
        include_few_shot_reasoning=True,
    ))
    
    learner = FeedbackLearner(memory, llm_fn)
    
    # --- Simulate Session 1: Initial Learning ---
    print("📚 SESSION 1: Teaching the System")
    print("-" * 40)
    
    # Simulate a successful Q&A interaction
    question1 = "How do I configure Spark executor memory?"
    answer1 = """To configure Spark executor memory:

Step 1: Set spark.executor.memory in your Spark configuration
Step 2: The value should be in format like "4g" or "8g" 
Step 3: Ensure you leave room for overhead (spark.executor.memoryOverhead)

Therefore, use: spark.executor.memory=4g in your spark-defaults.conf or pass it via --conf."""

    print(f"Q: {question1}")
    print(f"A: {answer1[:200]}...")
    print()
    
    # User gives positive feedback
    learned = await learner.process_feedback(
        question=question1,
        answer=answer1,
        rating=0.9,  # 4.5/5 stars
        feedback_type=FeedbackType.POSITIVE,
    )
    print(f"✅ Learned from feedback: {learned}")
    
    # Manually add a high-confidence fact
    spark_fact = LearnedFact(
        topic="Spark Memory",
        fact="spark.executor.memory controls the heap size of each executor JVM",
        source_question=question1,
        source_answer=answer1,
        confidence=0.85,
    )
    await memory.store_fact(spark_fact)
    print(f"📝 Stored fact: {spark_fact.fact}")
    
    # Store a reasoning chain
    chain = ReasoningChain(
        question="How should I size Spark executors?",
        steps=[
            "Check available cluster memory",
            "Divide by number of executors desired",
            "Subtract ~10% for memory overhead",
            "Set spark.executor.memory to the result"
        ],
        final_answer="Use spark.executor.memory = (node_memory / num_executors) - overhead",
        was_correct=True,
        feedback_score=0.9,
    )
    await memory.store_reasoning_chain(chain)
    print(f"🔗 Stored reasoning chain: {chain.question}")
    
    print()
    stats = await memory.get_statistics()
    print(f"📊 Knowledge Base Stats: {stats}")
    print()
    
    # --- Simulate Session 2: Using Learned Knowledge ---
    print("=" * 60)
    print("🔄 SESSION 2: New Ollama Instance (knowledge injected)")
    print("-" * 40)
    
    # A new question comes in
    new_question = "What's the best way to set Spark memory settings?"
    
    # Build context with injected knowledge
    context = await injector.build_context(
        query=new_question,
        topic="Spark",
    )
    
    print(f"New Question: {new_question}")
    print()
    print("📎 Injected Context from Previous Sessions:")
    print("-" * 40)
    print(context[:1000] if context else "(no relevant knowledge found)")
    print()
    
    # In real usage, this context would be prepended to the prompt
    full_prompt = f"""{context}
## Current Question
{new_question}

Please answer based on the context above and your knowledge."""

    print("🤖 Sending to Ollama with injected knowledge...")
    response = await llm_fn(full_prompt)
    print()
    print("Response:")
    print(response[:500])
    print()
    
    # --- Demonstrate Correction Learning ---
    print("=" * 60)
    print("📝 Learning from a Correction")
    print("-" * 40)
    
    # User corrects a mistake
    correction_feedback = await learner.process_feedback(
        question="What's the default Spark executor memory?",
        answer="The default is 2g.",
        rating=-0.5,  # User marked as wrong
        feedback_type=FeedbackType.CORRECTION,
        correction="The default is actually 1g, not 2g.",
    )
    print(f"Correction processed: {correction_feedback}")
    
    # --- Show Final Statistics ---
    print()
    print("=" * 60)
    print("📊 FINAL KNOWLEDGE BASE STATISTICS")
    print("-" * 40)
    
    final_stats = await memory.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # --- Demonstrate Knowledge Retrieval for Next Session ---
    print()
    print("=" * 60)
    print("🔮 PREVIEW: Knowledge for Next Session")
    print("-" * 40)
    
    future_query = "How do I optimize Spark memory configuration?"
    knowledge = await memory.get_relevant_knowledge(
        query=future_query,
        max_facts=5,
        max_chains=2,
    )
    
    print(f"Query: {future_query}")
    print()
    print(f"Retrieved Facts: {len(knowledge['facts'])}")
    for fact in knowledge['facts']:
        print(f"  • [{fact.confidence:.0%}] {fact.topic}: {fact.fact}")
    
    print(f"\nRetrieved Reasoning Chains: {len(knowledge['reasoning_chains'])}")
    for chain in knowledge['reasoning_chains']:
        print(f"  • Q: {chain.question}")
        print(f"    Steps: {len(chain.steps)}")
    
    print(f"\nCorrections to Apply: {len(knowledge['corrections'])}")
    for corr in knowledge['corrections']:
        print(f"  • {corr.correction}")
    
    # Cleanup
    await provider.close()
    await store.close()
    
    print()
    print("=" * 60)
    print("✨ Demo complete! Knowledge persists across Ollama sessions.")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
