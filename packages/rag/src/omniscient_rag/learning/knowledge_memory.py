"""Persistent knowledge memory for cross-session learning.

This module stores learned knowledge that persists across Ollama sessions:
- Successful Q&A pairs with high ratings
- Reasoning chains that led to correct answers
- Query refinements that improved retrieval
- User corrections and feedback

When a new Ollama session starts, this knowledge is injected as context
to give the model "memory" of past learning.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

from omniscient_core import optional_import

HAS_POSTGRES, _ = optional_import("psycopg")
if HAS_POSTGRES:
    from psycopg.rows import dict_row


class FeedbackType(Enum):
    """Types of user feedback."""
    POSITIVE = "positive"      # Answer was helpful
    NEGATIVE = "negative"      # Answer was wrong/unhelpful
    CORRECTION = "correction"  # User provided correct answer
    CLARIFICATION = "clarification"  # User clarified the question


@dataclass
class LearnedFact:
    """A fact learned from user interactions.
    
    These are extracted from successful Q&A sessions and stored
    for injection into future sessions.
    """
    id: UUID = field(default_factory=uuid4)
    topic: str = ""
    fact: str = ""
    source_question: str = ""
    source_answer: str = ""
    confidence: float = 0.5
    usage_count: int = 0
    success_rate: float = 0.0
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_format(self) -> str:
        """Format fact for inclusion in prompt."""
        return f"• {self.topic}: {self.fact}"


@dataclass 
class ReasoningChain:
    """A successful chain of reasoning steps.
    
    Stored when the model produces a correct answer through
    step-by-step reasoning, so similar patterns can be reused.
    """
    id: UUID = field(default_factory=uuid4)
    question: str = ""
    steps: List[str] = field(default_factory=list)
    final_answer: str = ""
    was_correct: bool = False
    feedback_score: float = 0.0
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_few_shot_example(self) -> str:
        """Format as few-shot example for prompting."""
        steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(self.steps))
        return f"""Question: {self.question}
Reasoning:
{steps_text}
Answer: {self.final_answer}"""


@dataclass
class QueryRefinement:
    """A learned query refinement pattern.
    
    When a user's initial query fails but a refined version succeeds,
    we store the mapping to improve future queries automatically.
    """
    id: UUID = field(default_factory=uuid4)
    original_query: str = ""
    refined_query: str = ""
    improvement_score: float = 0.0  # How much retrieval improved
    times_applied: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """User feedback on a response."""
    id: UUID = field(default_factory=uuid4)
    question: str = ""
    answer: str = ""
    feedback_type: FeedbackType = FeedbackType.POSITIVE
    correction: Optional[str] = None  # If user provided correct answer
    rating: float = 0.0  # -1 to 1
    context_chunks: List[UUID] = field(default_factory=list)  # Which chunks were used
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


def _ensure_uuid(value) -> UUID:
    """Convert to UUID if string."""
    return value if isinstance(value, UUID) else UUID(str(value))


class KnowledgeMemory:
    """Persistent knowledge store for cross-session learning.
    
    This class manages the storage and retrieval of learned knowledge
    that persists across Ollama sessions. When a new session starts,
    relevant knowledge is retrieved and injected into the context.
    
    Example:
        >>> memory = KnowledgeMemory(store)
        >>> await memory.initialize()
        >>> 
        >>> # Store learned fact
        >>> fact = LearnedFact(
        ...     topic="Spark",
        ...     fact="spark.executor.memory controls executor heap size",
        ...     source_question="How do I set executor memory?",
        ...     source_answer="Use spark.executor.memory configuration...",
        ...     confidence=0.9
        ... )
        >>> await memory.store_fact(fact)
        >>>
        >>> # On next session, retrieve relevant knowledge
        >>> context = await memory.get_relevant_knowledge(
        ...     query="Spark memory configuration",
        ...     top_k=5
        ... )
    """
    
    def __init__(self, store, embed_fn=None):
        """Initialize knowledge memory.
        
        Args:
            store: PostgresVectorStore instance
            embed_fn: Async function to generate embeddings
        """
        self.store = store
        self.embed_fn = embed_fn
        self._initialized = False
    
    async def initialize(self) -> None:
        """Create required database tables."""
        if self._initialized:
            return
            
        schema = self.store.config.schema
        dim = self.store.config.embedding_dimensions
        
        async with self.store._pool.connection() as conn:
            # Enable pg_trgm for text similarity search
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            
            # Learned facts table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.learned_facts (
                    id UUID PRIMARY KEY,
                    topic TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    source_question TEXT,
                    source_answer TEXT,
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'
                )
            """)
            
            # Reasoning chains table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.reasoning_chains (
                    id UUID PRIMARY KEY,
                    question TEXT NOT NULL,
                    steps JSONB NOT NULL,
                    final_answer TEXT NOT NULL,
                    was_correct BOOLEAN DEFAULT FALSE,
                    feedback_score REAL DEFAULT 0.0,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'
                )
            """)
            
            # Query refinements table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.query_refinements (
                    id UUID PRIMARY KEY,
                    original_query TEXT NOT NULL,
                    refined_query TEXT NOT NULL,
                    improvement_score REAL DEFAULT 0.0,
                    times_applied INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'
                )
            """)
            
            # User feedback table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.user_feedback (
                    id UUID PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    correction TEXT,
                    rating REAL DEFAULT 0.0,
                    context_chunks UUID[],
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'
                )
            """)
            
            # Indexes for efficient retrieval
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_facts_embedding 
                ON {schema}.learned_facts 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_chains_embedding 
                ON {schema}.reasoning_chains 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_facts_confidence 
                ON {schema}.learned_facts (confidence DESC)
            """)
            
        self._initialized = True
    
    # =========================================================================
    # Fact Operations
    # =========================================================================
    
    async def store_fact(self, fact: LearnedFact) -> UUID:
        """Store a learned fact."""
        schema = self.store.config.schema
        
        # Generate embedding if not provided
        if fact.embedding is None and self.embed_fn:
            fact.embedding = await self.embed_fn(f"{fact.topic}: {fact.fact}")
        
        async with self.store._pool.connection() as conn:
            await conn.execute(
                f"""
                INSERT INTO {schema}.learned_facts
                (id, topic, fact, source_question, source_answer, confidence,
                 usage_count, success_rate, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    usage_count = EXCLUDED.usage_count,
                    success_rate = EXCLUDED.success_rate,
                    updated_at = NOW()
                """,
                (
                    str(fact.id), fact.topic, fact.fact,
                    fact.source_question, fact.source_answer,
                    fact.confidence, fact.usage_count, fact.success_rate,
                    fact.embedding, json.dumps(fact.metadata)
                )
            )
        
        return fact.id
    
    async def get_relevant_facts(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.3,
    ) -> List[LearnedFact]:
        """Retrieve facts relevant to a query."""
        if not self.embed_fn:
            return []
        
        query_embedding = await self.embed_fn(query)
        schema = self.store.config.schema
        
        async with self.store._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT *, 1 - (embedding <=> %s::vector) AS similarity
                    FROM {schema}.learned_facts
                    WHERE confidence >= %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, min_confidence, query_embedding, top_k)
                )
                rows = await cur.fetchall()
                
                return [
                    LearnedFact(
                        id=_ensure_uuid(row["id"]),
                        topic=row["topic"],
                        fact=row["fact"],
                        source_question=row["source_question"],
                        source_answer=row["source_answer"],
                        confidence=row["confidence"],
                        usage_count=row["usage_count"],
                        success_rate=row["success_rate"],
                        metadata=row["metadata"] or {},
                    )
                    for row in rows
                ]
    
    async def update_fact_usage(self, fact_id: UUID, was_helpful: bool) -> None:
        """Update fact usage statistics after it was used."""
        schema = self.store.config.schema
        
        async with self.store._pool.connection() as conn:
            # Update usage count and success rate
            await conn.execute(
                f"""
                UPDATE {schema}.learned_facts
                SET usage_count = usage_count + 1,
                    success_rate = (success_rate * usage_count + %s) / (usage_count + 1),
                    confidence = LEAST(1.0, confidence + %s),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (1.0 if was_helpful else 0.0, 0.05 if was_helpful else -0.02, str(fact_id))
            )
    
    # =========================================================================
    # Reasoning Chain Operations
    # =========================================================================
    
    async def store_reasoning_chain(self, chain: ReasoningChain) -> UUID:
        """Store a successful reasoning chain."""
        schema = self.store.config.schema
        
        # Generate embedding from question
        if chain.embedding is None and self.embed_fn:
            chain.embedding = await self.embed_fn(chain.question)
        
        async with self.store._pool.connection() as conn:
            await conn.execute(
                f"""
                INSERT INTO {schema}.reasoning_chains
                (id, question, steps, final_answer, was_correct, 
                 feedback_score, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(chain.id), chain.question, json.dumps(chain.steps),
                    chain.final_answer, chain.was_correct, chain.feedback_score,
                    chain.embedding, json.dumps(chain.metadata)
                )
            )
        
        return chain.id
    
    async def get_similar_reasoning(
        self,
        query: str,
        top_k: int = 3,
        only_correct: bool = True,
    ) -> List[ReasoningChain]:
        """Find similar reasoning chains for few-shot prompting."""
        if not self.embed_fn:
            return []
        
        query_embedding = await self.embed_fn(query)
        schema = self.store.config.schema
        
        correct_filter = "AND was_correct = TRUE" if only_correct else ""
        
        async with self.store._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT *, 1 - (embedding <=> %s::vector) AS similarity
                    FROM {schema}.reasoning_chains
                    WHERE embedding IS NOT NULL {correct_filter}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k)
                )
                rows = await cur.fetchall()
                
                return [
                    ReasoningChain(
                        id=_ensure_uuid(row["id"]),
                        question=row["question"],
                        steps=row["steps"] if isinstance(row["steps"], list) else json.loads(row["steps"]),
                        final_answer=row["final_answer"],
                        was_correct=row["was_correct"],
                        feedback_score=row["feedback_score"],
                        metadata=row["metadata"] or {},
                    )
                    for row in rows
                ]
    
    # =========================================================================
    # Query Refinement Operations
    # =========================================================================
    
    async def store_refinement(self, refinement: QueryRefinement) -> UUID:
        """Store a query refinement pattern."""
        schema = self.store.config.schema
        
        async with self.store._pool.connection() as conn:
            await conn.execute(
                f"""
                INSERT INTO {schema}.query_refinements
                (id, original_query, refined_query, improvement_score, 
                 times_applied, success_rate, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(refinement.id), refinement.original_query,
                    refinement.refined_query, refinement.improvement_score,
                    refinement.times_applied, refinement.success_rate,
                    json.dumps(refinement.metadata)
                )
            )
        
        return refinement.id
    
    async def find_refinement(self, query: str) -> Optional[QueryRefinement]:
        """Find if there's a learned refinement for this query pattern."""
        schema = self.store.config.schema
        
        async with self.store._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # Look for similar queries using trigram similarity
                await cur.execute(
                    f"""
                    SELECT *, similarity(original_query, %s) AS sim
                    FROM {schema}.query_refinements
                    WHERE similarity(original_query, %s) > 0.3
                      AND success_rate > 0.5
                    ORDER BY sim DESC
                    LIMIT 1
                    """,
                    (query, query)
                )
                row = await cur.fetchone()
                
                if not row:
                    return None
                
                return QueryRefinement(
                    id=_ensure_uuid(row["id"]),
                    original_query=row["original_query"],
                    refined_query=row["refined_query"],
                    improvement_score=row["improvement_score"],
                    times_applied=row["times_applied"],
                    success_rate=row["success_rate"],
                    metadata=row["metadata"] or {},
                )
    
    # =========================================================================
    # Feedback Operations
    # =========================================================================
    
    async def store_feedback(self, feedback: UserFeedback) -> UUID:
        """Store user feedback."""
        schema = self.store.config.schema
        
        async with self.store._pool.connection() as conn:
            await conn.execute(
                f"""
                INSERT INTO {schema}.user_feedback
                (id, question, answer, feedback_type, correction, 
                 rating, context_chunks, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(feedback.id), feedback.question, feedback.answer,
                    feedback.feedback_type.value, feedback.correction,
                    feedback.rating, 
                    [str(c) for c in feedback.context_chunks] if feedback.context_chunks else None,
                    json.dumps(feedback.metadata)
                )
            )
        
        return feedback.id
    
    async def get_corrections(self, topic: Optional[str] = None) -> List[UserFeedback]:
        """Get user corrections for learning."""
        schema = self.store.config.schema
        
        async with self.store._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT * FROM {schema}.user_feedback
                    WHERE feedback_type = 'correction'
                      AND correction IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 100
                    """
                )
                rows = await cur.fetchall()
                
                return [
                    UserFeedback(
                        id=_ensure_uuid(row["id"]),
                        question=row["question"],
                        answer=row["answer"],
                        feedback_type=FeedbackType(row["feedback_type"]),
                        correction=row["correction"],
                        rating=row["rating"],
                        metadata=row["metadata"] or {},
                    )
                    for row in rows
                ]
    
    # =========================================================================
    # Composite Knowledge Retrieval
    # =========================================================================
    
    async def get_relevant_knowledge(
        self,
        query: str,
        max_facts: int = 5,
        max_chains: int = 2,
        include_corrections: bool = True,
    ) -> Dict[str, Any]:
        """Get all relevant knowledge for a query.
        
        This is the main method called at the start of each session
        to inject learned knowledge into the context.
        
        Returns:
            Dictionary with facts, reasoning chains, and corrections
        """
        facts = await self.get_relevant_facts(query, top_k=max_facts)
        chains = await self.get_similar_reasoning(query, top_k=max_chains)
        
        # Get recent corrections if requested
        corrections = []
        if include_corrections:
            all_corrections = await self.get_corrections()
            # Filter to relevant ones (simple text matching for now)
            query_terms = set(query.lower().split())
            corrections = [
                c for c in all_corrections
                if any(term in c.question.lower() for term in query_terms)
            ][:3]
        
        # Check for query refinements
        refinement = await self.find_refinement(query)
        
        return {
            "facts": facts,
            "reasoning_chains": chains,
            "corrections": corrections,
            "query_refinement": refinement,
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge memory statistics."""
        schema = self.store.config.schema
        
        async with self.store._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.learned_facts")
                facts_count = (await cur.fetchone())["count"]
                
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.reasoning_chains WHERE was_correct = TRUE")
                chains_count = (await cur.fetchone())["count"]
                
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.query_refinements")
                refinements_count = (await cur.fetchone())["count"]
                
                await cur.execute(f"SELECT COUNT(*) as count FROM {schema}.user_feedback")
                feedback_count = (await cur.fetchone())["count"]
                
                await cur.execute(f"""
                    SELECT AVG(confidence) as avg_confidence, 
                           AVG(success_rate) as avg_success
                    FROM {schema}.learned_facts
                """)
                row = await cur.fetchone()
                
                return {
                    "total_facts": facts_count,
                    "successful_chains": chains_count,
                    "query_refinements": refinements_count,
                    "total_feedback": feedback_count,
                    "average_fact_confidence": row["avg_confidence"] or 0,
                    "average_fact_success_rate": row["avg_success"] or 0,
                }
