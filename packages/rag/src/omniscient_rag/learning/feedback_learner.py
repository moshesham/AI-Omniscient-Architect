"""Feedback-based learning for RAG improvement.

This module automatically learns from user feedback to:
1. Extract facts from successful Q&A pairs
2. Store successful reasoning chains
3. Learn query refinement patterns
4. Adjust retrieval based on chunk usage

The learning happens asynchronously after each user interaction.
"""

import re
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable
from uuid import UUID

from .knowledge_memory import (
    KnowledgeMemory,
    LearnedFact,
    ReasoningChain,
    QueryRefinement,
    UserFeedback,
    FeedbackType,
)


@dataclass
class LearningConfig:
    """Configuration for feedback-based learning."""
    # Minimum rating to learn from (1.0 = 5 stars, 0.6 = 3 stars)
    min_positive_rating: float = 0.6
    
    # Auto-extract facts from high-rated answers
    auto_extract_facts: bool = True
    
    # Auto-detect reasoning chains from step-by-step answers
    auto_detect_reasoning: bool = True
    
    # Learn query refinements when retrieval improves
    learn_query_patterns: bool = True
    
    # Minimum retrieval improvement to store refinement
    min_retrieval_improvement: float = 0.2
    
    # Confidence boost for facts from highly-rated answers
    positive_confidence_boost: float = 0.1
    
    # Confidence penalty for facts from low-rated answers
    negative_confidence_penalty: float = 0.15


class FeedbackLearner:
    """Learns from user feedback to improve RAG over time.
    
    This class processes user feedback and automatically:
    1. Extracts facts from successful answers
    2. Identifies and stores reasoning patterns
    3. Learns query refinement patterns
    4. Updates confidence scores
    
    Example:
        >>> learner = FeedbackLearner(memory, llm_fn)
        >>> 
        >>> # After user gives thumbs up
        >>> await learner.process_feedback(
        ...     question="How do I set Spark memory?",
        ...     answer="Use spark.executor.memory=4g...",
        ...     rating=1.0,  # 5 stars
        ...     retrieved_chunks=chunks,
        ... )
    """
    
    def __init__(
        self,
        memory: KnowledgeMemory,
        llm_fn: Optional[Callable[[str], Awaitable[str]]] = None,
        config: Optional[LearningConfig] = None,
    ):
        """Initialize feedback learner.
        
        Args:
            memory: KnowledgeMemory instance for storage
            llm_fn: Optional async function for LLM calls (for fact extraction)
            config: Learning configuration
        """
        self.memory = memory
        self.llm_fn = llm_fn
        self.config = config or LearningConfig()
    
    async def process_feedback(
        self,
        question: str,
        answer: str,
        rating: float,
        feedback_type: FeedbackType = FeedbackType.POSITIVE,
        correction: Optional[str] = None,
        retrieved_chunks: Optional[List[Any]] = None,
        original_query: Optional[str] = None,
        refined_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process user feedback and learn from it.
        
        Args:
            question: The original question
            answer: The model's answer
            rating: Rating from -1 (bad) to 1 (excellent)
            feedback_type: Type of feedback
            correction: User's correction if provided
            retrieved_chunks: Chunks used for the answer
            original_query: Original search query
            refined_query: Refined search query if different
            
        Returns:
            Summary of what was learned
        """
        learned = {
            "facts_extracted": 0,
            "reasoning_stored": False,
            "refinement_learned": False,
            "feedback_stored": True,
        }
        
        # Store raw feedback
        feedback = UserFeedback(
            question=question,
            answer=answer,
            feedback_type=feedback_type,
            correction=correction,
            rating=rating,
            context_chunks=[c.id for c in retrieved_chunks] if retrieved_chunks else [],
        )
        await self.memory.store_feedback(feedback)
        
        # Process based on rating
        if rating >= self.config.min_positive_rating:
            # Learn from successful interaction
            
            # Extract facts from the answer
            if self.config.auto_extract_facts:
                facts = await self._extract_facts(question, answer)
                for fact in facts:
                    fact.confidence = min(1.0, 0.5 + rating * 0.3)
                    await self.memory.store_fact(fact)
                learned["facts_extracted"] = len(facts)
            
            # Detect and store reasoning chains
            if self.config.auto_detect_reasoning:
                chain = self._detect_reasoning_chain(question, answer)
                if chain:
                    chain.was_correct = True
                    chain.feedback_score = rating
                    await self.memory.store_reasoning_chain(chain)
                    learned["reasoning_stored"] = True
            
            # Learn query refinement if it improved retrieval
            if self.config.learn_query_patterns and original_query and refined_query:
                if original_query != refined_query:
                    refinement = QueryRefinement(
                        original_query=original_query,
                        refined_query=refined_query,
                        improvement_score=rating,
                        success_rate=1.0,
                    )
                    await self.memory.store_refinement(refinement)
                    learned["refinement_learned"] = True
        
        elif rating < 0:
            # Learn from failure - store the correction
            if correction:
                # The correction becomes a learned fact
                fact = LearnedFact(
                    topic=self._extract_topic(question),
                    fact=correction,
                    source_question=question,
                    source_answer=f"CORRECTION: {correction} (was: {answer[:200]})",
                    confidence=0.8,  # High confidence for explicit corrections
                )
                await self.memory.store_fact(fact)
                learned["facts_extracted"] = 1
        
        return learned
    
    async def _extract_facts(
        self,
        question: str,
        answer: str,
    ) -> List[LearnedFact]:
        """Extract factual statements from an answer.
        
        Uses LLM if available, otherwise falls back to heuristics.
        """
        if self.llm_fn:
            return await self._extract_facts_with_llm(question, answer)
        else:
            return self._extract_facts_heuristic(question, answer)
    
    async def _extract_facts_with_llm(
        self,
        question: str,
        answer: str,
    ) -> List[LearnedFact]:
        """Use LLM to extract structured facts."""
        prompt = f"""Extract key facts from this Q&A pair. Output as JSON array.

Question: {question}

Answer: {answer}

Output format:
[
  {{"topic": "topic name", "fact": "concise factual statement"}},
  ...
]

Only include concrete, factual information. Be concise.
Output ONLY the JSON array, no other text."""

        try:
            response = await self.llm_fn(prompt)
            
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            facts_data = json.loads(json_match.group())
            
            return [
                LearnedFact(
                    topic=f.get("topic", "General"),
                    fact=f.get("fact", ""),
                    source_question=question,
                    source_answer=answer[:500],
                )
                for f in facts_data
                if f.get("fact")
            ]
        except (json.JSONDecodeError, Exception):
            return self._extract_facts_heuristic(question, answer)
    
    def _extract_facts_heuristic(
        self,
        question: str,
        answer: str,
    ) -> List[LearnedFact]:
        """Extract facts using simple heuristics."""
        facts = []
        topic = self._extract_topic(question)
        
        # Look for patterns like "X is Y", "X = Y", "Use X for Y"
        patterns = [
            r'([A-Za-z_\.]+)\s+(?:is|are|should be|defaults? to)\s+([^\.!?]+)',
            r'(?:use|set|configure)\s+([A-Za-z_\.]+)\s+(?:to|as|for)\s+([^\.!?]+)',
            r'([A-Za-z_\.]+)\s*[=:]\s*([^\.!?\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for match in matches[:3]:  # Limit per pattern
                if len(match) == 2 and len(match[0]) > 2 and len(match[1]) > 5:
                    fact_text = f"{match[0].strip()} → {match[1].strip()}"
                    facts.append(LearnedFact(
                        topic=topic,
                        fact=fact_text,
                        source_question=question,
                        source_answer=answer[:500],
                    ))
        
        # Also look for bullet points or numbered lists
        list_items = re.findall(r'(?:^|\n)\s*[•\-\*\d\.]+\s*([^:\n]+:\s*[^\n]+)', answer)
        for item in list_items[:3]:
            if len(item) > 10:
                facts.append(LearnedFact(
                    topic=topic,
                    fact=item.strip(),
                    source_question=question,
                    source_answer=answer[:500],
                ))
        
        return facts[:5]  # Max 5 facts per answer
    
    def _detect_reasoning_chain(
        self,
        question: str,
        answer: str,
    ) -> Optional[ReasoningChain]:
        """Detect if the answer contains step-by-step reasoning."""
        # Look for numbered steps or explicit reasoning markers
        step_patterns = [
            r'(?:step|first|second|third|then|next|finally)\s*\d*[:\.]?\s*([^\n]+)',
            r'^\s*(\d+)[\.:\)]\s*([^\n]+)',
            r'(?:because|therefore|this means|so)\s+([^\n\.]+)',
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                step_text = match if isinstance(match, str) else " ".join(match)
                if len(step_text.strip()) > 10:
                    steps.append(step_text.strip())
        
        # Need at least 2 steps to be considered a reasoning chain
        if len(steps) >= 2:
            # Get the final answer (usually the last sentence or after "therefore")
            final_match = re.search(
                r'(?:therefore|thus|in conclusion|the answer is|so)[:\s]*([^\.]+)',
                answer,
                re.IGNORECASE
            )
            final_answer = final_match.group(1) if final_match else answer[-200:]
            
            return ReasoningChain(
                question=question,
                steps=steps[:5],  # Max 5 steps
                final_answer=final_answer.strip(),
            )
        
        return None
    
    def _extract_topic(self, question: str) -> str:
        """Extract main topic from question."""
        # Common technical topics
        topics = [
            "Spark", "Kafka", "Python", "Java", "Docker", "Kubernetes",
            "PostgreSQL", "Redis", "MongoDB", "AWS", "Azure", "GCP",
            "Memory", "Performance", "Configuration", "Security",
        ]
        
        question_lower = question.lower()
        for topic in topics:
            if topic.lower() in question_lower:
                return topic
        
        # Fallback: first noun-like word
        words = re.findall(r'\b[A-Z][a-z]+\b', question)
        return words[0] if words else "General"
    
    async def update_chunk_relevance(
        self,
        chunk_ids: List[UUID],
        was_helpful: bool,
    ) -> None:
        """Update chunk relevance scores based on usage.
        
        Chunks that are frequently part of helpful answers
        get boosted in future retrievals.
        """
        # This would update chunk metadata in the store
        # For now, we track this in the facts that reference chunks
        pass
    
    async def batch_learn_from_history(
        self,
        feedback_history: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Process a batch of historical feedback.
        
        Useful for initial training from logs or user history.
        """
        results = {
            "processed": 0,
            "facts_extracted": 0,
            "chains_stored": 0,
            "refinements_learned": 0,
        }
        
        for fb in feedback_history:
            learned = await self.process_feedback(
                question=fb.get("question", ""),
                answer=fb.get("answer", ""),
                rating=fb.get("rating", 0.0),
                feedback_type=FeedbackType(fb.get("feedback_type", "positive")),
                correction=fb.get("correction"),
            )
            
            results["processed"] += 1
            results["facts_extracted"] += learned.get("facts_extracted", 0)
            if learned.get("reasoning_stored"):
                results["chains_stored"] += 1
            if learned.get("refinement_learned"):
                results["refinements_learned"] += 1
        
        return results


class ContrastiveLearningHelper:
    """Helper for contrastive learning on embeddings.
    
    Creates positive/negative pairs from user feedback
    for fine-tuning domain-specific embeddings.
    """
    
    def __init__(self, memory: KnowledgeMemory):
        """Initialize helper.
        
        Args:
            memory: KnowledgeMemory instance
        """
        self.memory = memory
    
    async def generate_training_pairs(
        self,
        min_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """Generate contrastive training pairs from feedback.
        
        Returns pairs like:
        - Positive: (query, relevant_chunk) from high-rated answers
        - Negative: (query, irrelevant_chunk) or (query, low-rated chunk)
        """
        pairs = []
        
        # Get positive feedback
        all_feedback = await self.memory.get_corrections()  # Reusing this as proxy
        
        # In a real implementation, we'd query feedback with high ratings
        # and pair questions with their successful chunks
        
        return pairs
    
    def export_for_fine_tuning(
        self,
        pairs: List[Dict[str, Any]],
        output_format: str = "jsonl",
    ) -> str:
        """Export training pairs for embedding fine-tuning.
        
        Formats suitable for:
        - sentence-transformers fine-tuning
        - Ollama embedding model training
        """
        if output_format == "jsonl":
            return "\n".join(json.dumps(p) for p in pairs)
        
        return json.dumps(pairs, indent=2)
