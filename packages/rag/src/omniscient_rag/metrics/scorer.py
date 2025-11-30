"""Knowledge scorer for evaluating RAG system quality."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from uuid import UUID
import re

from ..models import KnowledgeQuestion, KnowledgeScore, RetrievalResult
from ..store.postgres import PostgresVectorStore
from ..search.hybrid import HybridSearcher


@dataclass
class ScoringConfig:
    """Configuration for knowledge scoring.
    
    Attributes:
        questions_per_eval: Number of questions to use per evaluation
        retrieval_top_k: Top-k for retrieval during evaluation
        similarity_threshold: Minimum similarity for "relevant" classification
        use_llm_scoring: Use LLM for answer quality scoring (vs. heuristic)
    """
    questions_per_eval: int = 10
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.3
    use_llm_scoring: bool = True


# Prompt for LLM-based answer scoring
ANSWER_SCORING_PROMPT = """Evaluate how well the retrieved context answers the question.

Question: {question}
Expected Answer: {expected_answer}

Retrieved Context:
---
{context}
---

Score the answer quality from 0-100 based on:
- Relevance: Does the context address the question?
- Completeness: Does it contain the expected information?
- Accuracy: Is the information correct?

Respond with a JSON object:
{{"score": <0-100>, "explanation": "<brief explanation>"}}
"""


class KnowledgeScorer:
    """Evaluate RAG system knowledge quality.
    
    Runs evaluation on two triggers:
    1. On Ingestion: Auto-generates questions from new documents
    2. On Demand: Runs retrieval tests against stored questions
    
    Metrics tracked:
    - Retrieval Precision: Fraction of retrieved chunks that are relevant
    - Answer Accuracy: Quality of answers (0-100 scale)
    - Coverage Ratio: Fraction of knowledge base used
    
    Example:
        >>> scorer = KnowledgeScorer(store, searcher, llm_fn)
        >>> score = await scorer.evaluate()
        >>> print(f"Knowledge Score: {score.overall_score:.1f}%")
    """
    
    def __init__(
        self,
        store: PostgresVectorStore,
        searcher: HybridSearcher,
        llm_fn: Optional[Callable[[str], Any]] = None,
        config: Optional[ScoringConfig] = None,
    ):
        """Initialize knowledge scorer.
        
        Args:
            store: Vector store with questions and scores
            searcher: Hybrid searcher for retrieval
            llm_fn: Optional LLM function for answer scoring
            config: Scoring configuration
        """
        self.store = store
        self.searcher = searcher
        self.llm_fn = llm_fn
        self.config = config or ScoringConfig()
    
    async def evaluate(
        self,
        document_id: Optional[UUID] = None,
        topic: Optional[str] = None,
    ) -> KnowledgeScore:
        """Run knowledge evaluation.
        
        Args:
            document_id: Evaluate specific document only
            topic: Evaluate specific topic only
            
        Returns:
            KnowledgeScore with metrics
        """
        # Get test questions
        questions = await self.store.get_questions(
            document_id=document_id,
            topic=topic,
            limit=self.config.questions_per_eval,
        )
        
        if not questions:
            return KnowledgeScore(
                retrieval_precision=0.0,
                answer_accuracy=0.0,
                coverage_ratio=0.0,
                questions_evaluated=0,
                details={"error": "No test questions available"},
            )
        
        # Evaluate each question
        results = []
        retrieved_chunks = set()
        
        for question in questions:
            result = await self._evaluate_question(question)
            results.append(result)
            
            # Track unique chunks for coverage
            for chunk_id in result.get("chunk_ids", []):
                retrieved_chunks.add(chunk_id)
        
        # Calculate aggregate metrics
        precision_scores = [r["precision"] for r in results]
        accuracy_scores = [r["accuracy"] for r in results]
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        
        # Calculate coverage (unique chunks / total chunks)
        stats = await self.store.get_stats()
        total_chunks = stats.get("chunks_with_embeddings", 1)
        coverage = len(retrieved_chunks) / total_chunks if total_chunks > 0 else 0
        
        score = KnowledgeScore(
            retrieval_precision=avg_precision,
            answer_accuracy=avg_accuracy,
            coverage_ratio=coverage,
            questions_evaluated=len(questions),
            details={
                "per_question": results,
                "unique_chunks_retrieved": len(retrieved_chunks),
                "total_chunks": total_chunks,
            },
        )
        
        # Store the score
        await self.store.insert_score(score)
        
        return score
    
    async def _evaluate_question(self, question: KnowledgeQuestion) -> Dict[str, Any]:
        """Evaluate a single question.
        
        Args:
            question: Test question to evaluate
            
        Returns:
            Dict with precision, accuracy, and details
        """
        # Retrieve relevant chunks
        results = await self.searcher.search(
            query=question.question,
            top_k=self.config.retrieval_top_k,
        )
        
        # Calculate retrieval precision
        relevant_count = sum(
            1 for r in results
            if r.combined_score >= self.config.similarity_threshold
        )
        precision = relevant_count / len(results) if results else 0
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[{r.source}]\n{r.chunk.content}"
            for r in results
        ])
        
        # Score answer quality
        if self.config.use_llm_scoring and self.llm_fn:
            accuracy = await self._score_with_llm(question, context)
        else:
            accuracy = self._score_heuristic(question, context)
        
        return {
            "question_id": str(question.id),
            "question": question.question,
            "precision": precision,
            "accuracy": accuracy,
            "chunks_retrieved": len(results),
            "chunk_ids": [str(r.chunk.id) for r in results],
        }
    
    async def _score_with_llm(
        self,
        question: KnowledgeQuestion,
        context: str,
    ) -> float:
        """Score answer quality using LLM."""
        prompt = ANSWER_SCORING_PROMPT.format(
            question=question.question,
            expected_answer=question.expected_answer,
            context=context[:2000],  # Truncate for prompt limits
        )
        
        try:
            response = await self.llm_fn(prompt)
            
            # Parse JSON response
            import json
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                return min(100, max(0, score))
        except Exception:
            pass
        
        # Fallback to heuristic
        return self._score_heuristic(question, context)
    
    def _score_heuristic(
        self,
        question: KnowledgeQuestion,
        context: str,
    ) -> float:
        """Score answer quality using keyword matching heuristic.
        
        Compares expected answer keywords against retrieved context.
        """
        if not context:
            return 0.0
        
        # Extract keywords from expected answer
        expected_lower = question.expected_answer.lower()
        context_lower = context.lower()
        
        # Tokenize expected answer
        keywords = re.findall(r'\b\w{3,}\b', expected_lower)
        keywords = [k for k in keywords if k not in STOP_WORDS]
        
        if not keywords:
            # No meaningful keywords, check if any overlap
            return 50.0 if any(w in context_lower for w in expected_lower.split()) else 0.0
        
        # Count keyword matches
        matches = sum(1 for k in keywords if k in context_lower)
        match_ratio = matches / len(keywords)
        
        # Scale to 0-100
        return min(100, match_ratio * 100)
    
    async def get_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get knowledge score trend over time.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of score summaries by date
        """
        scores = await self.store.get_score_history(limit=days * 3)
        
        # Group by date
        by_date: Dict[str, List[KnowledgeScore]] = {}
        for score in scores:
            date_str = score.timestamp.strftime("%Y-%m-%d")
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(score)
        
        # Calculate daily averages
        trend = []
        for date_str, day_scores in sorted(by_date.items()):
            avg_overall = sum(s.overall_score for s in day_scores) / len(day_scores)
            avg_precision = sum(s.retrieval_precision for s in day_scores) / len(day_scores)
            avg_accuracy = sum(s.answer_accuracy for s in day_scores) / len(day_scores)
            
            trend.append({
                "date": date_str,
                "overall_score": avg_overall,
                "retrieval_precision": avg_precision,
                "answer_accuracy": avg_accuracy,
                "evaluations": len(day_scores),
            })
        
        return trend


# Common stop words to filter
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "you",
    "your", "he", "she", "him", "her", "his", "which", "what", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "not", "only", "same", "than", "too",
    "very", "just", "also", "now", "here", "there", "then", "so", "if",
}
