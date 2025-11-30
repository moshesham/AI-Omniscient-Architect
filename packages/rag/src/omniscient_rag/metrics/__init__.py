"""Knowledge metrics for tracking RAG system quality.

Provides:
- KnowledgeScorer: Evaluate retrieval and answer quality
- QuestionGenerator: Auto-generate test questions from documents
"""

from .scorer import KnowledgeScorer, ScoringConfig
from .questions import QuestionGenerator

__all__ = [
    "KnowledgeScorer",
    "ScoringConfig",
    "QuestionGenerator",
]
