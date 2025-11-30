"""Machine Learning enhancements for RAG.

This module provides:
- Knowledge persistence across sessions
- Learning from user feedback
- Query refinement learning
- Reasoning chain storage
"""

from .knowledge_memory import (
    KnowledgeMemory,
    LearnedFact,
    ReasoningChain,
    QueryRefinement,
    UserFeedback,
    FeedbackType,
)
from .context_injector import ContextInjector, InjectionConfig, AdaptivePromptBuilder
from .feedback_learner import FeedbackLearner, LearningConfig

__all__ = [
    "KnowledgeMemory",
    "LearnedFact",
    "ReasoningChain",
    "QueryRefinement",
    "UserFeedback",
    "FeedbackType",
    "ContextInjector",
    "InjectionConfig",
    "AdaptivePromptBuilder",
    "FeedbackLearner",
    "LearningConfig",
]
