"""Context injection for knowledge-augmented prompting.

This module injects learned knowledge into prompts at the start
of each Ollama session, giving the model "memory" of past learning.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .knowledge_memory import (
    KnowledgeMemory,
    LearnedFact,
    ReasoningChain,
    UserFeedback,
    QueryRefinement,
)


@dataclass
class InjectionConfig:
    """Configuration for context injection."""
    max_facts: int = 5
    max_reasoning_chains: int = 2
    max_corrections: int = 3
    include_topic_header: bool = True
    include_confidence_scores: bool = False
    include_few_shot_reasoning: bool = True
    min_fact_confidence: float = 0.4


class ContextInjector:
    """Injects learned knowledge into prompts.
    
    This class formats learned knowledge into prompt sections
    that can be prepended to user queries, giving the model
    access to knowledge learned in previous sessions.
    
    Example:
        >>> injector = ContextInjector(memory)
        >>> 
        >>> # Get context to inject
        >>> context = await injector.build_context(
        ...     query="How do I configure Spark memory?",
        ...     topic="spark"
        ... )
        >>> 
        >>> # Use in prompt
        >>> prompt = f"{context}\\n\\nUser Question: {query}"
    """
    
    def __init__(
        self,
        memory: KnowledgeMemory,
        config: Optional[InjectionConfig] = None,
    ):
        """Initialize context injector.
        
        Args:
            memory: KnowledgeMemory instance
            config: Injection configuration
        """
        self.memory = memory
        self.config = config or InjectionConfig()
    
    async def build_context(
        self,
        query: str,
        topic: Optional[str] = None,
        include_reasoning: bool = True,
    ) -> str:
        """Build context string to inject into prompt.
        
        Args:
            query: The user's current query
            topic: Optional topic to filter knowledge
            include_reasoning: Whether to include reasoning examples
            
        Returns:
            Formatted context string
        """
        # Retrieve relevant knowledge
        knowledge = await self.memory.get_relevant_knowledge(
            query=query,
            max_facts=self.config.max_facts,
            max_chains=self.config.max_reasoning_chains if include_reasoning else 0,
            include_corrections=True,
        )
        
        sections = []
        
        # Add learned facts
        if knowledge["facts"]:
            facts_section = self._format_facts(
                knowledge["facts"],
                topic=topic,
            )
            if facts_section:
                sections.append(facts_section)
        
        # Add corrections (things the model got wrong before)
        if knowledge["corrections"]:
            corrections_section = self._format_corrections(knowledge["corrections"])
            if corrections_section:
                sections.append(corrections_section)
        
        # Add reasoning examples (few-shot)
        if include_reasoning and knowledge["reasoning_chains"]:
            reasoning_section = self._format_reasoning_chains(
                knowledge["reasoning_chains"]
            )
            if reasoning_section:
                sections.append(reasoning_section)
        
        if not sections:
            return ""
        
        # Combine all sections
        header = "# Relevant Knowledge from Previous Sessions\n\n" if self.config.include_topic_header else ""
        return header + "\n\n".join(sections) + "\n\n---\n"
    
    def _format_facts(
        self,
        facts: List[LearnedFact],
        topic: Optional[str] = None,
    ) -> str:
        """Format learned facts section."""
        # Filter by topic if specified
        if topic:
            facts = [f for f in facts if topic.lower() in f.topic.lower()]
        
        # Filter by confidence
        facts = [f for f in facts if f.confidence >= self.config.min_fact_confidence]
        
        if not facts:
            return ""
        
        lines = ["## Previously Learned Facts\n"]
        
        for fact in facts:
            if self.config.include_confidence_scores:
                lines.append(f"• [{fact.confidence:.0%}] **{fact.topic}**: {fact.fact}")
            else:
                lines.append(f"• **{fact.topic}**: {fact.fact}")
        
        return "\n".join(lines)
    
    def _format_corrections(self, corrections: List[UserFeedback]) -> str:
        """Format corrections section."""
        if not corrections:
            return ""
        
        lines = ["## Important Corrections\n"]
        lines.append("*The following are corrections from previous sessions. Use the corrected information.*\n")
        
        for corr in corrections[:self.config.max_corrections]:
            lines.append(f"❌ **Previous (incorrect)**: {corr.answer[:200]}...")
            lines.append(f"✅ **Correct**: {corr.correction}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_reasoning_chains(self, chains: List[ReasoningChain]) -> str:
        """Format reasoning chains as few-shot examples."""
        if not chains or not self.config.include_few_shot_reasoning:
            return ""
        
        lines = ["## Example Reasoning Patterns\n"]
        lines.append("*Use similar step-by-step reasoning for your answer.*\n")
        
        for chain in chains:
            lines.append(f"**Example Question**: {chain.question}")
            lines.append("**Reasoning Steps**:")
            for i, step in enumerate(chain.steps, 1):
                lines.append(f"  {i}. {step}")
            lines.append(f"**Answer**: {chain.final_answer}")
            lines.append("")
        
        return "\n".join(lines)
    
    async def get_refined_query(self, query: str) -> str:
        """Get refined version of query if available.
        
        Checks if we've learned a better way to phrase this query
        for improved retrieval.
        """
        knowledge = await self.memory.get_relevant_knowledge(query)
        
        if knowledge.get("query_refinement"):
            refinement = knowledge["query_refinement"]
            return refinement.refined_query
        
        return query
    
    async def build_system_prompt_additions(self) -> str:
        """Build additions to the system prompt based on learned knowledge.
        
        This provides high-level guidance learned from user feedback.
        """
        stats = await self.memory.get_statistics()
        
        if stats["total_facts"] == 0:
            return ""
        
        # Get high-confidence facts across all topics
        all_facts = await self.memory.get_relevant_facts(
            query="",  # Empty query to get top facts by confidence
            top_k=10,
            min_confidence=0.7,
        )
        
        if not all_facts:
            return ""
        
        # Group facts by topic
        topics = {}
        for fact in all_facts:
            if fact.topic not in topics:
                topics[fact.topic] = []
            topics[fact.topic].append(fact)
        
        lines = ["\n## Domain Knowledge\n"]
        lines.append("You have learned the following from previous interactions:\n")
        
        for topic, facts in topics.items():
            lines.append(f"\n### {topic}")
            for fact in facts:
                lines.append(f"• {fact.fact}")
        
        return "\n".join(lines)


class AdaptivePromptBuilder:
    """Builds prompts that adapt based on learned knowledge.
    
    This class creates prompts that:
    1. Include relevant facts from memory
    2. Use learned reasoning patterns
    3. Apply query refinements
    4. Avoid known mistakes
    """
    
    def __init__(
        self,
        injector: ContextInjector,
        base_system_prompt: str = "",
    ):
        """Initialize prompt builder.
        
        Args:
            injector: ContextInjector instance
            base_system_prompt: Base system prompt to augment
        """
        self.injector = injector
        self.base_system_prompt = base_system_prompt
    
    async def build_prompt(
        self,
        query: str,
        retrieved_context: str = "",
        topic: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build a complete prompt with injected knowledge.
        
        Args:
            query: User's query
            retrieved_context: RAG-retrieved context
            topic: Optional topic filter
            
        Returns:
            Dict with 'system' and 'user' prompt components
        """
        # Get refined query if available
        refined_query = await self.injector.get_refined_query(query)
        
        # Build knowledge context
        knowledge_context = await self.injector.build_context(
            query=refined_query,
            topic=topic,
        )
        
        # Get system prompt additions
        system_additions = await self.injector.build_system_prompt_additions()
        
        # Build system prompt
        system_prompt = self.base_system_prompt
        if system_additions:
            system_prompt += system_additions
        
        # Build user prompt with all context
        user_prompt_parts = []
        
        if knowledge_context:
            user_prompt_parts.append(knowledge_context)
        
        if retrieved_context:
            user_prompt_parts.append("## Retrieved Context\n")
            user_prompt_parts.append(retrieved_context)
            user_prompt_parts.append("")
        
        user_prompt_parts.append("## Current Question\n")
        user_prompt_parts.append(query)
        
        if refined_query != query:
            user_prompt_parts.append(f"\n(Refined query for better retrieval: {refined_query})")
        
        return {
            "system": system_prompt,
            "user": "\n".join(user_prompt_parts),
            "refined_query": refined_query,
        }
