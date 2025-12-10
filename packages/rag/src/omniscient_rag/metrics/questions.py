"""Question generator for knowledge evaluation."""

import re
import structlog
from typing import List, Optional, Callable, Any, Union, TYPE_CHECKING
from uuid import UUID

from ..models import Document, KnowledgeQuestion

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    try:
        from omniscient_llm.base import BaseLLMProvider  # type: ignore
    except Exception:  # pragma: no cover - optional import for typing only
        BaseLLMProvider = Any  # type: ignore


# Prompt template for question generation
QUESTION_GENERATION_PROMPT = """Based on the following document content, generate {num_questions} test questions that would verify understanding of the key concepts.

For each question:
1. Focus on factual information that can be verified from the document
2. Include the expected answer
3. Assign a topic category
4. Rate difficulty as easy/medium/hard

Document Source: {source}
---
{content}
---

Generate questions in this exact JSON format:
[
  {{
    "question": "What is...",
    "expected_answer": "The answer is...",
    "topic": "configuration",
    "difficulty": "medium"
  }}
]

Generate {num_questions} questions:"""


class QuestionGenerator:
    """Generate test questions from documents for knowledge evaluation.
    
    Uses an LLM to generate questions that test understanding of document content.
    Questions are used during knowledge evaluation to measure retrieval quality.
    
    Example:
        >>> generator = QuestionGenerator(llm_fn)
        >>> questions = await generator.generate(document, num_questions=3)
    """
    
    def __init__(
        self,
        llm_fn: Union[Callable[[str], Any], "BaseLLMProvider"],  # async function or provider
        questions_per_doc: int = 3,
    ):
        """Initialize question generator.
        
        Args:
            llm_fn: Async function that takes a prompt and returns response text
            questions_per_doc: Default number of questions per document
        """
        self.llm_fn = llm_fn
        self.questions_per_doc = questions_per_doc
    
    async def generate(
        self,
        document: Document,
        num_questions: Optional[int] = None,
    ) -> List[KnowledgeQuestion]:
        """Generate test questions for a document.
        
        Args:
            document: Document to generate questions for
            num_questions: Number of questions (default: questions_per_doc)
            
        Returns:
            List of KnowledgeQuestion objects
        """
        num_questions = num_questions or self.questions_per_doc
        
        # Truncate content if too long (keep first ~2000 chars for prompt)
        content = document.content
        if len(content) > 3000:
            content = content[:3000] + "\n...[truncated]..."
        
        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            source=document.source,
            content=content,
        )
        
        try:
            logger.debug(f"Generating questions for document {document.source}")
            response = await self.llm_fn(prompt)
            logger.debug(f"LLM response length: {len(response) if response else 0}")
            questions = self._parse_questions(response, document.id)
            logger.debug(f"Parsed {len(questions)} questions")
            return questions[:num_questions]
        except Exception as e:
            # Return empty list on failure
            logger.warning(f"Question generation failed: {type(e).__name__}: {str(e)}")
            return []
    
    async def generate_batch(
        self,
        documents: List[Document],
        num_questions: Optional[int] = None,
    ) -> List[KnowledgeQuestion]:
        """Generate questions for multiple documents.
        
        Args:
            documents: Documents to process
            num_questions: Questions per document
            
        Returns:
            Combined list of all generated questions
        """
        all_questions = []
        for doc in documents:
            questions = await self.generate(doc, num_questions)
            all_questions.extend(questions)
        return all_questions
    
    def _parse_questions(
        self,
        response: str,
        document_id: UUID,
    ) -> List[KnowledgeQuestion]:
        """Parse LLM response into question objects."""
        questions = []
        
        # Try to extract JSON array from response
        try:
            import json
            
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                data = json.loads(json_match.group())
                
                for item in data:
                    if isinstance(item, dict) and "question" in item:
                        questions.append(KnowledgeQuestion(
                            question=item.get("question", ""),
                            expected_answer=item.get("expected_answer", ""),
                            document_id=document_id,
                            topic=item.get("topic", "general"),
                            difficulty=item.get("difficulty", "medium"),
                        ))
        except (json.JSONDecodeError, Exception):
            # Fallback: try to parse line-by-line
            questions = self._parse_questions_fallback(response, document_id)
        
        return questions
    
    def _parse_questions_fallback(
        self,
        response: str,
        document_id: UUID,
    ) -> List[KnowledgeQuestion]:
        """Fallback parsing for non-JSON responses."""
        questions = []
        
        # Look for Q: ... A: ... patterns
        qa_pattern = re.compile(
            r'(?:Q:|Question:)\s*(.+?)\s*(?:A:|Answer:|Expected:)\s*(.+?)(?=(?:Q:|Question:)|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in qa_pattern.finditer(response):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            
            if question and answer:
                questions.append(KnowledgeQuestion(
                    question=question,
                    expected_answer=answer,
                    document_id=document_id,
                ))
        
        return questions


class SimpleQuestionGenerator:
    """Simple rule-based question generator (no LLM required).
    
    Generates basic questions based on document structure:
    - Heading-based: "What does [heading] cover?"
    - Code-based: "What does the function [name] do?"
    - Definition-based: "What is [term]?" for defined terms
    
    Useful as a fallback when LLM is unavailable.
    """
    
    def __init__(self, questions_per_doc: int = 3):
        """Initialize simple generator.
        
        Args:
            questions_per_doc: Max questions per document
        """
        self.questions_per_doc = questions_per_doc
    
    def generate(self, document: Document) -> List[KnowledgeQuestion]:
        """Generate questions from document structure.
        
        Args:
            document: Document to analyze
            
        Returns:
            List of generated questions
        """
        questions = []
        content = document.content
        
        # Extract headings (markdown)
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        for heading in headings[:2]:
            questions.append(KnowledgeQuestion(
                question=f"What does the section '{heading}' cover?",
                expected_answer=f"Content related to {heading}",
                document_id=document.id,
                topic="structure",
                difficulty="easy",
            ))
        
        # Extract function definitions (Python)
        functions = re.findall(r'def\s+(\w+)\s*\(', content)
        for func in functions[:2]:
            questions.append(KnowledgeQuestion(
                question=f"What does the function '{func}' do?",
                expected_answer=f"Implementation of {func}",
                document_id=document.id,
                topic="code",
                difficulty="medium",
            ))
        
        # Extract class definitions
        classes = re.findall(r'class\s+(\w+)', content)
        for cls in classes[:2]:
            questions.append(KnowledgeQuestion(
                question=f"What is the purpose of the class '{cls}'?",
                expected_answer=f"The {cls} class provides...",
                document_id=document.id,
                topic="code",
                difficulty="medium",
            ))
        
        return questions[:self.questions_per_doc]
