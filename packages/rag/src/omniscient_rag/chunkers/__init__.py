"""Chunking strategies for document processing.

Provides pluggable chunking strategies:
- FixedChunker: Token-based with configurable size/overlap
- SemanticChunker: Split by headings, paragraphs, code blocks
- ASTChunker: Parse by function/class boundaries (Python/JS)
- ChunkerFactory: Create chunker by strategy name
"""

from .base import BaseChunker
from .fixed import FixedChunker
from .semantic import SemanticChunker
from .ast_chunker import ASTChunker
from .factory import ChunkerFactory

__all__ = [
    "BaseChunker",
    "FixedChunker",
    "SemanticChunker",
    "ASTChunker",
    "ChunkerFactory",
]
