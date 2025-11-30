"""Chunker factory for creating chunkers by strategy name."""

from typing import Optional, Dict, Type

from .base import BaseChunker
from .fixed import FixedChunker
from .semantic import SemanticChunker
from .ast_chunker import ASTChunker
from ..models import ChunkingStrategy, Document


class AutoChunker(BaseChunker):
    """Auto-detecting chunker that selects strategy based on file type.
    
    Uses:
    - SemanticChunker for markdown, text, documentation
    - ASTChunker for source code (Python, JavaScript, etc.)
    - FixedChunker for everything else
    """
    
    # Extension to strategy mapping
    EXTENSION_MAP = {
        # Documentation
        'md': 'semantic',
        'markdown': 'semantic',
        'rst': 'semantic',
        'txt': 'semantic',
        'adoc': 'semantic',
        # Source code
        'py': 'ast',
        'pyw': 'ast',
        'js': 'ast',
        'jsx': 'ast',
        'ts': 'ast',
        'tsx': 'ast',
        'mjs': 'ast',
        # Default to fixed for others
    }
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: float = 0.1):
        super().__init__(chunk_size, chunk_overlap)
        self._chunkers = {
            'fixed': FixedChunker(chunk_size, chunk_overlap),
            'semantic': SemanticChunker(chunk_size, chunk_overlap),
            'ast': ASTChunker(chunk_size, chunk_overlap),
        }
    
    def chunk(self, document: Document):
        """Auto-select chunker based on document type and chunk."""
        ext = document.file_extension or ''
        strategy = self.EXTENSION_MAP.get(ext, 'fixed')
        chunker = self._chunkers[strategy]
        
        chunks = chunker.chunk(document)
        
        # Add auto-detection info to metadata
        for chunk in chunks:
            chunk.metadata['auto_detected_strategy'] = strategy
        
        return chunks


class ChunkerFactory:
    """Factory for creating chunkers by strategy name.
    
    Example:
        >>> chunker = ChunkerFactory.create("semantic", chunk_size=512)
        >>> chunks = chunker.chunk(document)
    """
    
    _registry: Dict[str, Type[BaseChunker]] = {
        'fixed': FixedChunker,
        'semantic': SemanticChunker,
        'ast': ASTChunker,
        'auto': AutoChunker,
    }
    
    @classmethod
    def create(
        cls,
        strategy: str | ChunkingStrategy,
        chunk_size: int = 512,
        chunk_overlap: float = 0.1,
        **kwargs,
    ) -> BaseChunker:
        """Create a chunker instance.
        
        Args:
            strategy: Chunking strategy name or enum
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap ratio between chunks
            **kwargs: Additional strategy-specific options
            
        Returns:
            Configured chunker instance
            
        Raises:
            ValueError: If strategy is not recognized
        """
        if isinstance(strategy, ChunkingStrategy):
            strategy = strategy.value
        
        strategy = strategy.lower()
        
        if strategy not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown chunking strategy: {strategy}. Available: {available}")
        
        chunker_class = cls._registry[strategy]
        return chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    
    @classmethod
    def register(cls, name: str, chunker_class: Type[BaseChunker]) -> None:
        """Register a custom chunker.
        
        Args:
            name: Strategy name
            chunker_class: Chunker class to register
        """
        cls._registry[name.lower()] = chunker_class
    
    @classmethod
    def available_strategies(cls) -> list[str]:
        """Get list of available strategy names."""
        return list(cls._registry.keys())
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, str]:
        """Get descriptions of available strategies."""
        return {
            'fixed': "Token-based chunking with configurable size and overlap. Best for general text.",
            'semantic': "Structure-aware chunking by headings, paragraphs, code blocks. Best for documentation.",
            'ast': "AST-based chunking by functions and classes. Best for source code.",
            'auto': "Auto-detect based on file extension. Recommended for mixed content.",
        }
