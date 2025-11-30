"""Base chunker interface."""

from abc import ABC, abstractmethod
from typing import List

from ..models import Document, Chunk


class BaseChunker(ABC):
    """Abstract base class for document chunkers.
    
    Implement this interface to create custom chunking strategies.
    
    Example:
        >>> class MyChunker(BaseChunker):
        ...     def chunk(self, document: Document) -> List[Chunk]:
        ...         # Custom chunking logic
        ...         pass
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: float = 0.1):
        """Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap ratio between chunks (0-1)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._overlap_tokens = int(chunk_size * chunk_overlap)
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects with content and metadata
        """
        pass
    
    def chunk_batch(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            Flat list of all chunks from all documents
        """
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
    
    @property
    def strategy_name(self) -> str:
        """Get the strategy name for this chunker."""
        return self.__class__.__name__.replace("Chunker", "").lower()
