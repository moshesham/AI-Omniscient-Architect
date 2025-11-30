"""Fixed-size token-based chunker."""

from typing import List, Optional
from uuid import uuid4

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from .base import BaseChunker
from ..models import Document, Chunk


class FixedChunker(BaseChunker):
    """Token-based chunker with fixed size and overlap.
    
    Splits documents into chunks of approximately equal token count,
    with configurable overlap between consecutive chunks.
    
    Uses tiktoken for accurate token counting when available,
    falls back to character-based approximation otherwise.
    
    Example:
        >>> chunker = FixedChunker(chunk_size=512, chunk_overlap=0.1)
        >>> chunks = chunker.chunk(document)
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: float = 0.1,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize fixed chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (256, 512, 1024)
            chunk_overlap: Overlap ratio between chunks (0.1 = 10%)
            encoding_name: Tiktoken encoding name for tokenization
        """
        super().__init__(chunk_size, chunk_overlap)
        self.encoding_name = encoding_name
        self._encoder: Optional[object] = None
        
        if HAS_TIKTOKEN:
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                pass
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: ~4 chars per token
        return len(text) // 4
    
    def _tokenize(self, text: str) -> List[int]:
        """Convert text to tokens."""
        if self._encoder:
            return self._encoder.encode(text)
        # Fallback: split by chars, 4 at a time
        return list(range(0, len(text), 4))
    
    def _detokenize(self, tokens: List[int], original_text: str) -> str:
        """Convert tokens back to text."""
        if self._encoder:
            return self._encoder.decode(tokens)
        # Fallback: slice original text
        start = tokens[0] if tokens else 0
        end = tokens[-1] + 4 if tokens else 0
        return original_text[start:end]
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks with approximately chunk_size tokens each
        """
        text = document.content
        if not text.strip():
            return []
        
        chunks = []
        
        if self._encoder:
            # Token-based chunking with tiktoken
            chunks = self._chunk_with_tokens(document)
        else:
            # Character-based fallback
            chunks = self._chunk_with_chars(document)
        
        return chunks
    
    def _chunk_with_tokens(self, document: Document) -> List[Chunk]:
        """Chunk using tiktoken tokenization."""
        tokens = self._encoder.encode(document.content)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_size:
            # Document fits in single chunk
            return [
                Chunk(
                    content=document.content,
                    document_id=document.id,
                    metadata={
                        "source": document.source,
                        "type": "text",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "strategy": "fixed",
                    },
                    start_char=0,
                    end_char=len(document.content),
                )
            ]
        
        chunks = []
        start_token = 0
        chunk_index = 0
        step = self.chunk_size - self._overlap_tokens
        
        while start_token < total_tokens:
            end_token = min(start_token + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self._encoder.decode(chunk_tokens)
            
            # Calculate approximate character positions
            prefix_text = self._encoder.decode(tokens[:start_token])
            start_char = len(prefix_text)
            end_char = start_char + len(chunk_text)
            
            chunks.append(
                Chunk(
                    content=chunk_text,
                    document_id=document.id,
                    metadata={
                        "source": document.source,
                        "type": "text",
                        "chunk_index": chunk_index,
                        "token_start": start_token,
                        "token_end": end_token,
                        "strategy": "fixed",
                    },
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            
            start_token += step
            chunk_index += 1
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def _chunk_with_chars(self, document: Document) -> List[Chunk]:
        """Fallback character-based chunking."""
        text = document.content
        # Approximate: 4 chars per token
        char_size = self.chunk_size * 4
        char_overlap = self._overlap_tokens * 4
        step = char_size - char_overlap
        
        if len(text) <= char_size:
            return [
                Chunk(
                    content=text,
                    document_id=document.id,
                    metadata={
                        "source": document.source,
                        "type": "text",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "strategy": "fixed",
                    },
                    start_char=0,
                    end_char=len(text),
                )
            ]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + char_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Look for space within last 10% of chunk
                search_start = max(start, end - char_size // 10)
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space + 1
            
            chunk_text = text[start:end]
            
            chunks.append(
                Chunk(
                    content=chunk_text,
                    document_id=document.id,
                    metadata={
                        "source": document.source,
                        "type": "text",
                        "chunk_index": chunk_index,
                        "strategy": "fixed",
                    },
                    start_char=start,
                    end_char=end,
                )
            )
            
            start = end - char_overlap if end < len(text) else len(text)
            chunk_index += 1
        
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
