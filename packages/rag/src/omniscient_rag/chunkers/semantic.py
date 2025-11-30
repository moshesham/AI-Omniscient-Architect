"""Semantic chunker - splits by document structure."""

import re
from typing import List, Tuple
from uuid import uuid4

from .base import BaseChunker
from ..models import Document, Chunk


class SemanticChunker(BaseChunker):
    """Semantic chunker that splits by document structure.
    
    Recognizes and splits on:
    - Markdown headings (# ## ###)
    - Code blocks (``` or indented)
    - Paragraph breaks (double newlines)
    - Docstrings and comments
    
    Best for documentation, README files, and markdown content.
    
    Example:
        >>> chunker = SemanticChunker(chunk_size=512)
        >>> chunks = chunker.chunk(document)
    """
    
    # Patterns for semantic splitting
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\w]*\n(.*?)\n```', re.DOTALL)
    DOCSTRING_PATTERN = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')')
    PARAGRAPH_BREAK = re.compile(r'\n\s*\n')
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: float = 0.1,
        respect_code_blocks: bool = True,
        min_chunk_size: int = 50,
    ):
        """Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap ratio (applied at section boundaries)
            respect_code_blocks: Keep code blocks intact when possible
            min_chunk_size: Minimum chunk size (merge smaller chunks)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.respect_code_blocks = respect_code_blocks
        self.min_chunk_size = min_chunk_size
        # Approximate chars per token
        self._chars_per_token = 4
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document by semantic structure.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of semantically coherent chunks
        """
        text = document.content
        if not text.strip():
            return []
        
        # Step 1: Identify semantic sections
        sections = self._extract_sections(text)
        
        # Step 2: Group sections into appropriately sized chunks
        chunks = self._group_sections(sections, document)
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract semantic sections from text.
        
        Returns:
            List of (section_type, content, start_char, end_char) tuples
        """
        sections = []
        
        # Find all headings
        heading_positions = []
        for match in self.HEADING_PATTERN.finditer(text):
            level = len(match.group(1))
            heading_positions.append((match.start(), match.end(), f"heading_{level}", match.group(0)))
        
        # Find all code blocks
        code_positions = []
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            code_positions.append((match.start(), match.end(), "code_block", match.group(0)))
        
        # Merge and sort all special sections
        all_special = heading_positions + code_positions
        all_special.sort(key=lambda x: x[0])
        
        # Build sections by splitting on special markers and paragraphs
        if not all_special:
            # No special sections, split by paragraphs
            sections = self._split_by_paragraphs(text, 0)
        else:
            current_pos = 0
            for start, end, section_type, content in all_special:
                # Add text before this special section
                if start > current_pos:
                    before_text = text[current_pos:start]
                    sections.extend(self._split_by_paragraphs(before_text, current_pos))
                
                # Add the special section
                sections.append((section_type, content, start, end))
                current_pos = end
            
            # Add remaining text
            if current_pos < len(text):
                remaining = text[current_pos:]
                sections.extend(self._split_by_paragraphs(remaining, current_pos))
        
        return sections
    
    def _split_by_paragraphs(self, text: str, offset: int) -> List[Tuple[str, str, int, int]]:
        """Split text into paragraphs."""
        sections = []
        paragraphs = self.PARAGRAPH_BREAK.split(text)
        
        current_pos = offset
        for para in paragraphs:
            para = para.strip()
            if para:
                start = text.find(para, current_pos - offset) + offset if current_pos > offset else offset
                end = start + len(para)
                sections.append(("paragraph", para, start, end))
                current_pos = end
        
        return sections
    
    def _group_sections(self, sections: List[Tuple[str, str, int, int]], document: Document) -> List[Chunk]:
        """Group sections into appropriately sized chunks."""
        if not sections:
            return []
        
        chunks = []
        current_content = []
        current_start = sections[0][2]
        current_tokens = 0
        max_chars = self.chunk_size * self._chars_per_token
        
        for section_type, content, start, end in sections:
            section_tokens = len(content) // self._chars_per_token
            
            # If this section alone is too big, split it
            if section_tokens > self.chunk_size:
                # Flush current chunk first
                if current_content:
                    chunks.append(self._create_chunk(
                        "\n\n".join(current_content),
                        document,
                        current_start,
                        start,
                        len(chunks),
                    ))
                    current_content = []
                    current_tokens = 0
                
                # Split large section
                sub_chunks = self._split_large_section(content, section_type, document, start, len(chunks))
                chunks.extend(sub_chunks)
                current_start = end
                continue
            
            # Check if adding this section exceeds chunk size
            if current_tokens + section_tokens > self.chunk_size and current_content:
                # Create chunk from accumulated content
                chunks.append(self._create_chunk(
                    "\n\n".join(current_content),
                    document,
                    current_start,
                    start,
                    len(chunks),
                ))
                
                # Start new chunk with overlap (take last section if it's small enough)
                if current_content and len(current_content[-1]) // self._chars_per_token < self._overlap_tokens:
                    current_content = [current_content[-1], content]
                    current_tokens = len(current_content[-1]) // self._chars_per_token + section_tokens
                else:
                    current_content = [content]
                    current_tokens = section_tokens
                current_start = start
            else:
                current_content.append(content)
                current_tokens += section_tokens
        
        # Flush remaining content
        if current_content:
            chunks.append(self._create_chunk(
                "\n\n".join(current_content),
                document,
                current_start,
                sections[-1][3] if sections else 0,
                len(chunks),
            ))
        
        # Update total_chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def _split_large_section(
        self,
        content: str,
        section_type: str,
        document: Document,
        start_char: int,
        start_index: int,
    ) -> List[Chunk]:
        """Split a large section that exceeds chunk size."""
        chunks = []
        max_chars = self.chunk_size * self._chars_per_token
        overlap_chars = self._overlap_tokens * self._chars_per_token
        
        # For code blocks, try to split on newlines
        if section_type == "code_block":
            lines = content.split("\n")
            current_lines = []
            current_len = 0
            
            for line in lines:
                if current_len + len(line) > max_chars and current_lines:
                    chunk_text = "\n".join(current_lines)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        document,
                        start_char,
                        start_char + len(chunk_text),
                        start_index + len(chunks),
                        section_type,
                    ))
                    # Overlap: keep last few lines
                    overlap_lines = []
                    overlap_len = 0
                    for l in reversed(current_lines):
                        if overlap_len + len(l) < overlap_chars:
                            overlap_lines.insert(0, l)
                            overlap_len += len(l)
                        else:
                            break
                    current_lines = overlap_lines + [line]
                    current_len = overlap_len + len(line)
                    start_char += len(chunk_text) - overlap_len
                else:
                    current_lines.append(line)
                    current_len += len(line) + 1
            
            if current_lines:
                chunk_text = "\n".join(current_lines)
                chunks.append(self._create_chunk(
                    chunk_text,
                    document,
                    start_char,
                    start_char + len(chunk_text),
                    start_index + len(chunks),
                    section_type,
                ))
        else:
            # For paragraphs, split on sentences or spaces
            sentences = re.split(r'(?<=[.!?])\s+', content)
            current_text = ""
            
            for sentence in sentences:
                if len(current_text) + len(sentence) > max_chars and current_text:
                    chunks.append(self._create_chunk(
                        current_text.strip(),
                        document,
                        start_char,
                        start_char + len(current_text),
                        start_index + len(chunks),
                        section_type,
                    ))
                    # Overlap
                    overlap_start = max(0, len(current_text) - overlap_chars)
                    current_text = current_text[overlap_start:] + " " + sentence
                    start_char += len(current_text) - overlap_chars
                else:
                    current_text += " " + sentence if current_text else sentence
            
            if current_text:
                chunks.append(self._create_chunk(
                    current_text.strip(),
                    document,
                    start_char,
                    start_char + len(current_text),
                    start_index + len(chunks),
                    section_type,
                ))
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        document: Document,
        start_char: int,
        end_char: int,
        chunk_index: int,
        chunk_type: str = "semantic",
    ) -> Chunk:
        """Create a chunk with metadata."""
        return Chunk(
            content=content,
            document_id=document.id,
            metadata={
                "source": document.source,
                "type": chunk_type,
                "chunk_index": chunk_index,
                "strategy": "semantic",
            },
            start_char=start_char,
            end_char=end_char,
        )
