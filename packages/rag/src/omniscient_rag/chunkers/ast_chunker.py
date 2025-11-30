"""AST-based chunker for source code."""

import re
from typing import List, Tuple, Optional, Dict, Any
from uuid import uuid4

from .base import BaseChunker
from ..models import Document, Chunk

# Try to import tree-sitter for proper AST parsing
try:
    import tree_sitter
    import tree_sitter_python
    import tree_sitter_javascript
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False


class ASTChunker(BaseChunker):
    """AST-based chunker that splits code by structure.
    
    Parses source code and splits by:
    - Functions/methods
    - Classes
    - Module-level blocks
    
    Uses tree-sitter when available for accurate parsing,
    falls back to regex-based detection otherwise.
    
    Supported languages:
    - Python (.py)
    - JavaScript (.js, .jsx)
    - TypeScript (.ts, .tsx) - via regex fallback
    
    Example:
        >>> chunker = ASTChunker(chunk_size=512)
        >>> chunks = chunker.chunk(python_document)
    """
    
    # Regex patterns for fallback parsing
    PYTHON_FUNCTION = re.compile(
        r'^(async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:',
        re.MULTILINE
    )
    PYTHON_CLASS = re.compile(
        r'^class\s+(\w+)\s*(?:\([^)]*\))?:',
        re.MULTILINE
    )
    JS_FUNCTION = re.compile(
        r'^(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
        re.MULTILINE
    )
    JS_CLASS = re.compile(
        r'^class\s+(\w+)',
        re.MULTILINE
    )
    
    # Language detection by extension
    LANGUAGE_MAP = {
        'py': 'python',
        'pyw': 'python',
        'js': 'javascript',
        'jsx': 'javascript',
        'ts': 'typescript',
        'tsx': 'typescript',
        'mjs': 'javascript',
        'cjs': 'javascript',
    }
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: float = 0.1,
        include_imports: bool = True,
        include_docstrings: bool = True,
    ):
        """Initialize AST chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap ratio for large functions
            include_imports: Include imports in first chunk
            include_docstrings: Include docstrings with functions
        """
        super().__init__(chunk_size, chunk_overlap)
        self.include_imports = include_imports
        self.include_docstrings = include_docstrings
        self._parsers: Dict[str, Any] = {}
        self._chars_per_token = 4
        
        if HAS_TREE_SITTER:
            self._init_tree_sitter()
    
    def _init_tree_sitter(self) -> None:
        """Initialize tree-sitter parsers."""
        try:
            # Python parser
            py_lang = tree_sitter_python.language()
            py_parser = tree_sitter.Parser(py_lang)
            self._parsers['python'] = py_parser
            
            # JavaScript parser
            js_lang = tree_sitter_javascript.language()
            js_parser = tree_sitter.Parser(js_lang)
            self._parsers['javascript'] = js_parser
        except Exception:
            pass
    
    def _detect_language(self, document: Document) -> str:
        """Detect programming language from document."""
        ext = document.file_extension
        if ext:
            return self.LANGUAGE_MAP.get(ext, 'unknown')
        return document.metadata.get('language', 'unknown')
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split code document by AST structure.
        
        Args:
            document: Source code document
            
        Returns:
            List of chunks, each containing a logical code unit
        """
        text = document.content
        if not text.strip():
            return []
        
        language = self._detect_language(document)
        
        # Try tree-sitter first
        if language in self._parsers:
            chunks = self._chunk_with_tree_sitter(document, language)
        elif language in ('python', 'javascript', 'typescript'):
            chunks = self._chunk_with_regex(document, language)
        else:
            # Fall back to line-based chunking
            chunks = self._chunk_by_lines(document)
        
        return chunks
    
    def _chunk_with_tree_sitter(self, document: Document, language: str) -> List[Chunk]:
        """Chunk using tree-sitter AST parsing."""
        parser = self._parsers[language]
        tree = parser.parse(bytes(document.content, 'utf-8'))
        root = tree.root_node
        
        # Extract code blocks (functions, classes, etc.)
        blocks = self._extract_ast_blocks(root, document.content, language)
        
        # Group blocks into chunks
        return self._group_blocks(blocks, document)
    
    def _extract_ast_blocks(
        self,
        node: Any,
        source: str,
        language: str,
    ) -> List[Tuple[str, str, int, int, str]]:
        """Extract code blocks from AST.
        
        Returns:
            List of (block_type, name, start_byte, end_byte, content) tuples
        """
        blocks = []
        
        # Node types to extract
        if language == 'python':
            target_types = {'function_definition', 'class_definition', 'decorated_definition'}
        else:  # javascript
            target_types = {'function_declaration', 'class_declaration', 'arrow_function', 'method_definition'}
        
        def traverse(node):
            if node.type in target_types:
                # Get function/class name
                name = "unknown"
                for child in node.children:
                    if child.type in ('identifier', 'name'):
                        name = source[child.start_byte:child.end_byte]
                        break
                
                content = source[node.start_byte:node.end_byte]
                blocks.append((
                    node.type,
                    name,
                    node.start_byte,
                    node.end_byte,
                    content,
                ))
            else:
                for child in node.children:
                    traverse(child)
        
        traverse(node)
        return blocks
    
    def _chunk_with_regex(self, document: Document, language: str) -> List[Chunk]:
        """Chunk using regex-based code structure detection."""
        text = document.content
        lines = text.split('\n')
        
        # Find function and class definitions
        blocks = []
        
        if language == 'python':
            func_pattern = self.PYTHON_FUNCTION
            class_pattern = self.PYTHON_CLASS
        else:
            func_pattern = self.JS_FUNCTION
            class_pattern = self.JS_CLASS
        
        # Find all function matches
        for match in func_pattern.finditer(text):
            name = match.group(2) if match.lastindex >= 2 and match.group(2) else match.group(1) or "anonymous"
            start = match.start()
            end = self._find_block_end(text, start, language)
            blocks.append(('function', name, start, end, text[start:end]))
        
        # Find all class matches
        for match in class_pattern.finditer(text):
            name = match.group(1)
            start = match.start()
            end = self._find_block_end(text, start, language)
            blocks.append(('class', name, start, end, text[start:end]))
        
        # Sort blocks by position
        blocks.sort(key=lambda x: x[2])
        
        # Remove overlapping blocks (class methods are inside classes)
        filtered_blocks = []
        for block in blocks:
            if not any(
                other[2] < block[2] < other[3] 
                for other in filtered_blocks
            ):
                filtered_blocks.append(block)
        
        return self._group_blocks(filtered_blocks, document)
    
    def _find_block_end(self, text: str, start: int, language: str) -> int:
        """Find the end of a code block (function/class)."""
        lines = text[start:].split('\n')
        if not lines:
            return start
        
        if language == 'python':
            # Python: find by indentation
            first_line = lines[0]
            base_indent = len(first_line) - len(first_line.lstrip())
            
            end_offset = len(first_line) + 1
            for line in lines[1:]:
                stripped = line.lstrip()
                if stripped and (len(line) - len(stripped)) <= base_indent:
                    break
                end_offset += len(line) + 1
            
            return start + end_offset
        else:
            # JavaScript/TypeScript: count braces
            brace_count = 0
            in_string = False
            string_char = None
            
            for i, char in enumerate(text[start:]):
                if in_string:
                    if char == string_char and (i == 0 or text[start + i - 1] != '\\'):
                        in_string = False
                elif char in '"\'`':
                    in_string = True
                    string_char = char
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return start + i + 1
            
            return len(text)
    
    def _chunk_by_lines(self, document: Document) -> List[Chunk]:
        """Fallback: chunk by lines for unsupported languages."""
        text = document.content
        lines = text.split('\n')
        max_lines = self.chunk_size // 10  # ~10 tokens per line average
        
        chunks = []
        current_lines = []
        current_start = 0
        
        for i, line in enumerate(lines):
            current_lines.append(line)
            
            if len(current_lines) >= max_lines:
                chunk_text = '\n'.join(current_lines)
                chunks.append(Chunk(
                    content=chunk_text,
                    document_id=document.id,
                    metadata={
                        "source": document.source,
                        "type": "code_block",
                        "chunk_index": len(chunks),
                        "strategy": "ast_fallback",
                        "line_start": current_start,
                        "line_end": i,
                    },
                    start_char=sum(len(l) + 1 for l in lines[:current_start]),
                    end_char=sum(len(l) + 1 for l in lines[:i+1]),
                ))
                
                # Overlap: keep last few lines
                overlap_lines = max(1, int(max_lines * self.chunk_overlap))
                current_lines = current_lines[-overlap_lines:]
                current_start = i - overlap_lines + 1
        
        # Remaining lines
        if current_lines:
            chunk_text = '\n'.join(current_lines)
            chunks.append(Chunk(
                content=chunk_text,
                document_id=document.id,
                metadata={
                    "source": document.source,
                    "type": "code_block",
                    "chunk_index": len(chunks),
                    "strategy": "ast_fallback",
                },
                start_char=sum(len(l) + 1 for l in lines[:current_start]),
                end_char=len(text),
            ))
        
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def _group_blocks(
        self,
        blocks: List[Tuple[str, str, int, int, str]],
        document: Document,
    ) -> List[Chunk]:
        """Group code blocks into appropriately sized chunks."""
        if not blocks:
            return self._chunk_by_lines(document)
        
        text = document.content
        chunks = []
        max_chars = self.chunk_size * self._chars_per_token
        
        # Handle imports at the start
        first_block_start = blocks[0][2] if blocks else len(text)
        preamble = text[:first_block_start].strip()
        
        current_content = []
        current_tokens = 0
        
        if preamble and self.include_imports:
            preamble_tokens = len(preamble) // self._chars_per_token
            if preamble_tokens < self.chunk_size:
                current_content.append(preamble)
                current_tokens = preamble_tokens
            else:
                # Preamble alone is a chunk
                chunks.append(Chunk(
                    content=preamble,
                    document_id=document.id,
                    metadata={
                        "source": document.source,
                        "type": "imports",
                        "chunk_index": 0,
                        "strategy": "ast",
                    },
                    start_char=0,
                    end_char=first_block_start,
                ))
        
        for block_type, name, start, end, content in blocks:
            block_tokens = len(content) // self._chars_per_token
            
            # If block is too large, split it
            if block_tokens > self.chunk_size:
                # Flush current chunk
                if current_content:
                    chunks.append(self._make_chunk(
                        "\n\n".join(current_content),
                        document,
                        len(chunks),
                    ))
                    current_content = []
                    current_tokens = 0
                
                # Split large block
                sub_chunks = self._split_large_block(content, block_type, name, document, len(chunks))
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding block exceeds limit
            if current_tokens + block_tokens > self.chunk_size and current_content:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_content),
                    document,
                    len(chunks),
                ))
                current_content = [content]
                current_tokens = block_tokens
            else:
                current_content.append(content)
                current_tokens += block_tokens
        
        # Flush remaining
        if current_content:
            chunks.append(self._make_chunk(
                "\n\n".join(current_content),
                document,
                len(chunks),
            ))
        
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def _split_large_block(
        self,
        content: str,
        block_type: str,
        name: str,
        document: Document,
        start_index: int,
    ) -> List[Chunk]:
        """Split a large function/class into smaller chunks."""
        lines = content.split('\n')
        max_lines = self.chunk_size // 10
        overlap_lines = max(1, int(max_lines * self.chunk_overlap))
        
        chunks = []
        current_lines = []
        
        for line in lines:
            current_lines.append(line)
            
            if len(current_lines) >= max_lines:
                chunk_text = '\n'.join(current_lines)
                chunks.append(Chunk(
                    content=chunk_text,
                    document_id=document.id,
                    metadata={
                        "source": document.source,
                        "type": block_type,
                        "name": name,
                        "chunk_index": start_index + len(chunks),
                        "strategy": "ast",
                        "is_partial": True,
                    },
                ))
                current_lines = current_lines[-overlap_lines:]
        
        if current_lines:
            chunks.append(Chunk(
                content='\n'.join(current_lines),
                document_id=document.id,
                metadata={
                    "source": document.source,
                    "type": block_type,
                    "name": name,
                    "chunk_index": start_index + len(chunks),
                    "strategy": "ast",
                    "is_partial": len(chunks) > 0,
                },
            ))
        
        return chunks
    
    def _make_chunk(self, content: str, document: Document, index: int) -> Chunk:
        """Create a chunk with standard metadata."""
        return Chunk(
            content=content,
            document_id=document.id,
            metadata={
                "source": document.source,
                "type": "code_block",
                "chunk_index": index,
                "strategy": "ast",
            },
        )
