# RAG Improvements & Ollama Integration

## Overview
We have improved the RAG pipeline to support real embeddings and LLM-based question generation using Ollama.

## Key Changes

1.  **Dual Provider Setup**:
    *   **Embedding**: Uses `nomic-embed-text` for vector embeddings.
    *   **LLM**: Uses `qwen2.5-coder:1.5b` (or `llama3.2`) for generating test questions.

2.  **Robust Embedding Wrappers**:
    *   `safe_embed`: Wraps single embedding calls with a 30s timeout to prevent hangs.
    *   `safe_embed_batch`: Forces serial execution of batches to avoid Ollama concurrency issues (which caused hangs).

3.  **LLM Integration**:
    *   Added `llm_wrapper` to convert string prompts (from `QuestionGenerator`) into `GenerationRequest` objects required by `OllamaProvider`.
    *   Enabled `auto_generate_questions=True` in `RAGConfig`.

4.  **Pipeline Updates**:
    *   Used `pipeline.ingest_file()` which automatically handles chunking, embedding, and question generation.

## How to Run
Run the improved evaluation script:
```powershell
& .\.venv\Scripts\python.exe examples\evaluate_rag_improved.py
```

## Troubleshooting
*   **Ollama Hangs**: If Ollama hangs, the script uses serial processing. Ensure your Ollama server is running and has enough resources.
*   **Missing Models**: Run `ollama pull nomic-embed-text` and `ollama pull qwen2.5-coder:1.5b`.
