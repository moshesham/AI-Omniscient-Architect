from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from pydantic import BaseModel

from omniscient_rag.models import RetrievalResult
from ..auth import verify_api_key

router = APIRouter(tags=["RAG"], dependencies=[Depends(verify_api_key)])

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    hybrid_alpha: float = 0.5

class SearchResponse(BaseModel):
    results: List[RetrievalResult]

@router.post("/rag/ingest", summary="Ingest files into RAG")
async def ingest_files(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """Ingest uploaded files into the RAG system."""
    pipeline = request.app.state.rag_pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Save files temporarily or process in memory
    # For now, we'll assume text files and process in memory
    count = 0
    for file in files:
        content = (await file.read()).decode("utf-8", errors="ignore")
        
        try:
            await pipeline.ingest_text(
                content=content,
                source=file.filename,
                metadata={"filename": file.filename, "content_type": file.content_type}
            )
            count += 1
        except Exception as e:
            # Log error but continue
            pass
            
    return {"message": f"Ingested {count} files"}

@router.post("/rag/search", response_model=SearchResponse, summary="Search RAG knowledge base")
async def search(
    request: Request,
    search_req: SearchRequest,
):
    """Search the RAG knowledge base."""
    pipeline = request.app.state.rag_pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    results = await pipeline.query(
        search_req.query, 
        top_k=search_req.top_k,
        hybrid_alpha=search_req.hybrid_alpha
    )
    
    return SearchResponse(results=results)
