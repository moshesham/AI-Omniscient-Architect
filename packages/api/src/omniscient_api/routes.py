"""API route definitions."""

from typing import List, Optional
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from omniscient_core.logging import get_logger
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    Finding,
    AnalysisSummary,
    AnalysisMetrics,
    AgentInfo,
    AgentListResponse,
    ErrorResponse,
    Severity,
    FindingCategory,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Analysis"])

# In-memory storage for demo (replace with proper storage)
_analyses: dict = {}


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Analyze a GitHub repository",
    description="Submit a repository for analysis. Returns immediately with an analysis ID.",
)
async def analyze_repository(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisResponse:
    """Analyze a GitHub repository.
    
    Submits the repository for async analysis. Use the returned
    analysis_id to check status and retrieve results.
    """
    analysis_id = str(uuid.uuid4())
    
    # Create initial response
    response = AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        repository_url=request.repository_url,
        branch=request.config.branch if request.config else None,
        created_at=datetime.utcnow(),
    )
    
    # Store analysis
    _analyses[analysis_id] = response
    
    # Schedule background analysis
    background_tasks.add_task(
        _run_analysis,
        analysis_id,
        request,
    )
    
    logger.info(f"Analysis {analysis_id} queued for {request.repository_url}")
    
    return response


async def _run_analysis(analysis_id: str, request: AnalysisRequest):
    """Run analysis in background.
    
    TODO: Implement full analysis pipeline integration.
    """
    try:
        analysis = _analyses.get(analysis_id)
        if not analysis:
            return
        
        # Update status
        analysis.status = AnalysisStatus.RUNNING
        analysis.started_at = datetime.utcnow()
        
        # TODO: Integrate with actual analysis pipeline
        # For now, simulate with placeholder results
        import asyncio
        await asyncio.sleep(2)  # Simulate processing
        
        # Mock findings for demonstration
        analysis.findings = [
            Finding(
                agent_name="architecture",
                severity=Severity.MEDIUM,
                category=FindingCategory.ARCHITECTURE,
                title="Consider extracting common utilities",
                description="Several modules contain similar helper functions that could be consolidated.",
                suggestions=["Create a shared utils module", "Apply DRY principles"],
            ),
        ]
        
        analysis.summary = AnalysisSummary(
            total_findings=len(analysis.findings),
            total_files=10,  # Placeholder
            findings_by_severity={"medium": 1},
            overall_score=85.0,
        )
        
        analysis.metrics = AnalysisMetrics(
            files_analyzed=10,
            analysis_time_seconds=2.0,
            agents_used=request.agents or ["architecture", "reliability"],
        )
        
        analysis.status = AnalysisStatus.COMPLETED
        analysis.completed_at = datetime.utcnow()
        
        logger.info(f"Analysis {analysis_id} completed")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        if analysis_id in _analyses:
            _analyses[analysis_id].status = AnalysisStatus.FAILED
            _analyses[analysis_id].error = str(e)


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get analysis status and results",
)
async def get_analysis(analysis_id: str) -> AnalysisResponse:
    """Get analysis by ID.
    
    Returns current status and results (if completed).
    """
    analysis = _analyses.get(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    return analysis


@router.get(
    "/analyses",
    response_model=List[AnalysisResponse],
    summary="List recent analyses",
)
async def list_analyses(
    limit: int = 10,
    status: Optional[AnalysisStatus] = None,
) -> List[AnalysisResponse]:
    """List recent analyses.
    
    Args:
        limit: Maximum number of results
        status: Filter by status
    """
    analyses = list(_analyses.values())
    
    if status:
        analyses = [a for a in analyses if a.status == status]
    
    # Sort by created_at descending
    analyses.sort(key=lambda a: a.created_at, reverse=True)
    
    return analyses[:limit]


@router.delete(
    "/analysis/{analysis_id}",
    summary="Cancel or delete an analysis",
)
async def delete_analysis(analysis_id: str) -> dict:
    """Cancel or delete an analysis."""
    if analysis_id not in _analyses:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    
    analysis = _analyses[analysis_id]
    
    if analysis.status in [AnalysisStatus.PENDING, AnalysisStatus.RUNNING]:
        analysis.status = AnalysisStatus.CANCELLED
        return {"message": f"Analysis {analysis_id} cancelled"}
    else:
        del _analyses[analysis_id]
        return {"message": f"Analysis {analysis_id} deleted"}


@router.post(
    "/analyze/files",
    response_model=AnalysisResponse,
    summary="Analyze uploaded files",
)
async def analyze_files(
    files: List[UploadFile] = File(...),
    agents: Optional[str] = Form(None),
    project_name: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
) -> AnalysisResponse:
    """Analyze uploaded files directly.
    
    Accepts multiple file uploads for analysis without requiring
    a GitHub repository.
    """
    analysis_id = str(uuid.uuid4())
    
    # Parse agents
    agent_list = agents.split(",") if agents else None
    
    # Read file contents
    file_contents = {}
    for file in files:
        content = await file.read()
        file_contents[file.filename] = content.decode("utf-8", errors="ignore")
    
    response = AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        repository_url=f"upload://{project_name or 'files'}",
        created_at=datetime.utcnow(),
    )
    
    _analyses[analysis_id] = response
    
    logger.info(f"File analysis {analysis_id} queued with {len(files)} files")
    
    # TODO: Schedule background analysis with file_contents
    
    return response


@router.get(
    "/agents",
    response_model=AgentListResponse,
    summary="List available agents",
)
async def list_agents() -> AgentListResponse:
    """List all available analysis agents."""
    agents = [
        AgentInfo(
            name="architecture",
            description="Analyzes code architecture, design patterns, and structural issues",
            focus_areas=["modularity", "coupling", "cohesion", "patterns"],
        ),
        AgentInfo(
            name="reliability",
            description="Identifies reliability and robustness concerns",
            focus_areas=["error_handling", "edge_cases", "validation", "testing"],
        ),
        AgentInfo(
            name="efficiency",
            description="Detects performance and efficiency issues",
            focus_areas=["complexity", "memory", "algorithms", "optimization"],
        ),
        AgentInfo(
            name="alignment",
            description="Checks code alignment with best practices and standards",
            focus_areas=["style", "documentation", "naming", "conventions"],
        ),
    ]
    
    return AgentListResponse(agents=agents, total=len(agents))


@router.get(
    "/agents/{agent_name}",
    response_model=AgentInfo,
    responses={404: {"model": ErrorResponse}},
    summary="Get agent details",
)
async def get_agent(agent_name: str) -> AgentInfo:
    """Get details for a specific agent."""
    agents = (await list_agents()).agents
    
    for agent in agents:
        if agent.name == agent_name:
            return agent
    
    raise HTTPException(
        status_code=404,
        detail=f"Agent '{agent_name}' not found"
    )


@router.get(
    "/stats",
    summary="Get API statistics",
)
async def get_stats() -> dict:
    """Get API usage statistics."""
    analyses = list(_analyses.values())
    
    status_counts = {}
    for analysis in analyses:
        status = analysis.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        "total_analyses": len(analyses),
        "by_status": status_counts,
        "agents_available": 4,
    }
