"""API route definitions."""

from typing import List, Optional
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from omniscient_core.logging import get_logger
from ..models import (
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
from ..db_models import Analysis, FindingModel
from ..database import get_session
from ..auth import verify_api_key

logger = get_logger(__name__)

router = APIRouter(tags=["Analysis"], dependencies=[Depends(verify_api_key)])


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
    session: AsyncSession = Depends(get_session),
) -> AnalysisResponse:
    """Analyze a GitHub repository.
    
    Submits the repository for async analysis. Use the returned
    analysis_id to check status and retrieve results.
    """
    analysis_id = str(uuid.uuid4())
    
    # Create DB record
    db_analysis = Analysis(
        id=analysis_id,
        status=AnalysisStatus.PENDING,
        repository_url=request.repository_url,
        branch=request.config.branch if request.config else None,
        created_at=datetime.utcnow(),
    )
    session.add(db_analysis)
    await session.commit()
    await session.refresh(db_analysis)
    
    # Schedule background analysis
    background_tasks.add_task(
        _run_analysis,
        analysis_id,
        request,
    )
    
    logger.info(f"Analysis {analysis_id} queued for {request.repository_url}")
    
    return AnalysisResponse(
        analysis_id=db_analysis.id,
        status=db_analysis.status,
        repository_url=db_analysis.repository_url,
        branch=db_analysis.branch,
        created_at=db_analysis.created_at,
    )


async def _run_analysis(analysis_id: str, request: AnalysisRequest):
    """Run analysis in background.
    
    TODO: Implement full analysis pipeline integration.
    """
    # Manually get session since we are in background task
    from .database import get_session
    async for session in get_session():
        try:
            result = await session.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result.scalar_one_or_none()
            
            if not analysis:
                return
            
            # Update status
            analysis.status = AnalysisStatus.RUNNING
            analysis.started_at = datetime.utcnow()
            await session.commit()
            
            # TODO: Integrate with actual analysis pipeline
            # For now, simulate with placeholder results
            import asyncio
            await asyncio.sleep(2)  # Simulate processing
            
            # Mock findings for demonstration
            findings = [
                FindingModel(
                    analysis_id=analysis_id,
                    agent_name="architecture",
                    severity=Severity.MEDIUM,
                    category=FindingCategory.ARCHITECTURE,
                    title="Consider extracting common utilities",
                    description="Several modules contain similar helper functions that could be consolidated.",
                    suggestions=["Create a shared utils module", "Apply DRY principles"],
                ),
            ]
            
            for f in findings:
                session.add(f)
            
            analysis.summary = {
                "total_findings": len(findings),
                "total_files": 10,  # Placeholder
                "findings_by_severity": {"medium": 1},
                "overall_score": 85.0,
            }
            
            analysis.metrics = {
                "files_analyzed": 10,
                "analysis_time_seconds": 2.0,
                "agents_used": request.agents or ["architecture", "reliability"],
            }
            
            analysis.status = AnalysisStatus.COMPLETED
            analysis.completed_at = datetime.utcnow()
            
            await session.commit()
            logger.info(f"Analysis {analysis_id} completed")
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            try:
                analysis.status = AnalysisStatus.FAILED
                analysis.error = str(e)
                await session.commit()
            except:
                pass
        break


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get analysis status and results",
)
async def get_analysis(
    analysis_id: str,
    session: AsyncSession = Depends(get_session),
) -> AnalysisResponse:
    """Get analysis by ID.
    
    Returns current status and results (if completed).
    """
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Analysis)
        .where(Analysis.id == analysis_id)
        .options(selectinload(Analysis.findings))
    )
    analysis = result.scalar_one_or_none()
    
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
    session: AsyncSession = Depends(get_session),
) -> List[AnalysisResponse]:
    """List recent analyses.
    
    Args:
        limit: Maximum number of results
        status: Filter by status
    """
    query = select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)
    
    if status:
        query = query.where(Analysis.status == status)
    
    result = await session.execute(query)
    analyses = result.scalars().all()
    
    return analyses


@router.delete(
    "/analysis/{analysis_id}",
    summary="Cancel or delete an analysis",
)
async def delete_analysis(
    analysis_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Cancel or delete an analysis."""
    result = await session.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    
    if analysis.status in [AnalysisStatus.PENDING, AnalysisStatus.RUNNING]:
        analysis.status = AnalysisStatus.CANCELLED
        await session.commit()
        return {"message": f"Analysis {analysis_id} cancelled"}
    else:
        await session.delete(analysis)
        await session.commit()
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
    session: AsyncSession = Depends(get_session),
) -> AnalysisResponse:
    """Analyze uploaded files directly.
    
    Accepts multiple file uploads for analysis without requiring
    a GitHub repository.
    """
    analysis_id = str(uuid.uuid4())
    
    # Read file contents
    file_contents = {}
    for file in files:
        content = await file.read()
        file_contents[file.filename] = content.decode("utf-8", errors="ignore")
    
    # Create DB record
    db_analysis = Analysis(
        id=analysis_id,
        status=AnalysisStatus.PENDING,
        repository_url=f"upload://{project_name or 'files'}",
        created_at=datetime.utcnow(),
    )
    session.add(db_analysis)
    await session.commit()
    await session.refresh(db_analysis)
    
    logger.info(f"File analysis {analysis_id} queued with {len(files)} files")
    
    # TODO: Schedule background analysis with file_contents
    
    return AnalysisResponse(
        analysis_id=db_analysis.id,
        status=db_analysis.status,
        repository_url=db_analysis.repository_url,
        created_at=db_analysis.created_at,
    )


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
async def get_stats(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get API usage statistics."""
    from sqlalchemy import func
    
    # Count total
    result = await session.execute(select(func.count(Analysis.id)))
    total = result.scalar()
    
    # Count by status
    result = await session.execute(
        select(Analysis.status, func.count(Analysis.id))
        .group_by(Analysis.status)
    )
    status_counts = {status.value: count for status, count in result.all()}
    
    return {
        "total_analyses": total,
        "by_status": status_counts,
        "agents_available": 4,
    }
