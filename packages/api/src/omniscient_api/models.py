"""API data models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
import uuid

from omniscient_core import AnalysisStatus, Severity, FindingCategory


# Request Models

class AnalysisConfig(BaseModel):
    """Configuration for analysis request."""
    
    max_files: Optional[int] = Field(default=None, description="Max files to analyze")
    max_file_size: Optional[int] = Field(default=None, description="Max file size")
    include_patterns: Optional[List[str]] = Field(default=None, description="Files to include")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="Files to exclude")
    branch: Optional[str] = Field(default=None, description="Branch to analyze")


class AnalysisRequest(BaseModel):
    """Request to analyze a repository."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "repository_url": "https://github.com/owner/repo",
            "agents": ["architecture", "reliability"],
            "config": {
                "max_files": 100,
                "include_patterns": ["*.py", "*.js"]
            }
        }
    })
    
    repository_url: str = Field(..., description="GitHub repository URL")
    agents: Optional[List[str]] = Field(
        default=None, 
        description="Agents to use (default: all)"
    )
    config: Optional[AnalysisConfig] = Field(
        default=None,
        description="Analysis configuration"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for completion notification"
    )


class FileUploadRequest(BaseModel):
    """Request for file upload analysis."""
    
    agents: Optional[List[str]] = None
    project_name: Optional[str] = None


# Response Models

class Finding(BaseModel):
    """A single finding from analysis."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "agent_name": "architecture",
            "severity": "high",
            "category": "architecture",
            "title": "Circular Dependency Detected",
            "description": "Module A imports Module B which imports Module A",
            "file_path": "src/module_a.py",
            "line_start": 5,
            "suggestions": [
                "Extract shared logic to a common module",
                "Use dependency injection"
            ]
        }
    })
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = Field(..., description="Agent that produced this finding")
    severity: Severity = Field(..., description="Severity level")
    category: FindingCategory = Field(..., description="Finding category")
    title: str = Field(..., description="Short title")
    description: str = Field(..., description="Detailed description")
    file_path: Optional[str] = Field(default=None, description="Affected file")
    line_start: Optional[int] = Field(default=None, description="Start line")
    line_end: Optional[int] = Field(default=None, description="End line")
    code_snippet: Optional[str] = Field(default=None, description="Relevant code")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional data")


class AgentSummary(BaseModel):
    """Summary of agent analysis."""
    
    agent_name: str
    findings_count: int
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    analysis_time_seconds: float = 0.0
    status: str = "completed"


class AnalysisSummary(BaseModel):
    """Summary of complete analysis."""
    
    total_findings: int = 0
    total_files: int = 0
    total_lines: int = 0
    findings_by_severity: Dict[str, int] = Field(default_factory=dict)
    findings_by_category: Dict[str, int] = Field(default_factory=dict)
    agent_summaries: List[AgentSummary] = Field(default_factory=list)
    top_issues: List[str] = Field(default_factory=list, description="Top 5 issues")
    overall_score: Optional[float] = Field(
        default=None, 
        ge=0, 
        le=100,
        description="Overall code quality score"
    )


class AnalysisMetrics(BaseModel):
    """Metrics from analysis."""
    
    files_analyzed: int = 0
    lines_analyzed: int = 0
    analysis_time_seconds: float = 0.0
    agents_used: List[str] = Field(default_factory=list)
    cache_hits: int = 0
    api_calls: int = 0


class AnalysisResponse(BaseModel):
    """Response for analysis request."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "repository_url": "https://github.com/owner/repo",
            "created_at": "2024-01-15T10:30:00Z",
            "completed_at": "2024-01-15T10:35:00Z",
            "findings": [],
            "summary": {
                "total_findings": 15,
                "overall_score": 78.5
            }
        }
    })
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AnalysisStatus = AnalysisStatus.PENDING
    repository_url: str
    branch: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    findings: List[Finding] = Field(default_factory=list)
    summary: Optional[AnalysisSummary] = None
    metrics: Optional[AnalysisMetrics] = None
    error: Optional[str] = None


class AgentInfo(BaseModel):
    """Information about an available agent."""
    
    name: str
    description: str
    focus_areas: List[str]
    enabled: bool = True


class AgentListResponse(BaseModel):
    """List of available agents."""
    
    agents: List[AgentInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    version: str
    uptime_seconds: float
    checks: Dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
