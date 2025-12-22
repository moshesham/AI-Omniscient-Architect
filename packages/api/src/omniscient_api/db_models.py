from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship, JSON
from omniscient_core import AnalysisStatus, Severity, FindingCategory
import uuid

class AnalysisBase(SQLModel):
    repository_url: str
    branch: Optional[str] = None
    status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    # Storing summary and metrics as JSON for simplicity for now
    summary: Optional[dict] = Field(default=None, sa_type=JSON)
    metrics: Optional[dict] = Field(default=None, sa_type=JSON)

class Analysis(AnalysisBase, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    findings: List["FindingModel"] = Relationship(back_populates="analysis")

class FindingModel(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    analysis_id: str = Field(foreign_key="analysis.id")
    
    agent_name: str
    severity: Severity
    category: FindingCategory
    title: str
    description: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list, sa_type=JSON)
    metadata_json: dict = Field(default_factory=dict, sa_type=JSON, alias="metadata")

    analysis: Analysis = Relationship(back_populates="findings")
