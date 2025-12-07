"""Common enums used across the Omniscient Architect platform."""

from enum import Enum


class AnalysisStatus(str, Enum):
    """Status of an analysis job or task."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Severity(str, Enum):
    """Finding severity levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(str, Enum):
    """Categories of findings."""
    
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    CODE_STYLE = "code_style"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
