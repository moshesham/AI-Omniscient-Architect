"""Data models for LLM abstraction layer."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field


class ProviderType(str, Enum):
    """Supported LLM providers."""
    
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class Role(str, Enum):
    """Message roles for chat."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """A single chat message."""
    
    role: Role
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        return result


class LLMUsage(BaseModel):
    """Token usage information."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost tracking (for paid providers)
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    
    @property
    def has_cost(self) -> bool:
        """Check if cost information is available."""
        return self.total_cost > 0


class LLMResponse(BaseModel):
    """Response from LLM generation."""
    
    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model used for generation")
    provider: ProviderType = Field(..., description="Provider used")
    
    # Metadata
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0
    
    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if generation completed normally."""
        return self.finish_reason in ("stop", "end_turn", None)
    
    @property
    def was_truncated(self) -> bool:
        """Check if response was truncated due to length."""
        return self.finish_reason in ("length", "max_tokens")


class StreamChunk(BaseModel):
    """Chunk from streaming response."""
    
    content: str
    is_final: bool = False
    index: int = 0
    
    # Only available on final chunk
    usage: Optional[LLMUsage] = None
    finish_reason: Optional[str] = None


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    model_config = ConfigDict(extra="allow")  # Allow provider-specific options
    
    # Model settings
    model: str = Field(default="llama3.2:latest")
    provider: ProviderType = Field(default=ProviderType.OLLAMA)
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    
    # Timeouts
    timeout_seconds: float = Field(default=120.0)
    connect_timeout_seconds: float = Field(default=10.0)
    
    # Retry settings
    max_retries: int = Field(default=3)
    retry_delay_seconds: float = Field(default=1.0)
    
    # Provider-specific
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # Streaming
    stream_by_default: bool = False


class ModelInfo(BaseModel):
    """Information about an available model."""
    
    name: str
    provider: ProviderType
    
    # Size and capabilities
    size_bytes: Optional[int] = None
    parameter_count: Optional[str] = None  # e.g., "7B", "13B"
    context_length: Optional[int] = None
    
    # Capabilities
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    
    # Metadata
    family: Optional[str] = None  # e.g., "llama", "codellama"
    quantization: Optional[str] = None  # e.g., "Q4_0", "Q8_0"
    modified_at: Optional[datetime] = None
    
    @property
    def size_gb(self) -> Optional[float]:
        """Get size in GB."""
        if self.size_bytes:
            return round(self.size_bytes / (1024**3), 2)
        return None
    
    @property
    def display_name(self) -> str:
        """Get human-readable name."""
        parts = [self.name]
        if self.parameter_count:
            parts.append(f"({self.parameter_count})")
        if self.quantization:
            parts.append(f"[{self.quantization}]")
        return " ".join(parts)


class ModelStatus(BaseModel):
    """Status of a loaded model."""
    
    name: str
    is_loaded: bool = False
    is_loading: bool = False
    
    # Resource usage
    memory_usage_bytes: Optional[int] = None
    gpu_memory_usage_bytes: Optional[int] = None
    
    # Performance
    tokens_per_second: Optional[float] = None
    
    # Errors
    error: Optional[str] = None
    
    @property
    def memory_usage_mb(self) -> Optional[float]:
        """Get memory usage in MB."""
        if self.memory_usage_bytes:
            return round(self.memory_usage_bytes / (1024**2), 1)
        return None
    
    @property
    def gpu_memory_usage_mb(self) -> Optional[float]:
        """Get GPU memory usage in MB."""
        if self.gpu_memory_usage_bytes:
            return round(self.gpu_memory_usage_bytes / (1024**2), 1)
        return None


@dataclass
class GenerationRequest:
    """Request for text generation."""
    
    # Either prompt or messages should be provided
    prompt: Optional[str] = None
    messages: List[ChatMessage] = field(default_factory=list)
    system_prompt: Optional[str] = None
    
    # Generation settings
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    
    # Stop sequences
    stop_sequences: List[str] = field(default_factory=list)
    
    # Context
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_prompt(self) -> str:
        """Get the prompt text (from prompt or messages)."""
        if self.prompt:
            return self.prompt
        if self.messages:
            # Combine messages into prompt
            return "\n\n".join(
                f"{msg.role.value}: {msg.content}" 
                for msg in self.messages
            )
        return ""
    
    def get_messages(self) -> List[ChatMessage]:
        """Get messages list (from messages or prompt)."""
        if self.messages:
            result = []
            if self.system_prompt:
                result.append(ChatMessage(Role.SYSTEM, self.system_prompt))
            result.extend(self.messages)
            return result
        elif self.prompt:
            result = []
            if self.system_prompt:
                result.append(ChatMessage(Role.SYSTEM, self.system_prompt))
            result.append(ChatMessage(Role.USER, self.prompt))
            return result
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {}
        
        if self.prompt:
            result["prompt"] = self.prompt
        if self.messages:
            result["messages"] = [m.to_dict() for m in self.messages]
        if self.system_prompt:
            result["system"] = self.system_prompt
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.stop_sequences:
            result["stop"] = self.stop_sequences
            
        return result
