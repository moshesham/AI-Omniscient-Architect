"""API configuration."""

from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CORSConfig(BaseModel):
    """CORS configuration."""
    
    allow_origins: List[str] = Field(default=["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = Field(default=["*"])
    allow_headers: List[str] = Field(default=["*"])


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10


class AnalysisConfig(BaseModel):
    """Analysis defaults."""
    
    max_file_size: int = Field(default=100_000, description="Max file size in bytes")
    max_files: int = Field(default=500, description="Max files per analysis")
    default_agents: List[str] = Field(
        default=["architecture", "reliability", "efficiency", "alignment"]
    )
    timeout_seconds: int = Field(default=300, description="Analysis timeout")


class LLMConfig(BaseModel):
    """LLM configuration."""
    
    provider: str = "ollama"
    model: str = "llama3.2:latest"
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: Optional[str] = None


class APIConfig(BaseSettings):
    """Main API configuration.
    
    Configuration is loaded from environment variables with
    OMNISCIENT_ prefix.
    """
    
    # Server settings
    host: str = Field(default="0.0.0.0", alias="OMNISCIENT_API_HOST")
    port: int = Field(default=8000, alias="OMNISCIENT_API_PORT")
    workers: int = Field(default=1, alias="OMNISCIENT_API_WORKERS")
    debug: bool = Field(default=False, alias="OMNISCIENT_DEBUG")
    
    # API settings
    api_prefix: str = "/api/v1"
    title: str = "Omniscient Architect API"
    description: str = "Code review and analysis API"
    version: str = "0.1.0"
    
    # GitHub
    github_token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")
    
    # LLM
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    
    # Sub-configs
    cors: CORSConfig = Field(default_factory=CORSConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    model_config = SettingsConfigDict(
        env_prefix="OMNISCIENT_",
        env_file=".env",
        extra="ignore",
    )
    
    def get_llm_base_url(self) -> str:
        """Get LLM base URL."""
        return self.llm.base_url or self.ollama_host


def load_api_config(config_path: Optional[str] = None) -> APIConfig:
    """Load API configuration.
    
    Args:
        config_path: Optional path to YAML config file
        
    Returns:
        APIConfig instance
    """
    import yaml
    from pathlib import Path
    
    config = APIConfig()
    
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                yaml_config = yaml.safe_load(f)
            
            # Update from YAML
            if "api" in yaml_config:
                api = yaml_config["api"]
                if "host" in api:
                    config.host = api["host"]
                if "port" in api:
                    config.port = api["port"]
                if "workers" in api:
                    config.workers = api["workers"]
                if "cors_origins" in api:
                    config.cors.allow_origins = api["cors_origins"]
            
            if "analysis" in yaml_config:
                analysis = yaml_config["analysis"]
                if "max_file_size" in analysis:
                    config.analysis.max_file_size = analysis["max_file_size"]
                if "default_agents" in analysis:
                    config.analysis.default_agents = analysis["default_agents"]
            
            if "llm" in yaml_config:
                llm = yaml_config["llm"]
                if "provider" in llm:
                    config.llm.provider = llm["provider"]
                if "model" in llm:
                    config.llm.model = llm["model"]
    
    return config
