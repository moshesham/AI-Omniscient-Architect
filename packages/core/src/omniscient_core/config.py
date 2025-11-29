"""Configuration loading for Omniscient Architect."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .models import AnalysisConfig


def get_config_path() -> Path:
    """Get the default config file path.
    
    Checks in order:
    1. OMNISCIENT_CONFIG env var
    2. ./config.yaml
    3. ~/.omniscient/config.yaml
    """
    env_path = os.getenv("OMNISCIENT_CONFIG")
    if env_path:
        return Path(env_path)
    
    local_config = Path.cwd() / "config.yaml"
    if local_config.exists():
        return local_config
    
    home_config = Path.home() / ".omniscient" / "config.yaml"
    if home_config.exists():
        return home_config
    
    return local_config  # Default to local even if it doesn't exist


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> AnalysisConfig:
    """Load configuration from YAML file and environment variables.

    Priority (highest to lowest):
    1. Explicit overrides parameter
    2. Environment variables
    3. Config file values
    4. AnalysisConfig defaults
    
    Args:
        config_path: Path to config file. If None, uses get_config_path().
        overrides: Dict of values to override config with.
        
    Returns:
        Populated AnalysisConfig instance.
    """
    cfg = AnalysisConfig()
    
    # Determine config file path
    if config_path is None:
        config_path = get_config_path()

    # Load from YAML if exists
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _apply_yaml_config(cfg, data)
        except Exception:
            pass  # Silently continue with defaults

    # Apply environment variable overrides
    _apply_env_overrides(cfg)

    # Apply explicit overrides
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg


def _apply_yaml_config(cfg: AnalysisConfig, data: Dict[str, Any]) -> None:
    """Apply YAML config data to AnalysisConfig."""
    mapping = {
        "max_file_size_mb": ("max_file_size", lambda x: int(x) * 1024 * 1024),
        "max_files": ("max_files", int),
        "include_patterns": ("include_patterns", list),
        "exclude_patterns": ("exclude_patterns", list),
        "exclude_extensions": ("exclude_extensions", list),
        "ollama_model": ("ollama_model", str),
        "ollama_host": ("ollama_host", str),
        "analysis_depth": ("analysis_depth", str),
        "cache_enabled": ("cache_enabled", bool),
        "cache_dir": ("cache_dir", str),
        "cache_ttl": ("cache_ttl", int),
        "api_host": ("api_host", str),
        "api_port": ("api_port", int),
        "agent_concurrency": ("agent_concurrency", int),
        "max_files_for_llm": ("max_files_for_llm", int),
        "sampling_strategy": ("sampling_strategy", str),
    }
    
    for yaml_key, (attr, converter) in mapping.items():
        if yaml_key in data:
            try:
                setattr(cfg, attr, converter(data[yaml_key]))
            except Exception:
                pass


def _apply_env_overrides(cfg: AnalysisConfig) -> None:
    """Apply environment variable overrides to config."""
    env_mapping = {
        "OLLAMA_MODEL": ("ollama_model", str),
        "OLLAMA_HOST": ("ollama_host", str),
        "ANALYSIS_DEPTH": ("analysis_depth", str),
        "MAX_FILE_SIZE_MB": ("max_file_size", lambda x: int(x) * 1024 * 1024),
        "MAX_FILES": ("max_files", int),
        "OMNISCIENT_CACHE_DIR": ("cache_dir", str),
        "OMNISCIENT_CACHE_TTL": ("cache_ttl", int),
        "OMNISCIENT_API_HOST": ("api_host", str),
        "OMNISCIENT_API_PORT": ("api_port", int),
        "OMNISCIENT_API_KEY": ("api_key", str),
    }
    
    for env_var, (attr, converter) in env_mapping.items():
        value = os.getenv(env_var)
        if value:
            try:
                setattr(cfg, attr, converter(value))
            except Exception:
                pass
