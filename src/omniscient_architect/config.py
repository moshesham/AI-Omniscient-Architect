import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .models import AnalysisConfig


def _repo_root() -> Path:
    # Assume repo root is two levels up from this file (src/omniscient_architect)
    return Path(__file__).resolve().parents[2]


def load_config(overrides: Dict[str, Any] | None = None) -> AnalysisConfig:
    """Load configuration from config.yaml and environment variables.

    Priority: defaults in AnalysisConfig -> config.yaml -> environment variables -> overrides
    """
    cfg = AnalysisConfig()

    repo_root = _repo_root()
    config_file = repo_root / "config.yaml"

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Map known keys
            if "max_file_size_mb" in data:
                cfg.max_file_size = int(data["max_file_size_mb"]) * 1024 * 1024
            if "max_files" in data:
                cfg.max_files = int(data["max_files"])
            if "include_patterns" in data:
                cfg.include_patterns = list(data["include_patterns"])
            if "exclude_patterns" in data:
                cfg.exclude_patterns = list(data["exclude_patterns"])
            if "ollama_model" in data:
                cfg.ollama_model = str(data["ollama_model"])
            if "analysis_depth" in data:
                cfg.analysis_depth = str(data["analysis_depth"])
        except Exception:
            # If YAML parse fails, ignore and proceed with defaults
            pass

    # Environment overrides
    env_ollama = os.getenv("OLLAMA_MODEL")
    if env_ollama:
        cfg.ollama_model = env_ollama

    env_max_file_size = os.getenv("MAX_FILE_SIZE_MB")
    if env_max_file_size:
        try:
            cfg.max_file_size = int(env_max_file_size) * 1024 * 1024
        except Exception:
            pass

    env_max_files = os.getenv("MAX_FILES")
    if env_max_files:
        try:
            cfg.max_files = int(env_max_files)
        except Exception:
            pass

    env_depth = os.getenv("ANALYSIS_DEPTH")
    if env_depth:
        cfg.analysis_depth = env_depth

    # Apply explicit overrides
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg
