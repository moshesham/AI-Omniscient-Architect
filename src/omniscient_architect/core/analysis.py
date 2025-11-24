"""Core analysis engine for the Omniscient Architect."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseLanguageModel

from ..models import (
    FileAnalysis, AgentFindings, ReviewResult,
    RepositoryInfo, AnalysisConfig
)
from ..agent_registry import create_agents

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """Main analysis engine coordinating AI agents."""

    def __init__(self, config: AnalysisConfig):
        """
        Args:
            config: AnalysisConfig object with analysis parameters.
        """
        self.config = config
        self.llm: Optional[BaseLanguageModel] = None
        self.agents = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def _check_ollama_server(self, base_url: str) -> bool:
        """Check if Ollama server is responding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def _check_model_available(self, base_url: str, model: str) -> bool:
        """Check if the specified model is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    return model in models
        except Exception as e:
            logger.error(f"Model check failed: {e}")
        return False

# ...existing code...
