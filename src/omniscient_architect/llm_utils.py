"""Utilities for managing Ollama LLM readiness and model availability."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

LOG = logging.getLogger(__name__)


async def check_server(base_url: str, timeout: float = 5.0) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/api/tags", timeout=timeout)
            return resp.status_code == 200
    except Exception:
        return False


async def check_model_available(base_url: str, model: str, timeout: float = 10.0) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/api/tags", timeout=timeout)
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            return model in models
    except Exception as e:
        LOG.debug("Model availability check failed: %s", e)
        return False


async def start_ollama_with_compose(project_root: str) -> bool:
    """Attempt to start the Ollama service using docker-compose."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker-compose", "up", "-d", "ollama",
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            LOG.error("docker-compose failed: %s", stderr.decode(errors="ignore"))
            return False
        return True
    except Exception as e:
        LOG.error("Failed to start Ollama: %s", e)
        return False


async def pull_model_with_docker(container: str, model: str) -> bool:
    """Pull a model inside an existing Ollama container via docker exec."""
    try:
        LOG.info("Pulling model %s in container %s", model, container)
        process = await asyncio.create_subprocess_exec(
            "docker", "exec", container, "ollama", "pull", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            LOG.error("ollama pull failed: %s", stderr.decode(errors="ignore"))
            return False
        return True
    except Exception as e:
        LOG.error("Model pull error: %s", e)
        return False


async def ensure_ollama_ready(
    base_url: str,
    model: str,
    project_root: str,
    container_name: str = "omniscient-architect-ollama",
    max_retries: int = 3,
    sleep_seconds: float = 5.0,
) -> bool:
    """Ensure Ollama server is reachable and model is available.

    Retries with exponential backoff when server/model are not yet ready.
    """
    # Ensure server
    for attempt in range(1, max_retries + 1):
        if await check_server(base_url):
            break
        LOG.info("Ollama server not responding (attempt %d/%d); trying to start...", attempt, max_retries)
        await start_ollama_with_compose(project_root)
        await asyncio.sleep(sleep_seconds * attempt)
    else:
        LOG.error("Ollama server not responding after retries")
        return False

    # Ensure model
    for attempt in range(1, max_retries + 1):
        if await check_model_available(base_url, model):
            return True
        LOG.info("Model %s not available (attempt %d/%d); pulling...", model, attempt, max_retries)
        await pull_model_with_docker(container_name, model)
        await asyncio.sleep(sleep_seconds * attempt)

    # Final check
    if await check_model_available(base_url, model):
        return True

    LOG.error("Model %s still not available after retries", model)
    return False
