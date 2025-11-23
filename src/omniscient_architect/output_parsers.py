"""Robust parsing utilities for agent outputs.

The parser attempts strict JSON parse first, validates against the Pydantic
schema, and falls back to heuristic extraction (bullet lines) when needed.
"""

from __future__ import annotations

import json
import logging
from typing import List

from .agent_output_schema import AgentOutput

LOG = logging.getLogger(__name__)


def _extract_bullets(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    findings: List[str] = []
    for l in lines:
        # common bullet markers
        if l.startswith(("- ", "* ", "• ", "•")):
            findings.append(l.lstrip("-*• "))
            continue

        # Skip header-like lines (e.g., "Analysis:", "Notes:")
        if l.endswith(":"):
            continue

        # heuristically treat short lines (< 200 chars) as findings
        if 0 < len(l) < 200:
            findings.append(l)
    return findings


def parse_agent_output(raw_output: str, agent_name: str) -> AgentOutput:
    """Parse raw agent output into `AgentOutput`.

    Returns a Pydantic `AgentOutput` object. On parse failure the returned
    object includes `parse_error` and preserves the `raw_output`.
    """
    # Try strict JSON first
    try:
        parsed = json.loads(raw_output)
        # If parsed is a dict and has expected fields, validate
        if isinstance(parsed, dict):
            parsed.setdefault("agent_name", agent_name)
            parsed.setdefault("findings", [])
            parsed.setdefault("evidence", [])
            try:
                ao = AgentOutput.model_validate(parsed)
                ao.raw_output = raw_output
                return ao
            except Exception as e:  # pydantic validation error
                LOG.debug("JSON parsed but validation failed: %s", e)
                # fall through to heuristics below
    except Exception:
        LOG.debug("Strict JSON parse failed for agent %s", agent_name)

    # Heuristic extraction
    try:
        findings = _extract_bullets(raw_output)
        # If the only extracted finding is the entire raw output, treat as unstructured
        parse_error = None
        if not findings:
            parse_error = "no findings extracted"
        elif len(findings) == 1 and findings[0].strip() == raw_output.strip():
            parse_error = "only unstructured text, no extracted findings"

        ao = AgentOutput(
            agent_name=agent_name,
            findings=findings,
            confidence=None,
            evidence=[],
            reasoning=None,
            raw_output=raw_output,
            parse_error=parse_error,
        )
        return ao
    except Exception as e:
        LOG.exception("Unexpected error while parsing agent output: %s", e)
        return AgentOutput(
            agent_name=agent_name,
            findings=[],
            confidence=0.0,
            evidence=[],
            reasoning=None,
            raw_output=raw_output,
            parse_error=str(e),
        )


__all__ = ["parse_agent_output"]
