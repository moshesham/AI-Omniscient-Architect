import json

from omniscient_architect.agent_output_schema import AgentOutput
from omniscient_architect.output_parsers import parse_agent_output


def test_parse_valid_json():
    raw = json.dumps({
        "agent_name": "architecture",
        "findings": ["modular design", "clear interfaces"],
        "confidence": 0.92,
        "evidence": ["src/omniscient_architect/models.py"],
    })
    ao = parse_agent_output(raw, "architecture")
    assert isinstance(ao, AgentOutput)
    assert ao.agent_name == "architecture"
    assert ao.findings == ["modular design", "clear interfaces"]
    assert ao.confidence == 0.92


def test_parse_malformed_bullets():
    raw = """
    Analysis:
    - Missing tests for critical paths
    - Use pydantic for schemas

    Notes:
    Consider adding CI checks
    """
    ao = parse_agent_output(raw, "efficiency")
    assert isinstance(ao, AgentOutput)
    assert ao.agent_name == "efficiency"
    assert "Missing tests for critical paths" in ao.findings[0]


def test_parse_unstructured_text_returns_parse_error_if_no_findings():
    raw = "This is a single very long paragraph with no clear bullets or short lines "
    raw += """that could be sensibly split into findings; it should lead to a parse_error"""
    ao = parse_agent_output(raw, "alignment")
    assert isinstance(ao, AgentOutput)
    # parser will attempt to include short lines — if none, parse_error should be set
    assert ao.raw_output == raw
    assert ao.parse_error is not None
