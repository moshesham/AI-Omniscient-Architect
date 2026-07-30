"""Tests for SmartPrioritizer – token-efficient repo inspection."""

import sys
from pathlib import Path
import pytest

# Ensure omniscient_tools is importable from the source tree
_root = Path(__file__).parent.parent
for _pkg in ["core", "tools"]:
    _path = _root / "packages" / _pkg / "src"
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from omniscient_tools.smart_prioritizer import (
    SmartPrioritizer,
    ScoredFile,
    SymbolInfo,
    _count_lines,
    _cyclomatic_proxy,
    _import_fan_out,
    _function_density,
    _todo_density,
    _extract_python_symbols,
    _extract_generic_symbols,
    _dedup_symbols,
)
from omniscient_core import FileAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fa(path: str, content: str, language: str = "Python") -> FileAnalysis:
    return FileAnalysis(path=path, content=content, language=language, size=len(content))


SIMPLE_PY = """\
import os
import sys

def greet(name):
    if name:
        return f"Hello {name}"
    else:
        return "Hello World"

class Greeter:
    def __init__(self, prefix):
        self.prefix = prefix

    def greet(self, name):
        for _ in range(3):
            if name:
                print(self.prefix + name)
"""

COMPLEX_PY = """\
import os, sys, re, json, pathlib

def alpha():
    for i in range(10):
        if i % 2 == 0:
            while i > 0:
                i -= 1
        elif i > 5:
            pass
        else:
            continue

def beta(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return None

class Outer:
    class Inner:
        def method(self):
            if True or False and True:
                pass
"""

JS_CONTENT = """\
import React from 'react';
import { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    if (count > 10) {
        return <div>Done</div>;
    }
    return <button onClick={() => setCount(count + 1)}>{count}</button>;
}

function Header({ title }) {
    return <h1>{title}</h1>;
}
"""

EMPTY_PY = ""


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

class TestHeuristics:
    def test_count_lines(self):
        assert _count_lines("a\nb\nc") == 3
        assert _count_lines("single") == 1
        assert _count_lines("") == 1

    def test_cyclomatic_proxy_empty(self):
        assert _cyclomatic_proxy("") == 0.0

    def test_cyclomatic_proxy_counts_keywords(self):
        code = "if x:\n  pass\nelif y:\n  pass\nfor i in r:\n  pass"
        score = _cyclomatic_proxy(code)
        assert score >= 3  # if, elif, for

    def test_import_fan_out(self):
        code = "import os\nfrom sys import path\nimport re"
        assert _import_fan_out(code) == 3.0

    def test_function_density_zero_for_empty(self):
        assert _function_density("x = 1") == 0.0

    def test_function_density_nonzero(self):
        code = "def a():\n    pass\ndef b():\n    pass"
        d = _function_density(code)
        assert d > 0

    def test_todo_density(self):
        code = "# TODO: fix this\n# FIXME later\nx = 1\n# HACK"
        assert _todo_density(code) == 3.0


# ---------------------------------------------------------------------------
# Python symbol extraction
# ---------------------------------------------------------------------------

class TestExtractPythonSymbols:
    def test_extracts_functions_and_classes(self):
        lines = SIMPLE_PY.splitlines()
        syms = _extract_python_symbols(SIMPLE_PY, lines)
        names = {s.name for s in syms}
        assert "greet" in names
        assert "Greeter" in names

    def test_kinds_are_correct(self):
        lines = SIMPLE_PY.splitlines()
        syms = _extract_python_symbols(SIMPLE_PY, lines)
        kinds = {s.name: s.kind for s in syms}
        assert kinds.get("greet") == "function"
        assert kinds.get("Greeter") == "class"

    def test_complex_file_higher_score(self):
        simple_lines = SIMPLE_PY.splitlines()
        complex_lines = COMPLEX_PY.splitlines()
        simple_syms = _extract_python_symbols(SIMPLE_PY, simple_lines)
        complex_syms = _extract_python_symbols(COMPLEX_PY, complex_lines)
        simple_total = sum(s.impact_score for s in simple_syms)
        complex_total = sum(s.impact_score for s in complex_syms)
        assert complex_total > simple_total

    def test_empty_source_returns_empty(self):
        assert _extract_python_symbols("", []) == []

    def test_syntax_error_returns_empty(self):
        bad = "def foo(\n  # unclosed"
        assert _extract_python_symbols(bad, bad.splitlines()) == []

    def test_source_field_populated(self):
        lines = SIMPLE_PY.splitlines()
        syms = _extract_python_symbols(SIMPLE_PY, lines)
        for s in syms:
            assert s.source.strip() != ""

    def test_line_numbers_valid(self):
        lines = SIMPLE_PY.splitlines()
        syms = _extract_python_symbols(SIMPLE_PY, lines)
        for s in syms:
            assert s.start_line >= 1
            assert s.end_line >= s.start_line


# ---------------------------------------------------------------------------
# Generic (non-Python) symbol extraction
# ---------------------------------------------------------------------------

class TestExtractGenericSymbols:
    def test_js_functions_found(self):
        lines = JS_CONTENT.splitlines()
        syms = _extract_generic_symbols(JS_CONTENT, lines)
        names = {s.name for s in syms}
        assert "Counter" in names or "Header" in names  # at least one

    def test_returns_list(self):
        lines = JS_CONTENT.splitlines()
        syms = _extract_generic_symbols(JS_CONTENT, lines)
        assert isinstance(syms, list)

    def test_empty_returns_empty(self):
        assert _extract_generic_symbols("", []) == []


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestDedupSymbols:
    def test_nested_removed(self):
        outer = SymbolInfo("outer", "function", 1, 20, "outer", 5.0)
        inner = SymbolInfo("inner", "function", 5, 10, "inner", 3.0)
        result = _dedup_symbols([outer, inner])
        names = {s.name for s in result}
        assert "outer" in names
        assert "inner" not in names

    def test_adjacent_both_kept(self):
        a = SymbolInfo("a", "function", 1, 10, "a", 2.0)
        b = SymbolInfo("b", "function", 11, 20, "b", 2.0)
        result = _dedup_symbols([a, b])
        assert len(result) == 2

    def test_empty_input(self):
        assert _dedup_symbols([]) == []


# ---------------------------------------------------------------------------
# SmartPrioritizer – file scoring
# ---------------------------------------------------------------------------

class TestScoreFile:
    def setup_method(self):
        self.sp = SmartPrioritizer()

    def test_empty_content_scores_zero(self):
        f = _fa("empty.py", "")
        score, _ = self.sp.score_file(f)
        assert score == 0.0

    def test_complex_scores_higher_than_simple(self):
        simple = _fa("simple.py", "x = 1\n")
        complex_ = _fa("complex.py", COMPLEX_PY)
        s_simple, _ = self.sp.score_file(simple)
        s_complex, _ = self.sp.score_file(complex_)
        assert s_complex > s_simple

    def test_churn_increases_score(self):
        f = _fa("f.py", SIMPLE_PY)
        s_no_churn, _ = self.sp.score_file(f, churn=0)
        s_with_churn, _ = self.sp.score_file(f, churn=50)
        assert s_with_churn > s_no_churn

    def test_breakdown_keys_present(self):
        f = _fa("f.py", SIMPLE_PY)
        _, breakdown = self.sp.score_file(f)
        for key in ("nloc", "cyclomatic", "fan_out", "function_density", "todo", "churn"):
            assert key in breakdown


# ---------------------------------------------------------------------------
# SmartPrioritizer – rank_files
# ---------------------------------------------------------------------------

class TestRankFiles:
    def setup_method(self):
        self.sp = SmartPrioritizer(max_files=5)

    def test_returns_sorted_descending(self):
        files = [
            _fa("a.py", "x = 1"),
            _fa("b.py", COMPLEX_PY),
            _fa("c.py", SIMPLE_PY),
        ]
        ranked = self.sp.rank_files(files)
        scores = [r.impact_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_max_files_respected(self):
        files = [_fa(f"f{i}.py", SIMPLE_PY) for i in range(20)]
        ranked = self.sp.rank_files(files)
        assert len(ranked) <= 5

    def test_files_without_content_excluded(self):
        files = [
            _fa("no_content.py", ""),
            _fa("has_content.py", SIMPLE_PY),
        ]
        ranked = self.sp.rank_files(files)
        paths = [r.path for r in ranked]
        assert "no_content.py" not in paths
        assert "has_content.py" in paths

    def test_churn_map_applied(self):
        files = [_fa("low.py", SIMPLE_PY), _fa("high.py", SIMPLE_PY)]
        ranked_no_churn = self.sp.rank_files(files)
        ranked_with_churn = self.sp.rank_files(files, churn_map={"high.py": 100.0})
        top_with_churn = ranked_with_churn[0].path
        assert top_with_churn == "high.py"

    def test_returns_scored_file_objects(self):
        ranked = self.sp.rank_files([_fa("f.py", SIMPLE_PY)])
        assert all(isinstance(r, ScoredFile) for r in ranked)


# ---------------------------------------------------------------------------
# SmartPrioritizer – extract_symbols
# ---------------------------------------------------------------------------

class TestExtractSymbols:
    def test_python_symbols_extracted(self):
        sp = SmartPrioritizer()
        ranked = sp.rank_files([_fa("f.py", SIMPLE_PY)])
        syms = sp.extract_symbols(ranked[0])
        assert len(syms) > 0
        assert all(isinstance(s, SymbolInfo) for s in syms)

    def test_js_symbols_extracted(self):
        sp = SmartPrioritizer()
        ranked = sp.rank_files([_fa("f.js", JS_CONTENT, language="JavaScript")])
        syms = sp.extract_symbols(ranked[0])
        assert isinstance(syms, list)

    def test_symbols_sorted_by_score_desc(self):
        sp = SmartPrioritizer()
        ranked = sp.rank_files([_fa("f.py", COMPLEX_PY)])
        syms = sp.extract_symbols(ranked[0])
        scores = [s.impact_score for s in syms]
        assert scores == sorted(scores, reverse=True)

    def test_symbols_stored_on_scored_file(self):
        sp = SmartPrioritizer()
        ranked = sp.rank_files([_fa("f.py", SIMPLE_PY)])
        sp.extract_symbols(ranked[0])
        assert ranked[0].symbols is not None


# ---------------------------------------------------------------------------
# SmartPrioritizer – build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_returns_string(self):
        sp = SmartPrioritizer(token_budget=1000)
        ctx = sp.build_context([_fa("f.py", SIMPLE_PY)])
        assert isinstance(ctx, str)

    def test_non_empty_for_content_files(self):
        sp = SmartPrioritizer(token_budget=2000)
        ctx = sp.build_context([_fa("f.py", COMPLEX_PY)])
        assert len(ctx) > 0

    def test_respects_token_budget(self):
        budget = 500
        sp = SmartPrioritizer(token_budget=budget)
        big_content = COMPLEX_PY * 50
        ctx = sp.build_context([_fa("f.py", big_content)])
        # Allow 10% overshoot due to header overhead
        assert len(ctx) <= budget * 4 * 1.1

    def test_empty_files_returns_empty(self):
        sp = SmartPrioritizer(token_budget=1000)
        ctx = sp.build_context([_fa("f.py", "")])
        assert ctx == ""

    def test_higher_impact_file_appears_first(self):
        sp = SmartPrioritizer(token_budget=8000, max_files=10)
        files = [
            _fa("simple.py", "x = 1"),
            _fa("complex.py", COMPLEX_PY),
        ]
        ctx = sp.build_context(files)
        complex_pos = ctx.find("complex.py")
        simple_pos = ctx.find("simple.py")
        # complex.py should appear before simple.py in the output
        assert complex_pos < simple_pos or simple_pos == -1

    def test_file_header_included(self):
        sp = SmartPrioritizer(token_budget=2000)
        ctx = sp.build_context([_fa("myfile.py", SIMPLE_PY)])
        assert "myfile.py" in ctx

    def test_multiple_files(self):
        sp = SmartPrioritizer(token_budget=8000, max_files=10)
        files = [_fa(f"f{i}.py", SIMPLE_PY) for i in range(5)]
        ctx = sp.build_context(files)
        # At least some files should appear
        count = sum(1 for i in range(5) if f"f{i}.py" in ctx)
        assert count >= 1


# ---------------------------------------------------------------------------
# SmartPrioritizer – get_impact_report
# ---------------------------------------------------------------------------

class TestGetImpactReport:
    def test_returns_list_of_dicts(self):
        sp = SmartPrioritizer()
        report = sp.get_impact_report([_fa("f.py", SIMPLE_PY)])
        assert isinstance(report, list)
        assert len(report) > 0
        assert "path" in report[0]
        assert "impact_score" in report[0]
        assert "breakdown" in report[0]
        assert "top_symbols" in report[0]
        assert "symbol_count" in report[0]

    def test_sorted_descending(self):
        sp = SmartPrioritizer(max_files=10)
        files = [_fa("a.py", "x=1"), _fa("b.py", COMPLEX_PY)]
        report = sp.get_impact_report(files)
        scores = [r["impact_score"] for r in report]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# BaseAIAgent integration
# ---------------------------------------------------------------------------

class TestBaseAgentIntegration:
    """Verify that BaseAIAgent.prepare_files_context delegates correctly."""

    def _make_agent(self):
        """Create a minimal concrete BaseAIAgent."""
        from omniscient_core.base import BaseAIAgent, AgentResponse

        class _DummyAgent(BaseAIAgent):
            def get_prompt_template(self):
                return "{context} {objective} {files_info} {format_instructions}"

        return _DummyAgent(
            llm=None,
            name="test",
            description="test",
            analysis_focus="test",
        )

    def test_smart_prioritizer_path_returns_string(self):
        agent = self._make_agent()
        files = [_fa("f.py", SIMPLE_PY)]
        ctx = agent.prepare_files_context(files, use_smart_prioritizer=True)
        assert isinstance(ctx, str)

    def test_legacy_path_returns_string(self):
        agent = self._make_agent()
        files = [_fa("f.py", SIMPLE_PY)]
        ctx = agent.prepare_files_context(files, use_smart_prioritizer=False)
        assert isinstance(ctx, str)
        assert "f.py" in ctx

    def test_smart_path_non_empty(self):
        agent = self._make_agent()
        files = [_fa("complex.py", COMPLEX_PY)]
        ctx = agent.prepare_files_context(
            files, token_budget=4000, use_smart_prioritizer=True
        )
        assert len(ctx) > 0
