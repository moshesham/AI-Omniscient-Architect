"""Smart Prioritizer – token-efficient repository inspection.

The goal is to answer one question as cheaply as possible:
  "Which lines of this codebase deserve the LLM's attention?"

The answer is produced entirely with the Python standard library – no LLM
calls, no heavy dependencies.  The engine has three layers:

1. **File scoring** – each file gets a numeric *impact score* from lightweight
   heuristics: lines of code, cyclomatic-complexity proxy, import fan-out, and
   optional git-churn weight.

2. **Symbol extraction** – within the top-ranked files the engine locates every
   function / method / class definition and scores each symbol by its own
   complexity proxy and call density.

3. **Token-budget context builder** – symbols are packed into a single context
   string in descending priority order until the caller's token budget is
   exhausted.  Callers trade a ``token_budget`` integer for a context string
   that fits within that budget, so the LLM never receives more than it needs.

Typical usage
-------------
::

    from omniscient_tools.smart_prioritizer import SmartPrioritizer
    from omniscient_core import FileAnalysis

    files = [FileAnalysis(path="app.py", content=..., language="Python", size=...)]

    sp = SmartPrioritizer(token_budget=3000)
    context = sp.build_context(files)

    # Inspect intermediate results
    ranked = sp.rank_files(files)
    symbols = sp.extract_symbols(ranked[0])
"""

from __future__ import annotations

import ast
import re
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sentinel for files the Python AST cannot parse (JS, TS, Go, etc.)
# ---------------------------------------------------------------------------
_UNPARSEABLE = object()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SymbolInfo:
    """A single scored symbol (function / class) extracted from a file."""

    name: str
    kind: str                     # "function" | "method" | "class"
    start_line: int
    end_line: int
    source: str                   # raw source lines for this symbol
    impact_score: float = 0.0    # higher = more important

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def token_estimate(self) -> int:
        """Rough token count: ~4 chars per token."""
        return max(1, len(self.source) // 4)


@dataclass
class ScoredFile:
    """A file together with its computed impact score and extracted symbols."""

    file: Any                      # FileAnalysis
    impact_score: float = 0.0
    symbols: List[SymbolInfo] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def language(self) -> str:
        return self.file.language


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

def _count_lines(text: str) -> int:
    return text.count("\n") + 1


def _cyclomatic_proxy(text: str) -> float:
    """Count branching keywords as a cheap cyclomatic-complexity proxy.

    Works on *any* language because it is purely regex-based.
    """
    pattern = re.compile(
        r"\b(if|elif|else|for|while|case|switch|catch|except|and|or|&&|\|\|)\b",
        re.MULTILINE,
    )
    return float(len(pattern.findall(text)))


def _import_fan_out(text: str) -> float:
    """Count import/require/include lines as a coupling indicator."""
    pattern = re.compile(
        r"^\s*(import|from|require|include|use)\b",
        re.MULTILINE,
    )
    return float(len(pattern.findall(text)))


def _function_density(text: str) -> float:
    """Count function/class definitions per 100 lines."""
    defs = re.findall(
        r"^\s*(def |class |function |func |fn |pub fn |async fn )",
        text,
        re.MULTILINE,
    )
    lines = max(1, _count_lines(text))
    return len(defs) / lines * 100


def _todo_density(text: str) -> float:
    """Count TODO/FIXME/HACK/XXX as a code-smell indicator."""
    return float(len(re.findall(r"\b(TODO|FIXME|HACK|XXX|BUG)\b", text, re.IGNORECASE)))


# ---------------------------------------------------------------------------
# Python AST-based symbol extraction
# ---------------------------------------------------------------------------

def _ast_symbol_score(node: ast.AST, source_lines: List[str]) -> float:
    """Score a Python AST function/class node."""
    score = 0.0

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body_source = "\n".join(
            source_lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)]
        )
        score += _cyclomatic_proxy(body_source)
        score += len(node.decorator_list) * 0.5
        # Long functions get a penalty-as-priority boost
        line_count = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
        score += math.log(max(1, line_count))
        # Count nested defs
        nested = sum(
            1
            for child in ast.walk(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child is not node
        )
        score += nested * 0.8

    elif isinstance(node, ast.ClassDef):
        method_count = sum(
            1
            for child in ast.walk(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        score += method_count * 0.5
        score += len(node.decorator_list) * 0.3
        score += len(node.bases) * 0.3

    return round(score, 2)


def _extract_python_symbols(
    content: str,
    source_lines: List[str],
) -> List[SymbolInfo]:
    """Extract top-level and class-level symbols from Python source."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    symbols: List[SymbolInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            raw = "\n".join(source_lines[node.lineno - 1 : end])
            kind = "class" if isinstance(node, ast.ClassDef) else "function"

            sym = SymbolInfo(
                name=node.name,
                kind=kind,
                start_line=node.lineno,
                end_line=end,
                source=raw,
                impact_score=_ast_symbol_score(node, source_lines),
            )
            symbols.append(sym)

    # De-duplicate: keep only the outer-most definition when nested
    # (e.g. inner functions are already counted in the parent's score)
    symbols = _dedup_symbols(symbols)
    return symbols


def _dedup_symbols(symbols: List[SymbolInfo]) -> List[SymbolInfo]:
    """Remove symbols that are fully contained within another symbol's range."""
    symbols_sorted = sorted(symbols, key=lambda s: (s.start_line, -(s.end_line)))
    result: List[SymbolInfo] = []
    current_end = -1
    for sym in symbols_sorted:
        if sym.start_line > current_end:
            result.append(sym)
            current_end = sym.end_line
    return result


# ---------------------------------------------------------------------------
# Generic (non-Python) symbol extraction
# ---------------------------------------------------------------------------

_GENERIC_DEF_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?:(?:public|private|protected|static|async|export|pub)\s+)*"
    r"(?:def|function|func|fn|sub|class|interface|struct|impl|trait)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_?!]*)",
    re.MULTILINE,
)


def _extract_generic_symbols(
    content: str,
    source_lines: List[str],
) -> List[SymbolInfo]:
    """Regex-based symbol extraction for non-Python languages."""
    symbols: List[SymbolInfo] = []

    matches = list(_GENERIC_DEF_RE.finditer(content))
    for idx, match in enumerate(matches):
        start_line = content[: match.start()].count("\n") + 1
        # Approximate end as the line before the next definition
        if idx + 1 < len(matches):
            end_line = content[: matches[idx + 1].start()].count("\n")
        else:
            end_line = len(source_lines)

        # Guard against empty ranges
        end_line = max(start_line, end_line)
        raw = "\n".join(source_lines[start_line - 1 : end_line])

        sym = SymbolInfo(
            name=match.group("name"),
            kind="function",
            start_line=start_line,
            end_line=end_line,
            source=raw,
            impact_score=round(_cyclomatic_proxy(raw) + math.log(max(1, end_line - start_line + 1)), 2),
        )
        symbols.append(sym)

    return _dedup_symbols(symbols)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SmartPrioritizer:
    """Ranks files and symbols for maximum LLM impact per token.

    Parameters
    ----------
    token_budget:
        Approximate number of tokens the caller can afford for the context
        string.  Tokens are estimated at 4 characters each.
    max_files:
        Upper bound on how many files enter symbol extraction.  Lower values
        keep costs down on very large repos.
    min_file_score:
        Files with a score below this threshold are skipped entirely.
    weights:
        Multipliers for each scoring component.  Override to tune for your
        codebase (e.g. bump ``churn`` weight when git history is available).
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "nloc": 0.3,          # raw line count (normalised)
        "cyclomatic": 2.0,    # branching keyword count
        "fan_out": 0.5,       # import count
        "function_density": 1.0,
        "todo": 0.4,
        "churn": 1.5,         # git change frequency (when provided)
    }

    def __init__(
        self,
        token_budget: int = 6000,
        max_files: int = 20,
        min_file_score: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.token_budget = token_budget
        self.max_files = max_files
        self.min_file_score = min_file_score
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}

    # ------------------------------------------------------------------
    # Layer 1 – file scoring
    # ------------------------------------------------------------------

    def score_file(
        self,
        file: Any,
        churn: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute impact score and breakdown for a single file.

        Parameters
        ----------
        file:
            A ``FileAnalysis`` object.  Must have ``.content`` (str | None)
            and ``.size`` (int).
        churn:
            Optional git churn value (number of commits touching the file).

        Returns
        -------
        (score, breakdown)
        """
        content: str = file.content or ""
        if not content:
            return 0.0, {}

        lines = _count_lines(content)
        cyclo = _cyclomatic_proxy(content)
        fan_out = _import_fan_out(content)
        fn_density = _function_density(content)
        todo = _todo_density(content)

        # Normalise nloc with a soft log so huge files don't dominate
        nloc_score = math.log(max(1, lines))

        breakdown = {
            "nloc": round(nloc_score, 2),
            "cyclomatic": round(cyclo, 2),
            "fan_out": round(fan_out, 2),
            "function_density": round(fn_density, 2),
            "todo": round(todo, 2),
            "churn": round(churn, 2),
        }

        w = self.weights
        score = (
            w["nloc"] * nloc_score
            + w["cyclomatic"] * cyclo
            + w["fan_out"] * fan_out
            + w["function_density"] * fn_density
            + w["todo"] * todo
            + w["churn"] * churn
        )
        return round(score, 3), breakdown

    def rank_files(
        self,
        files: List[Any],
        churn_map: Optional[Dict[str, float]] = None,
    ) -> List[ScoredFile]:
        """Score and rank *files* from highest to lowest impact.

        Parameters
        ----------
        files:
            List of ``FileAnalysis`` objects.
        churn_map:
            Optional ``{file_path: churn_value}`` mapping from git log.

        Returns
        -------
        List[ScoredFile] sorted descending by impact_score, capped at
        ``self.max_files``.
        """
        churn_map = churn_map or {}
        scored: List[ScoredFile] = []

        for f in files:
            if not f.content:
                continue
            churn = churn_map.get(f.path, 0.0)
            score, breakdown = self.score_file(f, churn)
            if score >= self.min_file_score:
                scored.append(
                    ScoredFile(file=f, impact_score=score, score_breakdown=breakdown)
                )

        scored.sort(key=lambda s: s.impact_score, reverse=True)
        return scored[: self.max_files]

    # ------------------------------------------------------------------
    # Layer 2 – symbol extraction
    # ------------------------------------------------------------------

    def extract_symbols(self, scored_file: ScoredFile) -> List[SymbolInfo]:
        """Extract and rank symbols from a single scored file.

        Python files use AST-based extraction for accuracy.  All other
        languages fall back to a regex heuristic.

        Returns symbols sorted descending by ``impact_score``.
        """
        content: str = scored_file.file.content or ""
        source_lines = content.splitlines()
        lang = (scored_file.language or "").lower()

        if "python" in lang:
            symbols = _extract_python_symbols(content, source_lines)
        else:
            symbols = _extract_generic_symbols(content, source_lines)

        symbols.sort(key=lambda s: s.impact_score, reverse=True)
        scored_file.symbols = symbols
        return symbols

    # ------------------------------------------------------------------
    # Layer 3 – token-budget context builder
    # ------------------------------------------------------------------

    def build_context(
        self,
        files: List[Any],
        churn_map: Optional[Dict[str, float]] = None,
        include_file_headers: bool = True,
        symbol_separator: str = "\n# --- next symbol ---\n",
    ) -> str:
        """Build the highest-impact context string within ``token_budget``.

        The algorithm:

        1. Rank all files by impact score.
        2. For each file (in rank order) extract and rank its symbols.
        3. Add symbols to the context buffer in rank order until the budget
           is exhausted.
        4. If a file has no extractable symbols the whole file content is
           added (subject to budget).

        Parameters
        ----------
        files:
            Raw ``FileAnalysis`` objects (content must be loaded).
        churn_map:
            Optional git churn data.
        include_file_headers:
            Prepend a one-line header before each file's block.
        symbol_separator:
            String inserted between consecutive symbols.

        Returns
        -------
        str
            A context string estimated to fit within ``token_budget`` tokens.
        """
        ranked = self.rank_files(files, churn_map)
        for sf in ranked:
            self.extract_symbols(sf)

        char_budget = self.token_budget * 4  # 4 chars ≈ 1 token
        parts: List[str] = []
        used_chars = 0

        for sf in ranked:
            if used_chars >= char_budget:
                break

            header = ""
            if include_file_headers:
                breakdown_str = ", ".join(
                    f"{k}={v}" for k, v in sf.score_breakdown.items()
                )
                header = (
                    f"# File: {sf.path} | lang={sf.language} "
                    f"| impact={sf.impact_score} ({breakdown_str})\n"
                )

            if sf.symbols:
                file_parts: List[str] = []
                for sym in sf.symbols:
                    snippet = (
                        f"# {sym.kind}: {sym.name}  "
                        f"[lines {sym.start_line}-{sym.end_line}, "
                        f"score={sym.impact_score}]\n"
                        + sym.source
                    )
                    needed = len(header) + len(snippet) + len(symbol_separator)
                    if used_chars + needed > char_budget:
                        break
                    file_parts.append(snippet)
                    used_chars += len(snippet) + len(symbol_separator)

                if file_parts:
                    block = header + symbol_separator.join(file_parts)
                    parts.append(block)
                    used_chars += len(header)
            else:
                # No symbols extracted: fall back to raw content (truncated)
                remaining = char_budget - used_chars - len(header)
                if remaining > 100:
                    raw = sf.file.content[:remaining]
                    parts.append(header + raw)
                    used_chars += len(header) + len(raw)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_impact_report(
        self,
        files: List[Any],
        churn_map: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Return a structured report of file impact scores.

        Useful for displaying a ranked file list in the UI *before* sending
        anything to the LLM.

        Returns
        -------
        List of dicts with keys: path, language, impact_score, breakdown,
        symbol_count, top_symbols.
        """
        ranked = self.rank_files(files, churn_map)
        report = []
        for sf in ranked:
            self.extract_symbols(sf)
            top_syms = [
                {"name": s.name, "kind": s.kind, "score": s.impact_score}
                for s in sf.symbols[:5]
            ]
            report.append(
                {
                    "path": sf.path,
                    "language": sf.language,
                    "impact_score": sf.impact_score,
                    "breakdown": sf.score_breakdown,
                    "symbol_count": len(sf.symbols),
                    "top_symbols": top_syms,
                }
            )
        return report
