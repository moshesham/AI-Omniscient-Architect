"""Path setup utility for scripts and tests.

Provides a simple, compact helper to add package directories to sys.path.
Usage is intentionally simple to minimize boilerplate.

Example:
    # At top of script/test file:
    import sys
    from pathlib import Path
    _r = Path(__file__).resolve().parent.parent
    for _p in ["core", "llm", "rag"]:
        _path = _r / "packages" / _p / "src"
        if _path.exists(): sys.path.insert(0, str(_path))

The above 5-line pattern replaces 8-10 lines of boilerplate.
"""
