"""Test configuration to make the workspace `src/` importable during pytest runs."""
import os
import sys

# Ensure the project's `src` directory is on sys.path so tests can import packages.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
