"""Minimal stubs for langchain_community package used in tests.

These stubs allow running unit tests without installing the full `langchain`
ecosystem. They should be harmless when real packages are installed.
"""

from .chat_models import ChatOllama

__all__ = ["ChatOllama"]
