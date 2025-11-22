"""Logging configuration for the Omniscient Architect."""

import logging
import sys
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging with Rich console output."""

    # Configure standard logging with Rich
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=False,
                show_path=False
            )
        ]
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)