"""Logging configuration for the Omniscient Architect."""

import logging
import sys
from typing import Optional

from rich.logging import RichHandler
from structlog import configure, processors, stdlib, write_to


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up structured logging with Rich console output."""

    # Configure structlog
    shared_processors = [
        processors.add_log_level,
        processors.TimeStamper(fmt="iso"),
        processors.JSONRenderer() if log_file else processors.KeyValueRenderer(),
    ]

    configure(
        processors=[
            stdlib.filter_by_level,
            stdlib.add_logger_name,
            stdlib.add_log_level,
            stdlib.PositionalArgumentsFormatter(),
            processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            processors.StackInfoRenderer(),
            processors.format_exc_info,
            processors.UnicodeDecoder(),
            write_to.RichHandler(console=None, rich_tracebacks=True),
        ],
        context_class=dict,
        logger_factory=stdlib.LoggerFactory(),
        wrapper_class=stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
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