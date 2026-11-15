"""
Logging infrastructure for the pipeline.

Provides a singleton logger with console and rotating file handlers,
synchronised across all pipeline modules.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_FILE = LOGS_DIR / "pipeline_verbose.log"

_logger_instance = None


def setup_logging(logs_dir: Path = None):
    """
    (Re)configures the file logging destination for the current run.

    Replaces the existing file handler when called after the instrument-
    specific output directory is known.
    """
    global _LOG_FILE, _logger_instance
    if logs_dir is not None:
        _LOG_FILE = Path(logs_dir) / "pipeline_verbose.log"
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if _logger_instance is not None:
        # Remove old file handlers
        for handler in _logger_instance.handlers[:]:
            if isinstance(handler, RotatingFileHandler):
                _logger_instance.removeHandler(handler)
                handler.close()

    # Create the file handler (whether or not logger was already init'd)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if _logger_instance is not None:
        _logger_instance.addHandler(file_handler)


def get_logger(name="pipeline"):
    """
    Returns the singleton logger, attaching a file handler and console
    stream handler on first call. Subsequent calls reuse the same handlers.
    """
    global _logger_instance
    if _logger_instance is not None:
        return (
            _logger_instance.getChild(name) if name != "pipeline" else _logger_instance
        )

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (file handler added later by setup_logging)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    _logger_instance = logger
    return logger


def log_stage_header(logger, title):
    """
    Emits a visually-distinct stage banner into both streams.
    """
    bar = "=" * 80
    logger.info(bar)
    logger.info("  %s", title.upper())
    logger.info(bar)


def log_substage(logger, title):
    """
    Emits a sub-stage separator for finer navigation of the log.
    """
    logger.info("-" * 80)
    logger.info("  >> %s", title)
    logger.info("-" * 80)
