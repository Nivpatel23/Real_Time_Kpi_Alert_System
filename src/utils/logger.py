"""
Logging configuration with file rotation and console output.

WHY: Production systems need persistent, searchable logs. Console output
alone is lost when the process ends. File rotation prevents disk bloat.
Structured format enables log aggregation tools (ELK, CloudWatch, etc.).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import settings


def setup_logger(
    name: str,
    log_file: str = "logs/pipeline.log",
    level: str = None
) -> logging.Logger:
    """
    Create a configured logger with both file and console handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Path to log file
        level: Log level override (default: from settings)

    Returns:
        Configured logging.Logger instance
    """
    level = level or settings.pipeline.log_level

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler with rotation (5MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (for development / interactive runs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
