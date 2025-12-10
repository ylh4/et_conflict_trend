"""
Logging utilities for the Ethiopia conflict analysis pipeline.

Provides centralized logging configuration with consistent formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for the pipeline.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, logs only to console.
        log_format: Optional custom format string. If None, uses default format.
    
    Returns:
        Configured logger instance.
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    logger = logging.getLogger("ethiopia_conflict_analysis")
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
    
    Returns:
        Logger instance with the specified name.
    """
    return logging.getLogger(f"ethiopia_conflict_analysis.{name}")

