"""
src/utils/logger.py
Centralised logging setup.  Import get_logger() in every module.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Pipeline started")
    log.debug("Frame %d | yaw=%.2f", frame_idx, yaw)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger that writes to stdout and to logs/isl.log.

    Args:
        name:  typically __name__ of the calling module.
        level: default DEBUG — all messages captured; INFO+ shown on console.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger          # already configured (module re-imports)

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — DEBUG and above (full trace)
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "isl.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
