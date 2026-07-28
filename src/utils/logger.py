"""Structured logging utility for training and inference."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout and optionally to a rotating file.

    Args:
        name: Logger name (usually __name__ of the calling module).
        log_dir: Directory to write log files; None disables file logging.
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_path / f"{name.replace('.', '_')}_{timestamp}.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
