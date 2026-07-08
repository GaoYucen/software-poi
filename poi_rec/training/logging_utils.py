from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_run_logger(name: str, log_path: str | Path, *, log_to_console: bool = True) -> logging.Logger:
    """Create a run-specific logger that writes concise epoch summaries to disk.

    Repeated training/evaluation calls in tests may reuse the same logger name, so
    existing handlers are removed before attaching the current file/console sinks.
    """

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger