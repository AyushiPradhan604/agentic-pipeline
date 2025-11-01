# utils/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(name: str = "agentic_research_pipeline", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid adding handlers multiple times in interactive sessions
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(os.path.join(log_dir, f"{name}.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
