"""
core/logger.py
Structured, colourised logger for Apex Engine 2.0.
All modules import this and call get_logger(__name__).
"""
import logging
import logging.handlers
import os
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console

_console = Console(stderr=True, legacy_windows=False)
_initialized = False


def _init_logging(log_file: str, level: str, max_bytes: int, backup_count: int) -> None:
    global _initialized
    if _initialized:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    # ── Root logger ────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(log_level)

    # ── Rich console handler (pretty terminal output) ──────────────
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(log_level)
    root.addHandler(rich_handler)

    # ── Rotating file handler ──────────────────────────────────────
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger for the given module name.
    Logging is initialised on first call using the loaded config.
    """
    from core.config_loader import get_config  # late import to avoid circular

    if not _initialized:
        cfg = get_config()
        log_cfg = cfg.get("logging", {})
        _init_logging(
            log_file=log_cfg.get("log_file", "logs/apex.log"),
            level=log_cfg.get("level", "INFO"),
            max_bytes=log_cfg.get("max_bytes", 10_485_760),
            backup_count=log_cfg.get("backup_count", 3),
        )

    return logging.getLogger(name)
