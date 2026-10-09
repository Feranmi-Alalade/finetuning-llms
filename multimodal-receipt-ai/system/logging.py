"""
Centralized Logging Utility for Multimodal Receipt AI
=====================================================
Provides consistent logging across the entire project with:
- File rotation based on size (default: 10MB per file, 5 backups)
- Console output with color coding
- Task-based log files
- Third-party log suppression

Usage:
    from system.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Task started")
"""

import logging
import sys
import os
import multiprocessing
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


# ANSI color codes for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        # Format the message
        result = super().format(record)

        # Reset levelname for next use
        record.levelname = levelname
        return result


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: Optional[int] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with file rotation and console output.
    """
    from system.config import config

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    if level is None:
        level = getattr(logging, config.logging.level.upper(), logging.INFO)

    logger.setLevel(level)
    logger.propagate = False

    # Format strings
    detailed_format = (
        "%(asctime)s | %(name)-25s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
    )
    simple_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(simple_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_logging:
        log_path = config.paths.project_root / log_dir
        log_path.mkdir(parents=True, exist_ok=True)

        # Create a sanitized filename
        if name == "__main__":
            script_name = Path(sys.argv[0]).stem
            log_file = log_path / f"{script_name}.log"
        else:
            safe_name = name.split(".")[0]
            log_file = log_path / f"{safe_name}.log"

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(detailed_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Primary interface for getting loggers throughout the project.
    """
    return setup_logger(name, **kwargs)


def disable_third_party_logs(level: int = logging.WARNING):
    """Reduce verbosity of third-party library logs."""
    noisy_loggers = [
        "transformers",
        "datasets",
        "urllib3",
        "filelock",
        "huggingface_hub",
        "fsspec",
        "torch",
        "accelerate",
        "peft",
        "trl",
        "wandb",
        "mlflow",
        "PIL",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(level)


def get_wandb_logger(
    lineage: str,
    experiment_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
):
    """Create a W&B logger (minimal shim to maintain API)."""
    from pytorch_lightning.loggers import WandbLogger
    from system.config import config
    import wandb

    if config.logging.wandb_api_key:
        try:
            wandb.login(key=config.logging.wandb_api_key)
        except Exception:
            os.environ["WANDB_MODE"] = "offline"

    return WandbLogger(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=experiment_name or lineage,
        tags=[lineage] + (tags or []),
        save_dir=str(config.paths.project_root / "wandb_logs"),
    )
