"""Teacher model registry."""

import torch
from pathlib import Path
from typing import Optional
import os

from teachers.base import Teacher, GeminiTeacher, QwenTeacher
from system.config import config


def get_default_device() -> str:
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_teacher(teacher_name: str, cache_dir: Optional[Path] = None, **kwargs) -> Teacher:
    """Get teacher model by name.

    Args:
        teacher_name: "auto", "gemini", "qwen", etc.
        cache_dir: Optional cache directory for responses
        **kwargs: Additional arguments passed to teacher constructor

    Returns:
        Teacher instance
    """
    teacher_name = teacher_name.lower()

    # Handle auto-selection based on environment
    if teacher_name == "auto":
        model_name = config.active_teacher
        # Identify model family from name if possible, or assume Qwen
        if "gemini" in model_name.lower():
            teacher_name = "gemini"
        else:
            teacher_name = "qwen"
            kwargs["model_name"] = model_name

    if teacher_name in ["gemini", "gemini-flash", "gemini-pro"]:
        api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Get one from: https://aistudio.google.com/apikey"
            )

        # Fallback for Gemini if not in ModelSettings
        model_name = kwargs.get("model_name", "gemini-3-flash-preview")
        if teacher_name == "gemini-pro":
            model_name = "gemini-1.5-pro"

        return GeminiTeacher(
            api_key=api_key,
            model_name=model_name,
            cache_dir=cache_dir,
        )

    # Qwen family - QwenTeacher now handles config defaults internally
    elif "qwen" in teacher_name:
        return QwenTeacher(
            model_name=kwargs.get("model_name"),
            cache_dir=cache_dir,
            device=kwargs.get("device"),
            load_in_4bit=kwargs.get("load_in_4bit"),
            load_in_8bit=kwargs.get("load_in_8bit"),
        )

    else:
        raise ValueError(f"Unknown teacher: {teacher_name}. Available: auto, gemini, qwen")


# Convenience functions
def get_gemini_teacher(cache_dir: Optional[Path] = None, api_key: Optional[str] = None) -> Teacher:
    """Get Gemini teacher (requires API key)."""
    return get_teacher("gemini", cache_dir=cache_dir, api_key=api_key)


def get_qwen_teacher(
    model_size: str = "8b",
    cache_dir: Optional[Path] = None,
    device: str = "cuda",
    load_in_8bit: bool = False,
) -> Teacher:
    """Get Qwen teacher (local model)."""
    return get_teacher(
        f"qwen-{model_size}",
        cache_dir=cache_dir,
        device=device,
        load_in_8bit=load_in_8bit,
    )
