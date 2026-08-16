"""Locate an optional reviewed checkpoint bundled with a binary distribution."""

from __future__ import annotations

import os
import sys
from importlib import resources
from pathlib import Path

BUNDLED_MODEL_FILENAME = "mesh2cad-automate-v0.1.pt"
SEGMENTATION_ENGINES = ("auto", "deterministic", "neural")


def bundled_checkpoint() -> Path | None:
    """Return the PyInstaller/package model without requiring one in source releases."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidate = Path(frozen_root) / "mesh2cad" / "models" / BUNDLED_MODEL_FILENAME
        if candidate.is_file():
            return candidate
    try:
        candidate_resource = resources.files("mesh2cad").joinpath(
            "models", BUNDLED_MODEL_FILENAME
        )
        candidate = Path(str(candidate_resource))
    except (ModuleNotFoundError, TypeError):
        return None
    return candidate if candidate.is_file() else None


def default_checkpoint() -> Path | None:
    configured = os.environ.get("MESH2CAD_CHECKPOINT")
    return Path(configured) if configured else bundled_checkpoint()


def default_segmentation_engine(checkpoint: Path | None) -> str:
    configured = os.environ.get("MESH2CAD_SEGMENTATION_ENGINE")
    if configured in SEGMENTATION_ENGINES:
        return configured
    return "neural" if checkpoint is not None else "auto"
