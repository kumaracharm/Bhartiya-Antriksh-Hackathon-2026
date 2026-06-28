"""Enhancement inference interface for the hackathon pipeline."""

from __future__ import annotations

from pathlib import Path

from project_models.enhancement.swinir import enhance_image


def run_enhancement(image_path: str | Path) -> str:
    return enhance_image(str(image_path))
