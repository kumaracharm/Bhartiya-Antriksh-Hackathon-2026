"""Colorization inference interface for the hackathon pipeline."""

from __future__ import annotations

from pathlib import Path

from project_models.colorization.pix2pix import colorize_image


def run_colorization(image_path: str | Path) -> str:
    return colorize_image(str(image_path))
