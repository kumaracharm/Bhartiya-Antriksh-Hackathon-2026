"""Wrapper for inference-only infrared colorization."""

from __future__ import annotations

import numpy as np

from project_models.colorization.pix2pix import Pix2PixColorizer, colorize, colorize_image

__all__ = ["Pix2PixColorizer", "colorize", "colorize_image"]


def colorize(image: np.ndarray) -> np.ndarray:
    return _DEFAULT_COLORIZER.colorize(image)


_DEFAULT_COLORIZER = Pix2PixColorizer()