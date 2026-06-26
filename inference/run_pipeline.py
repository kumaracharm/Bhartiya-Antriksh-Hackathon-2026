"""Hackathon pipeline entry point: load -> enhance -> colorize -> save."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from project_models.colorize import colorize
from project_models.enhance import Enhancer

DEFAULT_INPUT = Path("data/raw/test.jpg")
DEFAULT_OUTPUT = Path("outputs/final_output.jpg")


def load_image(path: Path = DEFAULT_INPUT) -> np.ndarray:
    path = Path(path)
    if path.is_file():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 1:
                return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
            return image

    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(synthetic, (64, 64), 40, 200, -1)
    cv2.imwrite(str(path), synthetic)
    print(f"Created synthetic input at {path}")
    return cv2.cvtColor(synthetic, cv2.COLOR_GRAY2BGR)


def save(image: np.ndarray, path: Path = DEFAULT_OUTPUT) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    print(f"Saved output to {path}")


def run_pipeline(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    weights_path: Path | None = None,
) -> np.ndarray:
    img = load_image(input_path)
    enhanced = Enhancer(weights_path=weights_path).enhance(img)
    final = colorize(enhanced)
    save(final, output_path)
    return final


if __name__ == "__main__":
    run_pipeline()