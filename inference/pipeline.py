"""BAH 2026 Infrared Enhancement + Colorization Pipeline"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import rasterio

from project_models.colorize import colorize
from project_models.enhancement.swinir import Enhancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

LOGGER = logging.getLogger(__name__)

INPUT_DIR = Path("input/product")

SR_OUTPUT = Path(
    "output/model_outputs/tir_superresolved_100m/product.tif"
)

COLOR_OUTPUT = Path(
    "output/model_outputs/colorized_tir_100m/product.tif"
)


def load_b10():

    b10_files = list(INPUT_DIR.glob("*_B10.TIF"))

    if len(b10_files) == 0:
        raise FileNotFoundError(
            f"No B10 file found inside {INPUT_DIR}"
        )

    b10 = b10_files[0]

    LOGGER.info(f"Loading B10: {b10}")

    with rasterio.open(b10) as src:

        image = src.read(1)

        metadata = src.meta.copy()

    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    image = image.astype(np.uint8)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    return image, metadata


def save_tif(image, path, metadata):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata.update(
        {
            "driver": "GTiff",
            "height": image.shape[0],
            "width": image.shape[1],
            "count": image.shape[2],
            "dtype": "uint8"
        }
    )

    with rasterio.open(
        path,
        "w",
        **metadata
    ) as dst:

        for i in range(image.shape[2]):
            dst.write(
                image[:, :, i],
                i + 1
            )

    LOGGER.info(
        f"Saved: {path}"
    )


def preprocess(image):

    image = cv2.GaussianBlur(
        image,
        (3,3),
        0
    )

    return image


def run_pipeline():

    try:

        image, metadata = load_b10()

        LOGGER.info(
            "Running enhancement..."
        )

        image = preprocess(image)

        enhanced = Enhancer().enhance(
            image
        )

        save_tif(
            enhanced,
            SR_OUTPUT,
            metadata
        )

        LOGGER.info(
            "Running colorization..."
        )

        colorized = colorize(
            enhanced
        )

        save_tif(
            colorized,
            COLOR_OUTPUT,
            metadata
        )

        LOGGER.info(
            "Pipeline complete"
        )

        return str(COLOR_OUTPUT)

    except Exception as e:

        LOGGER.exception(
            f"Pipeline failed: {e}"
        )

        raise