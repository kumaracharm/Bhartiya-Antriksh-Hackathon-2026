"""Robust inference-only infrared enhancement with SwinIR-first fallback stack."""

from __future__ import annotations

import importlib.util
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Type

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "SwinIR"
_NETWORK_PATH = _VENDOR_ROOT / "models" / "network_swinir.py"
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "swinir_x2.pth"
_OFFICIAL_WEIGHT_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    return image


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = _to_bgr(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image = image / 255.0
    return image


def _denoise(image: np.ndarray) -> np.ndarray:
    gray = _normalize_image(image)
    gray = (gray * 255.0).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised.astype(np.float32) / 255.0


def _edge_preserving(image: np.ndarray) -> np.ndarray:
    gray = (_normalize_image(image) * 255.0).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.15, blur, -0.15, 0)
    return sharpened.astype(np.float32) / 255.0


def _prepare_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    gray = _edge_preserving(image)
    tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def _tensor_to_bgr(output: torch.Tensor) -> np.ndarray:
    array = output.squeeze().float().cpu().clamp(0.0, 1.0).numpy()
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).round().astype(np.uint8)


def _ensure_weights(weights_path: Path) -> Path:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.is_file():
        return weights_path
    env_path = os.environ.get("SWINIR_WEIGHTS_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    try:
        LOGGER.info("Downloading SwinIR weights to %s", weights_path)
        urllib.request.urlretrieve(_OFFICIAL_WEIGHT_URL, weights_path)
    except Exception as exc:
        LOGGER.warning("SwinIR weights download failed: %s", exc)
    return weights_path


def _load_swinir_class() -> Type[torch.nn.Module]:
    if not _NETWORK_PATH.is_file():
        raise FileNotFoundError(f"Missing SwinIR source: {_NETWORK_PATH}")
    spec = importlib.util.spec_from_file_location("swinir_network", _NETWORK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SwinIR from {_NETWORK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


class SwinIREnhancer:
    def __init__(self, weights_path: str | os.PathLike[str] | None = None, device: str | None = None) -> None:
        self.weights_path = Path(weights_path or os.environ.get("SWINIR_WEIGHTS_PATH", _DEFAULT_WEIGHTS))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: torch.nn.Module | None = None
        self.loaded = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            weights_path = _ensure_weights(self.weights_path)
            if not weights_path.is_file():
                self.loaded = False
                self.model = None
                return
            sw = _load_swinir_class()
            model = sw(upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler="pixelshuffledirect", resi_connection="1conv")
            checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("params", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
            if isinstance(state_dict, dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
                model.load_state_dict(state_dict, strict=False)
            model.eval()
            self.model = model.to(self.device)
            self.loaded = True
        except Exception as exc:
            LOGGER.warning("SwinIR enhancement unavailable: %s", exc)
            self.model = None
            self.loaded = False

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            if not self.loaded or self.model is None:
                return self._fallback_enhance(image)
            tensor = _prepare_tensor(image, self.device)
            with torch.no_grad():
                output = self.model(tensor)
            result = _tensor_to_bgr(output)
            if result.ndim == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            return result
        except Exception as exc:
            LOGGER.warning("Inference failed: %s", exc)
            return self._fallback_enhance(image)

    def _fallback_enhance(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(_to_bgr(image), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.addWeighted(gray, 1.1, blurred, -0.1, 0)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def enhance(image: np.ndarray) -> np.ndarray:
    return _DEFAULT_ENHANCER.enhance(image)


_DEFAULT_ENHANCER = SwinIREnhancer()


def enhance_image(image_path: str | os.PathLike[str]) -> str:
    input_path = Path(image_path)
    output_path = Path("outputs") / "enhanced.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    result = enhance(image)
    cv2.imwrite(str(output_path), result)
    return str(output_path)
