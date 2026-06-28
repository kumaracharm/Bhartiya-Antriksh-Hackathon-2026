"""SwinIR x2 super-resolution with safe fallbacks."""

from __future__ import annotations

import importlib.util
import os
import urllib.request
from pathlib import Path
from typing import Any, Type

import cv2
import numpy as np
import torch

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "SwinIR"
_NETWORK_PATH = _VENDOR_ROOT / "models" / "network_swinir.py"
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "swinir_x2.pth"
_OFFICIAL_WEIGHT_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
)

LIGHTWEIGHT_X2_CONFIG: dict[str, Any] = {
    "upscale": 2,
    "in_chans": 3,
    "img_size": 64,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6],
    "embed_dim": 60,
    "num_heads": [6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffledirect",
    "resi_connection": "1conv",
}


def _load_swinir_class() -> Type[torch.nn.Module]:
    if not _NETWORK_PATH.is_file():
        raise FileNotFoundError(f"SwinIR source missing: {_NETWORK_PATH}")
    spec = importlib.util.spec_from_file_location("swinir_network", _NETWORK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SwinIR from {_NETWORK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def _ensure_weights(weights_path: Path) -> Path:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.is_file():
        return weights_path
    env_path = os.environ.get("SWINIR_WEIGHTS_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    try:
        print(f"Downloading SwinIR x2 weights to {weights_path} ...")
        urllib.request.urlretrieve(_OFFICIAL_WEIGHT_URL, weights_path)
    except Exception as exc:
        print(f"Weight download failed ({exc}); bicubic fallback will be used.")
    return weights_path


def _load_state_dict(weights_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("params", "params_ema", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    return image


def _bicubic_upscale(image: np.ndarray, scale: int = 2) -> np.ndarray:
    image = _to_bgr(image)
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def _prepare_tensor(image: np.ndarray, device: torch.device) -> tuple[torch.Tensor, int, int]:
    image = _to_bgr(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
    tensor = np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))
    height, width = tensor.shape[1], tensor.shape[2]
    return torch.from_numpy(tensor).float().unsqueeze(0).to(device), height, width


def _tensor_to_bgr(output: torch.Tensor) -> np.ndarray:
    array = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=0)
    array = np.transpose(array[[2, 1, 0], :, :], (1, 2, 0))
    return (array * 255.0).round().astype(np.uint8)


def _pad_and_infer(tensor: torch.Tensor, model: torch.nn.Module, scale: int, window_size: int) -> torch.Tensor:
    _, _, height, width = tensor.size()
    pad_h = (height // window_size + 1) * window_size - height
    pad_w = (width // window_size + 1) * window_size - width
    tensor = torch.cat([tensor, torch.flip(tensor, [2])], 2)[:, :, : height + pad_h, :]
    tensor = torch.cat([tensor, torch.flip(tensor, [3])], 3)[:, :, :, : width + pad_w]
    with torch.no_grad():
        output = model(tensor)
    return output[..., : height * scale, : width * scale]


class Enhancer:
    """SwinIR x2 enhancer with bicubic fallback."""

    def __init__(self, weights_path: str | Path | None = None, scale: int = 2, device: str | None = None) -> None:
        self.scale = scale
        self.weights_path = Path(weights_path or _DEFAULT_WEIGHTS)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: torch.nn.Module | None = None
        self.using_fallback = False

    def _try_load_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            weights_path = _ensure_weights(self.weights_path)
            if not weights_path.is_file():
                self.using_fallback = True
                return False
            swinir_cls = _load_swinir_class()
            model = swinir_cls(**LIGHTWEIGHT_X2_CONFIG)
            model.load_state_dict(_load_state_dict(weights_path, self.device), strict=True)
            model.eval()
            self.model = model.to(self.device)
            return True
        except Exception as exc:
            print(f"SwinIR load failed ({exc}); using bicubic fallback.")
            self.using_fallback = True
            self.model = None
            return False

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return _bicubic_upscale(np.zeros((64, 64), dtype=np.uint8), self.scale)

        if not self._try_load_model():
            return _bicubic_upscale(image, self.scale)

        try:
            tensor, _, _ = _prepare_tensor(image, self.device)
            output = _pad_and_infer(tensor, self.model, self.scale, LIGHTWEIGHT_X2_CONFIG["window_size"])
            return _tensor_to_bgr(output)
        except Exception as exc:
            print(f"SwinIR inference failed ({exc}); using bicubic fallback.")
            self.using_fallback = True
            return _bicubic_upscale(image, self.scale)