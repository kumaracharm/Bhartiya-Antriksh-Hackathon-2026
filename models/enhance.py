"""SwinIR super-resolution inference for the hackathon pipeline."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Optional, Type

import cv2
import numpy as np
import torch

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "SwinIR"
_NETWORK_PATH = _VENDOR_ROOT / "models" / "network_swinir.py"


def _load_swinir_class() -> Type[torch.nn.Module]:
    if not _NETWORK_PATH.is_file():
        raise FileNotFoundError(
            f"SwinIR source not found at '{_NETWORK_PATH}'. "
            "Run: git clone https://github.com/JingyunLiang/SwinIR.git vendor/SwinIR"
        )
    spec = importlib.util.spec_from_file_location("swinir_network", _NETWORK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SwinIR module from '{_NETWORK_PATH}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


SwinIR = _load_swinir_class()

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "swinir_tir_x2.pth"

NOTEBOOK_TIR_CONFIG: dict[str, Any] = {
    "upscale": 2,
    "in_chans": 1,
    "img_size": 64,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 60,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
}

OFFICIAL_LIGHTWEIGHT_X2_CONFIG: dict[str, Any] = {
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


def _load_checkpoint(weights_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "params" in checkpoint:
            return checkpoint["params"]
        if "params_ema" in checkpoint:
            return checkpoint["params_ema"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _prepare_tensor(image: np.ndarray, in_chans: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    if image.ndim == 2:
        image = image[..., np.newaxis]

    if in_chans == 1:
        gray = _to_grayscale(image)
        if gray.dtype == np.uint8:
            tensor = gray.astype(np.float32) / 255.0
        else:
            tensor = gray.astype(np.float32)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        tensor = np.expand_dims(tensor, axis=0)
    else:
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
        tensor = np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))

    h_old, w_old = tensor.shape[1], tensor.shape[2]
    return torch.from_numpy(tensor).float().unsqueeze(0).to(device), h_old, w_old


def _tensor_to_numpy(output: torch.Tensor, in_chans: int, source_channels: int) -> np.ndarray:
    array = output.squeeze().float().cpu().clamp_(0, 1).numpy()

    if array.ndim == 2:
        array = np.expand_dims(array, axis=2)
    elif in_chans == 3:
        array = np.transpose(array[[2, 1, 0], :, :], (1, 2, 0))

    array = (array * 255.0).round().astype(np.uint8)

    if in_chans == 1 and source_channels >= 3:
        array = cv2.cvtColor(array[..., 0], cv2.COLOR_GRAY2BGR)
    elif in_chans == 1 and source_channels == 1:
        return array

    return array


def _pad_for_window(tensor: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    _, _, h_old, w_old = tensor.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    tensor = torch.cat([tensor, torch.flip(tensor, [2])], 2)[:, :, : h_old + h_pad, :]
    tensor = torch.cat([tensor, torch.flip(tensor, [3])], 3)[:, :, :, : w_old + w_pad]
    return tensor, h_old, w_old


def _run_model(tensor, model, scale, window_size, tile, tile_overlap):
    tensor, h_old, w_old = _pad_for_window(tensor, window_size)
    with torch.no_grad():
        if tile is None:
            output = model(tensor)
        else:
            output = _tile_inference(tensor, model, scale, window_size, tile, tile_overlap)
    return output[..., : h_old * scale, : w_old * scale]


def _tile_inference(tensor, model, scale, window_size, tile, tile_overlap):
    _, channels, height, width = tensor.size()
    tile = min(tile, height, width)
    if tile % window_size != 0:
        raise ValueError("tile size must be a multiple of window_size")

    stride = tile - tile_overlap
    h_idx_list = list(range(0, height - tile, stride)) + [height - tile]
    w_idx_list = list(range(0, width - tile, stride)) + [width - tile]

    output = torch.zeros(1, channels, height * scale, width * scale, device=tensor.device, dtype=tensor.dtype)
    weights = torch.zeros_like(output)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            patch = tensor[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            out_patch = model(patch)
            mask = torch.ones_like(out_patch)
            output[..., h_idx * scale : (h_idx + tile) * scale, w_idx * scale : (w_idx + tile) * scale].add_(out_patch)
            weights[..., h_idx * scale : (h_idx + tile) * scale, w_idx * scale : (w_idx + tile) * scale].add_(mask)

    return output.div_(weights)


class Enhancer:
    def __init__(self, weights_path=None, config="notebook_tir", device=None, tile=None, tile_overlap=32):
        self.weights_path = Path(weights_path or os.environ.get("SWINIR_WEIGHTS_PATH", DEFAULT_WEIGHTS))
        self.config_name = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.model_config = self._resolve_config(config)
        self.model = None

    @staticmethod
    def _resolve_config(config: str) -> dict[str, Any]:
        if config == "notebook_tir":
            return dict(NOTEBOOK_TIR_CONFIG)
        if config in {"lightweight_sr", "official_lightweight_x2"}:
            return dict(OFFICIAL_LIGHTWEIGHT_X2_CONFIG)
        raise ValueError("Unknown config. Use 'notebook_tir' or 'lightweight_sr'.")

    def _build_model(self) -> SwinIR:
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"SwinIR weights not found at '{self.weights_path}'. "
                "Export your Colab checkpoint with torch.save(model.state_dict(), 'swinir_tir_x2.pth') "
                "or set SWINIR_WEIGHTS_PATH."
            )
        model = SwinIR(**self.model_config)
        state_dict = _load_checkpoint(self.weights_path, self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model.to(self.device)

    def _ensure_model(self) -> SwinIR:
        if self.model is None:
            self.model = self._build_model()
        return self.model

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        source_channels = 1 if image.ndim == 2 else image.shape[2]
        model = self._ensure_model()
        in_chans = int(self.model_config["in_chans"])
        scale = int(self.model_config["upscale"])
        window_size = int(self.model_config["window_size"])

        tensor, _, _ = _prepare_tensor(image, in_chans, self.device)
        output = _run_model(tensor, model, scale, window_size, self.tile, self.tile_overlap)
        return _tensor_to_numpy(output, in_chans, source_channels)


_default_enhancer = None


def enhance(image: np.ndarray) -> np.ndarray:
    global _default_enhancer
    if _default_enhancer is None:
        _default_enhancer = Enhancer()
    return _default_enhancer.enhance(image)