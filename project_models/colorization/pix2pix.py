"""Robust inference-only colorization with Pix2Pix-first fallback stack."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "pix2pix.pth"


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.block = nn.Sequential(*layers)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class _TransposedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 3) -> None:
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.encoder2 = _ConvBlock(64, 128)
        self.encoder3 = _ConvBlock(128, 256)
        self.encoder4 = _ConvBlock(256, 512)
        self.encoder5 = _ConvBlock(512, 512)
        self.encoder6 = _ConvBlock(512, 512)
        self.encoder7 = _ConvBlock(512, 512)
        self.bottleneck = _ConvBlock(512, 512)

        self.decoder1 = _TransposedConvBlock(512, 512, use_dropout=True)
        self.decoder2 = _TransposedConvBlock(1024, 512, use_dropout=True)
        self.decoder3 = _TransposedConvBlock(1024, 512, use_dropout=True)
        self.decoder4 = _TransposedConvBlock(1024, 256)
        self.decoder5 = _TransposedConvBlock(512, 128)
        self.decoder6 = _TransposedConvBlock(256, 64)
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)
        bottleneck = self.bottleneck(enc7)

        dec1 = self.decoder1(bottleneck)
        dec2 = self.decoder2(torch.cat([dec1, enc7], dim=1))
        dec3 = self.decoder3(torch.cat([dec2, enc6], dim=1))
        dec4 = self.decoder4(torch.cat([dec3, enc5], dim=1))
        dec5 = self.decoder5(torch.cat([dec4, enc4], dim=1))
        dec6 = self.decoder6(torch.cat([dec5, enc3], dim=1))
        return self.decoder7(torch.cat([dec6, enc2], dim=1))


class Pix2PixColorizer:
    def __init__(self, weights_path: str | os.PathLike[str] | None = None, device: str | None = None) -> None:
        self.weights_path = Path(weights_path or os.environ.get("PIX2PIX_WEIGHTS_PATH", _DEFAULT_WEIGHTS))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: nn.Module | None = None
        self.loaded = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.weights_path.is_file():
                self.loaded = False
                self.model = None
                return
            model = Generator().to(self.device)
            checkpoint = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
            if isinstance(state_dict, dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
                model.load_state_dict(state_dict, strict=False)
            model.eval()
            self.model = model
            self.loaded = True
        except Exception as exc:
            LOGGER.warning("Pix2Pix colorization unavailable: %s", exc)
            self.model = None
            self.loaded = False

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((64, 64), dtype=np.uint8)
        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.shape[2] == 1:
            return image[..., 0].astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        gray = self._to_gray(image)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32) / 255.0
        gray = np.expand_dims(gray, axis=0)
        gray = np.expand_dims(gray, axis=0)
        return torch.from_numpy(gray).to(self.device)

    def _postprocess(self, output: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            image = output.squeeze(0).cpu().float().clamp(-1.0, 1.0).numpy()
        image = (image.transpose(1, 2, 0) + 1.0) / 2.0
        image = np.clip(image, 0.0, 1.0)
        return (image * 255.0).astype(np.uint8)

    def colorize(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            if not self.loaded or self.model is None:
                return self._fallback_colorize(image)
            tensor = self._prepare_image(image)
            with torch.no_grad():
                output = self.model(tensor)
            result = self._postprocess(output)
            if result.ndim == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            LOGGER.warning("Pix2Pix inference failed: %s", exc)
            return self._fallback_colorize(image)

    def _fallback_colorize(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image)
        if gray.ndim == 2:
            colorized = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        return np.dstack([gray.astype(np.uint8)] * 3)


def colorize(image: np.ndarray) -> np.ndarray:
    return _DEFAULT_COLORIZER.colorize(image)


_DEFAULT_COLORIZER = Pix2PixColorizer()


def colorize_image(image_path: str | os.PathLike[str]) -> str:
    input_path = Path(image_path)
    output_path = Path("outputs") / "colorized.jpg"
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

    result = colorize(image)
    cv2.imwrite(str(output_path), result)
    return str(output_path)
