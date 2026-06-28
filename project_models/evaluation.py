"""Evaluation helpers for enhancement and colorization outputs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    return image


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)
    mse = np.mean((img_a - img_b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)
    mu_a = cv2.GaussianBlur(img_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(img_b, (11, 11), 1.5)
    mu_a_sq = mu_a**2
    mu_b_sq = mu_b**2
    mu_ab = mu_a * mu_b
    sigma_a_sq = cv2.GaussianBlur(img_a**2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(img_b**2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(img_a * img_b, (11, 11), 1.5) - mu_ab
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / ((mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2))
    return float(np.mean(ssim_map))


def compute_lpips(img_a: np.ndarray, img_b: np.ndarray) -> float:
    try:
        import lpips
    except Exception:
        return float("nan")
    model = lpips.LPIPS(net="alex")
    with torch.no_grad():
        t0 = torch.from_numpy(np.transpose(img_a, (2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0)
        t1 = torch.from_numpy(np.transpose(img_b, (2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0)
        return float(model(t0, t1).item())


def profile_inference(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    start = time.perf_counter()
    output = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return {"elapsed_seconds": elapsed, "output": output}


def profile_memory() -> dict[str, Any]:
    if torch.cuda.is_available():
        return {"device": "cuda", "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2)}
    return {"device": "cpu", "allocated_mb": 0.0}


def evaluate_images(reference_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    reference = _load_image(reference_path)
    output = _load_image(output_path)
    return {
        "psnr": compute_psnr(reference, output),
        "ssim": compute_ssim(reference, output),
        "lpips": compute_lpips(reference, output),
        "timing": profile_inference(lambda: None),
        "memory": profile_memory(),
    }
