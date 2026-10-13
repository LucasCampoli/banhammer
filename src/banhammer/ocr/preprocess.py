"""Zone preprocessing, configured in configs/ocr.yaml."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess(crop: np.ndarray, cfg: dict) -> np.ndarray:
    img = crop
    if cfg.get("grayscale", True) and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = cfg.get("upscale", 2)
    if scale and scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoise = cfg.get("denoise", "none")
    if denoise == "median":
        img = cv2.medianBlur(img, 3)
    elif denoise == "nlmeans":
        img = cv2.fastNlMeansDenoising(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, 10)
    thr = cfg.get("threshold", "none")
    if thr != "none":
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if thr == "otsu":
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif thr == "adaptive":
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        else:
            raise ValueError(f"unknown threshold: {thr}")
    return img
