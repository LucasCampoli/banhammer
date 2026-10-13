"""Image loading, resizing and optional perspective correction."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CARD_SIZE = (672, 936)  # (width, height), Scryfall "large"


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype="float32")


def perspective_correct(img: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * img.shape[0] * img.shape[1]:
            dst = np.array([[0, 0], [CARD_SIZE[0] - 1, 0], [CARD_SIZE[0] - 1, CARD_SIZE[1] - 1],
                            [0, CARD_SIZE[1] - 1]], dtype="float32")
            m = cv2.getPerspectiveTransform(order_corners(approx), dst)
            return cv2.warpPerspective(img, m, CARD_SIZE)
    return img


def normalize(path: Path | str | np.ndarray, photo: bool = False) -> np.ndarray:
    img = path if isinstance(path, np.ndarray) else cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if photo:
        img = perspective_correct(img)
    return cv2.resize(img, CARD_SIZE, interpolation=cv2.INTER_AREA if img.shape[1] > CARD_SIZE[0] else cv2.INTER_CUBIC)
