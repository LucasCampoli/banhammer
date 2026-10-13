"""OCR backend interface so the engine can be swapped."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class OCRResult:
    text: str
    confidence: float | None = None  # in [0, 1]; None if the engine gives none


class OCRBackend(Protocol):
    def read(self, image: np.ndarray, zone: str) -> OCRResult:
        ...
