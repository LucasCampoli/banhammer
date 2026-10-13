"""Tesseract backend."""
from __future__ import annotations

import os

import numpy as np
import pytesseract
from pytesseract import Output

from .base import OCRResult


class TesseractBackend:
    def __init__(self, lang: str = "eng", psm: dict[str, int] | None = None,
                 whitelist: dict[str, str] | None = None):
        self.lang, self.psm, self.whitelist = lang, psm or {}, whitelist or {}
        if os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

    def read(self, image: np.ndarray, zone: str) -> OCRResult:
        config = f"--psm {self.psm.get(zone, 6)}"
        if zone in self.whitelist:
            config += f" -c tessedit_char_whitelist={self.whitelist[zone]}"
        data = pytesseract.image_to_data(image, lang=self.lang, config=config, output_type=Output.DICT)
        lines: dict[tuple, list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            if float(data["conf"][i]) >= 0:
                confs.append(float(data["conf"][i]) / 100.0)
        text = "\n".join(" ".join(ws) for ws in lines.values())
        return OCRResult(text, sum(confs) / len(confs) if confs else None)
