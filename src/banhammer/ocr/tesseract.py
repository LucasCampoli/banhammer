"""Read a preprocessed crop with Tesseract."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class ReadResult:
    text: str
    confidence: float | None = None


class Tesseract:
    def __init__(self, lang: str = "eng", psm: dict[str, int] | None = None,
                 whitelist: dict[str, str] | None = None):
        self.lang = lang
        self.psm = psm or {}
        self.whitelist = whitelist or {}
        if os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

    def read(self, image: np.ndarray, zone: str) -> ReadResult:
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
        text = "\n".join(" ".join(words) for words in lines.values())
        return ReadResult(text, sum(confs) / len(confs) if confs else None)
