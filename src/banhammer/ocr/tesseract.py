"""Read a preprocessed crop with Tesseract."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pytesseract
from pytesseract import Output

SINGLE_LINE = 7  # page segmentation mode: one text line
RAW_LINE = 13  # one text line, without Tesseract's own line detection


@dataclass(frozen=True)
class Word:
    text: str
    confidence: float | None
    left: int  # box in pixels of the image that was read
    top: int
    width: int
    height: int

    @property
    def center(self) -> float:
        return self.left + self.width / 2


@dataclass
class ReadResult:
    text: str
    confidence: float | None = None
    lines: list[list[Word]] = field(default_factory=list)

    @property
    def words(self) -> list[Word]:
        return [word for line in self.lines for word in line]


class Tesseract:
    def __init__(self, lang: str = "eng", psm: dict[str, int] | None = None,
                 whitelist: dict[str, str] | None = None, tessdata_dir: str | None = None):
        self.lang = lang
        self.tessdata_dir = tessdata_dir
        self.psm = psm or {}
        self.whitelist = whitelist or {}
        if os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

    def page_mode(self, zone: str) -> int:
        return self.psm.get(zone, 6)

    def read(self, image: np.ndarray, zone: str, psm: int | None = None) -> ReadResult:
        """Read a zone with its configured page segmentation mode, or `psm` if given."""
        config = f"--psm {psm or self.page_mode(zone)}"
        if self.tessdata_dir:
            config += f' --tessdata-dir "{self.tessdata_dir}"'
        if zone in self.whitelist:
            config += f" -c tessedit_char_whitelist={self.whitelist[zone]}"
        data = pytesseract.image_to_data(image, lang=self.lang, config=config, output_type=Output.DICT)
        lines: dict[tuple, list[Word]] = {}
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            conf = float(data["conf"][i])
            lines.setdefault(key, []).append(Word(text, conf / 100.0 if conf >= 0 else None, data["left"][i],
                                                  data["top"][i], data["width"][i], data["height"][i]))
        confs = [w.confidence for line in lines.values() for w in line if w.confidence is not None]
        text = "\n".join(" ".join(w.text for w in line) for line in lines.values())
        return ReadResult(text, sum(confs) / len(confs) if confs else None, list(lines.values()))
