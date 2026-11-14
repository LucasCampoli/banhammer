"""OCR pipeline: normalize, crop zones, preprocess, read."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from banhammer.models import OCRCard
from .mana import ManaReading, load_templates, recognize_mana_cost
from .normalize import normalize
from .preprocess import preprocess
from .tesseract import ReadResult, Tesseract, Word
from .zones import DEFAULT_ZONES, crop_zones, draw_zone_overlay, load_zones

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "ocr.yaml"
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "¢": "•"})
NAME_MANA_GAP = 8  # px kept between the end of the name and the first mana symbol (and its shadow)
# Characters other than letters and digits that start or end a name in the OCR population.
NAME_LEADING_CHARS = "+"  # "+2 Mace"
NAME_TRAILING_CHARS = "!."  # "Shields Up!", "Agents of S.H.I.E.L.D."


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def clean_text(text: str, multiline: bool = False) -> str:
    text = text.translate(QUOTES)
    if multiline:
        return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip())
    return re.sub(r"\s+", " ", text).strip()


def strip_edge_chars(text: str, keep_leading: str = "", keep_trailing: str = "") -> str:
    """Drop frame and symbol debris at the ends of a name or type line.

    Names and type lines start and end with a letter or digit, except the
    characters passed in `keep_leading` and `keep_trailing`.
    """
    start, end = 0, len(text)
    while start < end and not (text[start].isalnum() or text[start] in keep_leading):
        start += 1
    while end > start and not (text[end - 1].isalnum() or text[end - 1] in keep_trailing):
        end -= 1
    return text[start:end]


def line_text(result: ReadResult, scale: float, max_gap: float, right_limit: float = float("inf")) -> str:
    """Text of a one-line zone without the words that belong to the frame or a symbol.

    Words centered right of `right_limit` (crop pixels; the read image is `scale`
    times the crop) are dropped. The remaining words are grouped while the gap
    between them is at most `max_gap` (read-image pixels), and only the group with
    the most letters and digits is kept: stray marks are far from the text.
    """
    groups: list[list[Word]] = []
    for word in sorted(result.words, key=lambda w: w.left):
        if word.center / scale >= right_limit:
            continue
        if groups and word.left - (groups[-1][-1].left + groups[-1][-1].width) <= max_gap:
            groups[-1].append(word)
        else:
            groups.append([word])
    best = max(groups, key=lambda g: sum(ch.isalnum() for w in g for ch in w.text), default=[])
    return clean_text(" ".join(w.text for w in best))


def resolve_tessdata_dir(tessdata_dir: str, lang: str) -> Path:
    path = Path(tessdata_dir)
    path = path if path.is_absolute() else ROOT / path
    if not (path / f"{lang}.traineddata").exists():
        raise FileNotFoundError(f"{path / f'{lang}.traineddata'} not found; run scripts/download_tessdata.py")
    return path


def split_pt(text: str) -> tuple[str | None, str | None]:
    m = re.match(r"^\s*([0-9*+-]+)\s*/\s*([0-9*+-]+)\s*$", text)
    return (m.group(1), m.group(2)) if m else (None, None)


class OCRPipeline:
    def __init__(self, config: dict | None = None, zones_path: Path | str = DEFAULT_ZONES):
        self.cfg = config or load_config()
        self.zones = load_zones(zones_path)
        reader_cfg = dict(self.cfg.get("tesseract", {}))
        if reader_cfg.get("tessdata_dir"):
            reader_cfg["tessdata_dir"] = str(resolve_tessdata_dir(reader_cfg["tessdata_dir"],
                                                                  reader_cfg.get("lang", "eng")))
        self.reader = Tesseract(**reader_cfg)
        tdir = Path(self.cfg.get("mana", {}).get("templates_dir", "data/mana_templates"))
        self.templates = load_templates(tdir if tdir.is_absolute() else ROOT / tdir)

    def run(self, image: Path | str | np.ndarray, photo: bool | None = None,
            debug_dir: Path | str | None = None) -> OCRCard:
        photo = self.cfg.get("perspective_correct", False) if photo is None else photo
        img = normalize(image, photo=photo)
        crops = crop_zones(img, self.zones)
        debug = Path(debug_dir) if debug_dir else None
        if debug:
            debug.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug / "normalized.png"), img)
            cv2.imwrite(str(debug / "overlay.png"), draw_zone_overlay(img, self.zones))
            cv2.imwrite(str(debug / "mana_cost.png"), crops["mana_cost"])

        mana = recognize_mana_cost(crops["mana_cost"], self.templates,
                                   self.cfg.get("mana", {}).get("min_score", 0.7))
        width = img.shape[1]
        name, name_scale = self._read("name", self._name_crop(crops["name"], mana, width), debug)
        type_line, type_scale = self._read("type_line", crops["type_line"], debug)
        rules, _ = self._read("rules_text", crops["rules_text"], debug)
        pt, _ = self._read("power_toughness", crops["power_toughness"], debug)

        name_gap = crops["name"].shape[0] * name_scale  # one line height
        name_text = strip_edge_chars(line_text(name, name_scale, name_gap), NAME_LEADING_CHARS, NAME_TRAILING_CHARS)
        set_symbol_x0 = (self.zones["set_symbol"].x0 - self.zones["type_line"].x0) * width
        type_gap = crops["type_line"].shape[0] * type_scale
        type_text = strip_edge_chars(line_text(type_line, type_scale, type_gap, set_symbol_x0))
        power, toughness = split_pt(clean_text(pt.text))
        conf = {"name": name.confidence, "mana_cost": mana.confidence, "type_line": type_line.confidence,
                "rules_text": rules.confidence, "power": pt.confidence, "toughness": pt.confidence}
        return OCRCard(name=name_text, mana_cost=mana.cost, type_line=type_text,
                       rules_text=clean_text(rules.text, multiline=True), power=power, toughness=toughness,
                       confidence=conf)

    def _name_crop(self, crop: np.ndarray, mana: ManaReading, width: int) -> np.ndarray:
        """Cut the name zone where the mana cost starts, so symbols are not read as letters."""
        if mana.left is None:
            return crop
        offset = (self.zones["mana_cost"].x0 - self.zones["name"].x0) * width
        return crop[:, :max(1, int(offset + mana.left) - NAME_MANA_GAP)]

    def _read(self, zone: str, crop: np.ndarray, debug: Path | None) -> tuple[ReadResult, float]:
        """Preprocess and read a crop; also return how many read-image pixels one crop pixel became."""
        if zone == "power_toughness":  # drop the box border, pad so Tesseract sees margins
            h, w = crop.shape[:2]
            crop = crop[int(h * .2):int(h * .8), int(w * .2):int(w * .85)]
        pre = preprocess(crop, self.cfg.get("preprocess", {}))
        scale = pre.shape[1] / crop.shape[1]
        if zone == "power_toughness":
            if pre.mean() < 110:  # light text on a dark box (vehicles): invert
                pre = 255 - pre
            pre = cv2.copyMakeBorder(pre, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        if debug:
            cv2.imwrite(str(debug / f"{zone}.png"), pre)
        return self.reader.read(pre, zone), scale
