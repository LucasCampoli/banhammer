"""OCR pipeline: normalize, crop zones, preprocess, read."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from banhammer.models import OCRCard
from .mana import load_templates, recognize_mana_cost
from .normalize import normalize
from .preprocess import preprocess
from .tesseract import Tesseract
from .zones import DEFAULT_ZONES, crop_zones, draw_zone_overlay, load_zones

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "ocr.yaml"
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "¢": "•"})
TEXT_ZONES = ("name", "type_line", "rules_text", "power_toughness")


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def clean_text(text: str, multiline: bool = False) -> str:
    text = text.translate(QUOTES)
    if multiline:
        return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip())
    text = re.sub(r"\s+", " ", text).strip()
    # junk at the edges (frame, set symbol)
    text = re.sub(r"^\w{0,2}\)+\s*", "", text)
    return re.sub(r"^[^\w*]+|[\s|>»<«'`\-]+$", "", text)


def split_pt(text: str) -> tuple[str | None, str | None]:
    m = re.match(r"^\s*([0-9*+-]+)\s*/\s*([0-9*+-]+)\s*$", text)
    return (m.group(1), m.group(2)) if m else (None, None)


class OCRPipeline:
    def __init__(self, config: dict | None = None, zones_path: Path | str = DEFAULT_ZONES):
        self.cfg = config or load_config()
        self.zones = load_zones(zones_path)
        self.reader = Tesseract(**self.cfg.get("tesseract", {}))
        tdir = Path(self.cfg.get("mana", {}).get("templates_dir", "data/mana_templates"))
        self.templates = load_templates(tdir if tdir.is_absolute() else ROOT / tdir)

    def run(self, image: Path | str | np.ndarray, photo: bool | None = None,
            debug_dir: Path | str | None = None) -> OCRCard:
        photo = self.cfg.get("perspective_correct", False) if photo is None else photo
        img = normalize(image, photo=photo)
        crops = crop_zones(img, self.zones)
        fields: dict[str, str] = {}
        conf: dict[str, float | None] = {}
        debug = Path(debug_dir) if debug_dir else None
        if debug:
            debug.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug / "normalized.png"), img)
            cv2.imwrite(str(debug / "overlay.png"), draw_zone_overlay(img, self.zones))

        for zone in TEXT_ZONES:
            crop = crops[zone]
            if zone == "power_toughness":  # drop the box border, pad so Tesseract sees margins
                h, w = crop.shape[:2]
                crop = crop[int(h * .2):int(h * .8), int(w * .2):int(w * .85)]
            pre = preprocess(crop, self.cfg.get("preprocess", {}))
            if zone == "power_toughness":
                if pre.mean() < 110:  # light text on a dark box (vehicles): invert
                    pre = 255 - pre
                pre = cv2.copyMakeBorder(pre, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
            if debug:
                cv2.imwrite(str(debug / f"{zone}.png"), pre)
            res = self.reader.read(pre, zone)
            fields[zone] = clean_text(res.text, multiline=zone == "rules_text")
            conf[zone] = res.confidence

        mana, mconf = recognize_mana_cost(crops["mana_cost"], self.templates,
                                          self.cfg.get("mana", {}).get("min_score", 0.7))
        conf["mana_cost"] = mconf
        if debug:
            cv2.imwrite(str(debug / "mana_cost.png"), crops["mana_cost"])

        power, toughness = split_pt(fields["power_toughness"])
        conf["power"] = conf["toughness"] = conf.pop("power_toughness")
        return OCRCard(name=fields["name"], mana_cost=mana, type_line=fields["type_line"],
                       rules_text=fields["rules_text"], power=power, toughness=toughness, confidence=conf)
