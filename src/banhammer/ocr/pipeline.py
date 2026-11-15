"""OCR pipeline: normalize, crop zones, preprocess, read."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from banhammer.models import OCRCard, ZoneBox
from .flavor import ITALIC_SLANT, is_measurable, line_styles, slant, split_flavor
from .mana import ManaReading, Symbol, cost_templates, load_templates, recognize_mana_cost
from .normalize import normalize
from .preprocess import preprocess
from .tesseract import RAW_LINE, SINGLE_LINE, ReadResult, Tesseract, Word
from .type_line import TypeVocabulary, correct_type_line
from .text_symbols import bare_templates, find_text_symbols, join_words, replace_with_placeholders, restore_symbols
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
    for word in result.words:  # reading order: raw-line boxes can be wider than their word
        if word.center / scale >= right_limit:
            continue
        if groups and word.left - (groups[-1][-1].left + groups[-1][-1].width) <= max_gap:
            groups[-1].append(word)
        else:
            groups.append([word])
    best = max(groups, key=lambda g: sum(ch.isalnum() for w in g for ch in w.text), default=[])
    return clean_text(" ".join(w.text for w in best))


def text_box_lines(result: ReadResult, min_line_confidence: float) -> list[list[Word]]:
    """Lines of the text box without the frame: box border bars and unreadable edge lines.

    Lines at the top or bottom whose mean confidence is below `min_line_confidence`
    are Tesseract reading the frame or the bottom bar, never rules text.
    """
    lines = [[w for w in line if w.text.strip("|")] for line in result.lines]
    lines = [line for line in lines if line]

    def confident(line: list[Word]) -> bool:
        return sum(w.confidence or 0.0 for w in line) / len(line) >= min_line_confidence

    while lines and not confident(lines[0]):
        lines.pop(0)
    while lines and not confident(lines[-1]):
        lines.pop()
    return lines


def split_text_box(lines: list[list[Word]], image: np.ndarray) -> tuple[list[str], list[str]]:
    """Split the text box lines into rules and flavor lines by the slant of their words."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    words = [[w.text for w in line] for line in lines]
    italic = [[slant(gray[w.top:w.top + w.height, w.left:w.left + w.width]) > ITALIC_SLANT
               if is_measurable(w.text) else None for w in line] for line in lines]
    start = split_flavor(line_styles(words, italic))
    texts = [clean_text(join_words(line)) for line in words]
    return texts[:start], texts[start:]


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
        self.zones_path = zones_path
        self._zones: dict[str | None, dict[str, ZoneBox]] = {}
        reader_cfg = dict(self.cfg.get("tesseract", {}))
        if reader_cfg.get("tessdata_dir"):
            reader_cfg["tessdata_dir"] = str(resolve_tessdata_dir(reader_cfg["tessdata_dir"],
                                                                  reader_cfg.get("lang", "eng")))
        self.reader = Tesseract(**reader_cfg)
        tdir = Path(self.cfg.get("mana", {}).get("templates_dir", "data/mana_templates"))
        self.templates = load_templates(tdir if tdir.is_absolute() else ROOT / tdir)
        self.cost_templates = cost_templates(self.templates)
        self.glyph_templates = bare_templates(self.templates)
        vocabulary = Path(self.cfg.get("type_vocabulary", "data/type_vocabulary.json"))
        self.type_vocabulary = TypeVocabulary.load(vocabulary if vocabulary.is_absolute() else ROOT / vocabulary)

    def zones(self, frame: str | None = None) -> dict[str, ZoneBox]:
        if frame not in self._zones:
            self._zones[frame] = load_zones(self.zones_path, frame)
        return self._zones[frame]

    def run(self, image: Path | str | np.ndarray, photo: bool | None = None,
            debug_dir: Path | str | None = None, frame: str | None = None) -> OCRCard:
        """OCR a card image; `frame` is the Scryfall frame ("2015", "2003"), None for the 2015 layout."""
        photo = self.cfg.get("perspective_correct", False) if photo is None else photo
        img = normalize(image, photo=photo)
        zones = self.zones(frame)
        crops = crop_zones(img, zones)
        debug = Path(debug_dir) if debug_dir else None
        if debug:
            debug.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug / "normalized.png"), img)
            cv2.imwrite(str(debug / "overlay.png"), draw_zone_overlay(img, zones))
            cv2.imwrite(str(debug / "mana_cost.png"), crops["mana_cost"])

        mana = recognize_mana_cost(crops["mana_cost"], self.cost_templates,
                                   self.cfg.get("mana", {}).get("min_score", 0.7))
        height, width = img.shape[:2]
        name_crop = self._name_crop(crops["name"], mana, width, zones)
        name, name_pre = self._read("name", name_crop, debug)
        type_line, type_pre = self._read("type_line", crops["type_line"], debug)
        pt, _ = self._read("power_toughness", crops["power_toughness"], debug)
        power, toughness = split_pt(clean_text(pt.text))
        rules_crop = crops["rules_text"]
        if power is not None:
            rules_crop = self._mask_pt_box(rules_crop, width, height, zones)
        symbols = self._text_symbols(rules_crop)
        rules, rules_pre = self._read("rules_text", replace_with_placeholders(rules_crop, symbols), debug)

        name_scale = name_pre.shape[1] / name_crop.shape[1]
        name_text = strip_edge_chars(line_text(name, name_scale, name_pre.shape[0]),
                                     NAME_LEADING_CHARS, NAME_TRAILING_CHARS)
        type_scale = type_pre.shape[1] / crops["type_line"].shape[1]
        set_symbol_x0 = (zones["set_symbol"].x0 - zones["type_line"].x0) * width
        type_text = correct_type_line(strip_edge_chars(line_text(type_line, type_scale, type_pre.shape[0],
                                                                 set_symbol_x0)), self.type_vocabulary)
        rules.lines = restore_symbols(rules.lines, symbols, rules_pre.shape[1] / rules_crop.shape[1])
        box_lines = text_box_lines(rules, self.cfg.get("min_line_confidence", 0.5))
        rules_lines, flavor_lines = split_text_box(box_lines, rules_pre)
        conf = {"name": name.confidence, "mana_cost": mana.confidence, "type_line": type_line.confidence,
                "rules_text": rules.confidence, "power": pt.confidence, "toughness": pt.confidence}
        return OCRCard(name=name_text, mana_cost=mana.cost, type_line=type_text,
                       rules_text="\n".join(rules_lines), flavor_text="\n".join(flavor_lines),
                       power=power, toughness=toughness, confidence=conf)

    def _text_symbols(self, rules_crop: np.ndarray) -> list[Symbol]:
        cfg = self.cfg.get("text_symbols", {})
        smallest, largest = cfg.get("diameters", [20, 27])
        return find_text_symbols(rules_crop, self.templates, self.glyph_templates, list(range(smallest, largest + 1)),
                                 cfg.get("min_score", 0.86), cfg.get("min_glyph_score", 0.8),
                                 cfg.get("min_disk_contrast", 15.0))

    @staticmethod
    def _mask_pt_box(rules_crop: np.ndarray, width: int, height: int, zones: dict[str, ZoneBox]) -> np.ndarray:
        """Paint over the part of the P/T box inside the text box, so its frame is not read as text."""
        rules, pt = zones["rules_text"], zones["power_toughness"]
        x0, y0 = int((pt.x0 - rules.x0) * width), int((pt.y0 - rules.y0) * height)
        masked = rules_crop.copy()
        masked[max(0, y0):, max(0, x0):] = np.median(rules_crop.reshape(-1, rules_crop.shape[-1]), axis=0)
        return masked

    @staticmethod
    def _name_crop(crop: np.ndarray, mana: ManaReading, width: int, zones: dict[str, ZoneBox]) -> np.ndarray:
        """Cut the name zone where the mana cost starts, so symbols are not read as letters."""
        if mana.left is None:
            return crop
        offset = (zones["mana_cost"].x0 - zones["name"].x0) * width
        return crop[:, :max(1, int(offset + mana.left) - NAME_MANA_GAP)]

    def _read(self, zone: str, crop: np.ndarray, debug: Path | None) -> tuple[ReadResult, np.ndarray]:
        """Preprocess and read a crop; also return the preprocessed image that was read."""
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
        result = self.reader.read(pre, zone)
        if not result.words and self.reader.page_mode(zone) == SINGLE_LINE:
            # A dark frame edge in the crop can make Tesseract find no line at all.
            result = self.reader.read(pre, zone, psm=RAW_LINE)
        return result, pre
