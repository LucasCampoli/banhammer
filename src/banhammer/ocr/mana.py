"""Symbol reading: template matching against Scryfall symbol images.

Used for the mana cost and for the symbols inside the rules text.
Not supported: three-way Phyrexian hybrids, half-mana and other odd symbols.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# symbol diameter / mana cost zone height
SCALE_RANGE = (0.58, 0.74)
# symbols that are printed in rules text but never in a mana cost
RULES_ONLY_SYMBOLS = frozenset({"T", "Q", "E", "P"})


@dataclass(frozen=True)
class Symbol:
    name: str
    score: float
    x: float  # center, in pixels of the searched image
    y: float
    diameter: int

    @property
    def left(self) -> float:
        return self.x - self.diameter / 2


@dataclass(frozen=True)
class ManaReading:
    cost: str
    confidence: float | None = None
    left: float | None = None  # x of the first symbol's left edge, in zone pixels


def format_mana_cost(symbols: list[str]) -> str:
    return "".join(f"{{{s}}}" for s in symbols)


def load_templates(directory: Path | str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # file names use _ instead of /  (W_B.png -> W/B)
    out = {}
    for f in sorted(Path(directory).glob("*.png")):
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        out[f.stem.replace("_", "/")] = (im[..., :3], im[..., 3])
    return out


def cost_templates(templates: dict) -> dict:
    return {name: t for name, t in templates.items() if name not in RULES_ONLY_SYMBOLS}


# Masked colour matching is ~40x slower than plain grayscale matching, so a cheap grayscale
# pass proposes candidates and only those are confirmed with the colour + mask comparison.
COARSE_SCORE = 0.4
CANDIDATES_PER_SPOT = 6
WINDOW = 6  # px around a coarse hit where the confirmation runs
PAD = 4
MIN_SEPARATION = 0.6  # two symbols overlap when their centers are closer than this many diameters

_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _scaled(templates: dict, name: str, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(BGR, 3-channel mask, gray over white) of a template at diameter d, cached."""
    key = (id(templates), name, d)
    if key not in _cache:
        bgr, alpha = templates[name]
        t = cv2.resize(bgr, (d, d), interpolation=cv2.INTER_AREA)
        a = cv2.resize(alpha, (d, d), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        over_white = (t * a[..., None] + 255 * (1 - a[..., None])).astype(np.uint8)
        _cache[key] = (t, cv2.merge([a, a, a]), cv2.cvtColor(over_white, cv2.COLOR_BGR2GRAY))
    return _cache[key]


def _clean(r: np.ndarray) -> np.ndarray:
    return np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)


def find_symbols(image_bgr: np.ndarray, templates: dict, sizes: list[int], min_score: float,
                 coarse_score: float = COARSE_SCORE) -> list[Symbol]:
    """All non-overlapping symbols in an image, best score first.

    A higher `coarse_score` sends fewer candidates to the slow colour check.
    """
    z = cv2.copyMakeBorder(image_bgr, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
    sizes = [d for d in sizes if z.shape[0] >= d and z.shape[1] >= d]

    # 1) grayscale pass: for each spot keep the best (name, size) hits
    spots: dict[tuple[int, int], dict[tuple, tuple[float, int, int]]] = {}
    for d in sizes:
        for name in templates:
            r = _clean(cv2.matchTemplate(gray, _scaled(templates, name, d)[2], cv2.TM_CCOEFF_NORMED))
            for y, x in zip(*np.where(r > coarse_score)):
                spot = (int((y + d / 2) // (d / 2)), int((x + d / 2) // (d / 2)))
                hits = spots.setdefault(spot, {})
                if r[y, x] > hits.get((name, d), (0.0, 0, 0))[0]:
                    hits[(name, d)] = (float(r[y, x]), int(x), int(y))

    # 2) confirm the best hits of each spot with colour + mask in a narrow window
    candidates: list[Symbol] = []
    for hits in spots.values():
        best = sorted(hits.items(), key=lambda kv: -kv[1][0])[:CANDIDATES_PER_SPOT]
        for (name, d), (_, x, y) in best:
            x0, x1 = max(0, x - WINDOW), min(z.shape[1], x + d + WINDOW)
            y0, y1 = max(0, y - WINDOW), min(z.shape[0], y + d + WINDOW)
            t, m, _ = _scaled(templates, name, d)
            r = _clean(cv2.matchTemplate(z[y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED, mask=m))
            dy, dx = np.unravel_index(r.argmax(), r.shape)
            if r[dy, dx] > min_score:
                candidates.append(Symbol(name, float(r[dy, dx]), x0 + dx + d / 2 - PAD, y0 + dy + d / 2 - PAD, d))

    return non_overlapping(candidates)


def non_overlapping(symbols: list[Symbol]) -> list[Symbol]:
    """Best-scoring symbols first, dropping any that overlaps one already kept."""
    picked: list[Symbol] = []
    for c in sorted(symbols, key=lambda s: -s.score):
        if all(np.hypot(c.x - p.x, c.y - p.y) > MIN_SEPARATION * max(c.diameter, p.diameter) for p in picked):
            picked.append(c)
    return picked


def recognize_mana_cost(zone_bgr: np.ndarray, templates: dict, min_score: float = 0.7) -> ManaReading:
    if not templates:
        return ManaReading("")
    h = zone_bgr.shape[0]
    sizes = list(range(round(h * SCALE_RANGE[0]), round(h * SCALE_RANGE[1]) + 1))
    picked = sorted(find_symbols(zone_bgr, templates, sizes, min_score), key=lambda s: s.x)
    # symbols are a contiguous run at the right edge; drop stray matches (name letters)
    run = picked[-1:]
    for p in reversed(picked[:-1]):
        if run[0].x - p.x <= 1.3 * max(run[0].diameter, p.diameter):
            run.insert(0, p)
        else:
            break
    if not run:
        return ManaReading("")
    return ManaReading(
        cost=format_mana_cost([s.name for s in run]),
        confidence=sum(s.score for s in run) / len(run),
        left=run[0].left,
    )
