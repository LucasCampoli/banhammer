"""Mana cost reading: template matching against Scryfall symbol images.

Not supported: three-way Phyrexian hybrids, half-mana and other odd symbols.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# symbol diameter / zone height
SCALE_RANGE = (0.58, 0.74)


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


# Masked colour matching is ~40x slower than plain grayscale matching, so a cheap grayscale
# pass proposes candidates and only those are confirmed with the colour + mask comparison.
COARSE_SCORE = 0.4
CANDIDATES_PER_SPOT = 6
WINDOW = 6  # px around a coarse hit where the confirmation runs

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


def recognize_mana_cost(zone_bgr: np.ndarray, templates: dict, min_score: float = 0.7
                        ) -> tuple[str, float | None]:
    if not templates:
        return "", None
    pad = 4
    z = cv2.copyMakeBorder(zone_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
    h = zone_bgr.shape[0]
    sizes = [d for d in range(round(h * SCALE_RANGE[0]), round(h * SCALE_RANGE[1]) + 1)
             if z.shape[0] >= d and z.shape[1] >= d]

    # 1) grayscale pass: for each horizontal spot keep the best (name, size) hits
    spots: dict[int, dict[tuple, tuple[float, int]]] = {}
    for d in sizes:
        for name in templates:
            r = _clean(cv2.matchTemplate(gray, _scaled(templates, name, d)[2], cv2.TM_CCOEFF_NORMED))
            for y, x in zip(*np.where(r > COARSE_SCORE)):
                hits = spots.setdefault(int((x + d / 2) // (d / 2)), {})
                if r[y, x] > hits.get((name, d), (0.0, 0))[0]:
                    hits[(name, d)] = (float(r[y, x]), int(x))

    # 2) confirm the best hits of each spot with colour + mask in a narrow window
    cands = []  # (score, x_center, diameter, name)
    for hits in spots.values():
        best = sorted(hits.items(), key=lambda kv: -kv[1][0])[:CANDIDATES_PER_SPOT]
        for (name, d), (_, x) in best:
            x0, x1 = max(0, x - WINDOW), min(z.shape[1], x + d + WINDOW)
            t, m, _ = _scaled(templates, name, d)
            r = _clean(cv2.matchTemplate(z[:, x0:x1], t, cv2.TM_CCOEFF_NORMED, mask=m))
            y, dx = np.unravel_index(r.argmax(), r.shape)
            if r[y, dx] > min_score:
                cands.append((float(r[y, dx]), x0 + dx + d / 2, d, name))
    # best score first, no overlapping symbols
    picked: list[tuple[float, float, int, str]] = []
    for c in sorted(cands, reverse=True):
        if all(abs(c[1] - p[1]) > 0.6 * max(c[2], p[2]) for p in picked):
            picked.append(c)
    picked.sort(key=lambda p: p[1])
    # symbols are a contiguous run at the right edge; drop stray matches (name letters)
    run = picked[-1:]
    for p in reversed(picked[:-1]):
        if run[0][1] - p[1] <= 1.3 * max(run[0][2], p[2]):
            run.insert(0, p)
        else:
            break
    picked = run
    if not picked:
        return "", None
    return format_mana_cost([p[3] for p in picked]), sum(p[0] for p in picked) / len(picked)
