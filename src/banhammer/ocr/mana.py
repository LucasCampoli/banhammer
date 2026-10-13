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


def recognize_mana_cost(zone_bgr: np.ndarray, templates: dict, min_score: float = 0.7
                        ) -> tuple[str, float | None]:
    if not templates:
        return "", None
    pad = 4
    z = cv2.copyMakeBorder(zone_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    h = zone_bgr.shape[0]
    sizes = range(round(h * SCALE_RANGE[0]), round(h * SCALE_RANGE[1]) + 1)
    cands = []  # (score, x_center, diameter, name)
    for d in sizes:
        if z.shape[0] < d or z.shape[1] < d:
            continue
        for name, (bgr, alpha) in templates.items():
            t = cv2.resize(bgr, (d, d), interpolation=cv2.INTER_AREA)
            m = cv2.resize(alpha, (d, d), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            r = cv2.matchTemplate(z, t, cv2.TM_CCOEFF_NORMED, mask=cv2.merge([m, m, m]))
            r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
            ys, xs = np.where(r > min_score)
            cands.extend((float(r[y, x]), x + d / 2, d, name) for y, x in zip(ys, xs))
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
