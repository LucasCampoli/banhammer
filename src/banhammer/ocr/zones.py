"""Zone configuration, cropping and debug overlay."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from banhammer.models import ZoneBox

DEFAULT_ZONES = Path(__file__).resolve().parents[3] / "configs" / "zones.yaml"
COLORS = {  # BGR
    "name": (0, 0, 255), "mana_cost": (0, 165, 255), "type_line": (0, 200, 0),
    "rules_text": (255, 0, 0), "power_toughness": (255, 0, 255),
}


def load_zones(path: Path | str = DEFAULT_ZONES) -> dict[str, ZoneBox]:
    data = yaml.safe_load(Path(path).read_text())
    return {k: ZoneBox(x0=v[0], y0=v[1], x1=v[2], y1=v[3]) for k, v in data["zones"].items()}


def crop_zones(img: np.ndarray, zones: dict[str, ZoneBox]) -> dict[str, np.ndarray]:
    h, w = img.shape[:2]
    out = {}
    for name, z in zones.items():
        x0, y0, x1, y1 = z.to_pixels(w, h)
        out[name] = img[y0:y1, x0:x1].copy()
    return out


def draw_zone_overlay(img: np.ndarray, zones: dict[str, ZoneBox]) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    for name, z in zones.items():
        x0, y0, x1, y1 = z.to_pixels(w, h)
        color = COLORS.get(name, (0, 255, 255))
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis, name, (x0 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis
