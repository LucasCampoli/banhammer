import numpy as np

from banhammer.ocr.zones import COLORS, crop_zones, draw_zone_overlay, load_zones

EXPECTED = {"name", "mana_cost", "type_line", "rules_text", "power_toughness"}


def test_zone_config_loads():
    zones = load_zones()
    assert set(zones) == EXPECTED == set(COLORS)


def test_crop_and_overlay_shapes():
    img = np.zeros((936, 672, 3), np.uint8)
    zones = load_zones()
    crops = crop_zones(img, zones)
    assert all(c.size > 0 for c in crops.values())
    assert draw_zone_overlay(img, zones).shape == img.shape
