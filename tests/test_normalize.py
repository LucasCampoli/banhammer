import numpy as np

from banhammer.ocr.normalize import CARD_SIZE, normalize


def test_normalize_resizes():
    assert normalize(np.zeros((500, 300, 3), np.uint8)).shape == (CARD_SIZE[1], CARD_SIZE[0], 3)
