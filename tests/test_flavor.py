import cv2
import numpy as np

from banhammer.ocr.flavor import ITALIC_SLANT, line_styles, slant, split_flavor

CARD_ITALIC_SHEAR = 0.25  # slant measured on italic words of the benchmark cards
U, I = False, True  # upright, italic


def _line(italic: bool) -> np.ndarray:
    img = np.full((60, 400), 255, np.uint8)
    cv2.putText(img, "Hold the line", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3, cv2.LINE_AA)
    if not italic:
        return img
    shear = np.float32([[1, -CARD_ITALIC_SHEAR, CARD_ITALIC_SHEAR * 30], [0, 1, 0]])
    return cv2.warpAffine(img, shear, (400, 60), borderValue=255)


def test_slant_tells_italic_from_upright():
    assert slant(_line(italic=False)) < ITALIC_SLANT < slant(_line(italic=True))


def test_reminder_words_do_not_vote():
    lines = [["Lifelink", "(Damage", "dealt", "by"], ["also", "causes", "life.)"], ['"The', "blood", "is"]]
    italic = [[U, I, I, None], [I, I, I], [I, I, None]]
    assert line_styles(lines, italic) == [False, None, True]


def test_split_flavor_takes_the_trailing_italic_paragraph():
    assert split_flavor([False, None, True, True]) == 2
    assert split_flavor([False, True, False]) == 3
    assert split_flavor([False, None]) == 2


def test_split_flavor_on_vanilla_and_empty_text_boxes():
    assert split_flavor([True, True]) == 0
    assert split_flavor([]) == 0
