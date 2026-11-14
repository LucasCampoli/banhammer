"""Split flavor text from rules text using the slant of each printed word.

Flavor text is the last paragraph of the text box and is printed in italics.
Reminder text is italic too, but it is enclosed in parentheses, so words inside
parentheses do not count when deciding whether a line is italic.
"""
from __future__ import annotations

import cv2
import numpy as np

# Horizontal shear per pixel of height that is tried when straightening a word.
SHEARS = np.linspace(-0.15, 0.45, 25)
# Upright words straighten at ~0.0 and italic words at ~0.25 on the benchmark.
ITALIC_SLANT = 0.12
# Shorter words and punctuation give unreliable slants.
MIN_MEASURED_CHARS = 3


def slant(gray: np.ndarray) -> float:
    """Shear that makes the strokes of a word or line most vertical (0 upright, ~0.25 italic).

    Each candidate shear is undone and the column sums of the ink are taken: the
    sharper the vertical strokes, the more peaked the projection, so the shear
    with the largest sum of squares wins.
    """
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1].astype(np.float32)
    height, width = ink.shape
    scores = []
    for shear in SHEARS:
        matrix = np.float32([[1, shear, -shear * height / 2], [0, 1, 0]])
        projection = cv2.warpAffine(ink, matrix, (width, height)).sum(axis=0)
        scores.append(float((projection ** 2).sum()))
    return float(SHEARS[int(np.argmax(scores))])


def is_measurable(word: str) -> bool:
    return sum(ch.isalnum() for ch in word) >= MIN_MEASURED_CHARS


def line_styles(lines: list[list[str]], italic: list[list[bool | None]]) -> list[bool | None]:
    """Whether each line is italic (True), upright (False) or only reminder text (None).

    `italic` holds one flag per word, None for words too short to measure.
    Words inside parentheses are skipped and the remaining words vote.
    """
    depth = 0
    styles: list[bool | None] = []
    for words, flags in zip(lines, italic):
        votes = []
        for word, flag in zip(words, flags):
            if depth == 0 and not word.startswith("(") and flag is not None:
                votes.append(flag)
            depth = max(0, depth + word.count("(") - word.count(")"))
        styles.append(sum(votes) * 2 > len(votes) if votes else None)
    return styles


def split_flavor(styles: list[bool | None]) -> int:
    """Index of the first flavor line (len(styles) when there is no flavor).

    Flavor is the run of lines at the end with no upright line, starting at its
    first italic line; reminder-only lines before that stay in the rules text.
    """
    start = len(styles)
    while start > 0 and styles[start - 1] is not False:
        start -= 1
    while start < len(styles) and styles[start] is not True:
        start += 1
    return start
