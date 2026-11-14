"""Mana and tap symbols inside the rules text.

Tesseract reads a symbol as junk ("@", "®", "e"), so symbols are found by
template matching and replaced with a plain "X" before Tesseract reads the
text box. Tesseract reads "X: Add X." well, while it drops or garbles a colon
left alone at the start of a line. Each placeholder read inside a word that
covers a symbol is then replaced with the symbol ("{T}: Add {C}.").

Mana symbols are printed on a colored disk. Recent text boxes print tap,
untap, energy and colorless mana as a bare black glyph instead, so those are
also searched with templates made from the glyph alone.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from .mana import Symbol, find_symbols, non_overlapping
from .tesseract import Word

# Within a disk ring vs just outside it; text digits and letters sit on plain paper.
INNER_RING = (0.72, 0.92)
OUTER_RING = (1.15, 1.45)
NO_SPACE_AFTER = "(“\""
NO_SPACE_BEFORE = ":,.;)”\""
# Everything inside the symbol is erased (radii). Around it, the blended edge of the disk and
# its drop shadow are erased too, but not dark ink: a colon or comma often follows a cost.
ERASE_CORE = 1.0
ERASE_HALO = 1.4
INK_DARKNESS = 0.6  # pixels darker than this fraction of the paper's gray are ink
SAME_LINE = 0.6  # a symbol belongs to a line or word whose center is within this many symbol diameters
PLACEHOLDER = "X"
PLACEHOLDER_READINGS = "Xx×"
PLACEHOLDER_SCALE = 0.026  # Hershey font scale per pixel of symbol diameter: about cap height
PLACEHOLDER_THICKNESS = 2


def _ring_mask(shape: tuple[int, int], symbol: Symbol, ring: tuple[float, float]) -> np.ndarray:
    radius = symbol.diameter / 2
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    distance = np.hypot(xx - symbol.x, yy - symbol.y) / radius
    return (distance > ring[0]) & (distance < ring[1])


def paper_color(image_bgr: np.ndarray, symbol: Symbol) -> np.ndarray:
    return np.median(image_bgr[_ring_mask(image_bgr.shape[:2], symbol, OUTER_RING)], axis=0)


def disk_contrast(image_bgr: np.ndarray, symbol: Symbol) -> float:
    """Color distance between the edge of the symbol's disk and the paper around it."""
    inner = image_bgr[_ring_mask(image_bgr.shape[:2], symbol, INNER_RING)].astype(np.float32)
    outer = image_bgr[_ring_mask(image_bgr.shape[:2], symbol, OUTER_RING)].astype(np.float32)
    if not len(inner) or not len(outer):
        return 0.0
    return float(np.linalg.norm(np.median(inner, axis=0) - np.median(outer, axis=0)))


# The text box is large, so only strong grayscale matches go on to the colour check.
COARSE_SCORE = 0.7
BARE_SYMBOLS = ("T", "Q", "E", "C")
GLYPH_MAX_GRAY = 90  # template pixels this dark are the glyph; the rest is the disk


def bare_templates(templates: dict) -> dict:
    """Templates of the symbols that can be printed without a disk: black glyph on white."""
    out = {}
    for name in BARE_SYMBOLS:
        if name not in templates:
            continue
        bgr, alpha = templates[name]
        glyph = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < GLYPH_MAX_GRAY) & (alpha > 0)
        image = np.full_like(bgr, 255)
        image[glyph] = 0
        out[name] = (image, alpha)
    return out


def find_text_symbols(image_bgr: np.ndarray, disk_templates: dict, glyph_templates: dict, sizes: list[int],
                      min_score: float, min_glyph_score: float, min_contrast: float) -> list[Symbol]:
    """Symbols in the text box.

    Digits and letters of the text also match the number templates, so a disk
    symbol must stand out from the paper around it. Bare glyphs have no disk;
    their shapes do not look like letters, so a high score is enough.
    """
    disks = [s for s in find_symbols(image_bgr, disk_templates, sizes, min_score, COARSE_SCORE)
             if disk_contrast(image_bgr, s) >= min_contrast]
    glyphs = find_symbols(image_bgr, glyph_templates, sizes, min_glyph_score, COARSE_SCORE)
    return non_overlapping(disks + glyphs)


def replace_with_placeholders(image_bgr: np.ndarray, symbols: list[Symbol]) -> np.ndarray:
    """Erase each symbol and draw the placeholder letter in its place."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = image_bgr.copy()
    for symbol in symbols:
        paper = paper_color(image_bgr, symbol)
        paper_gray = float(cv2.cvtColor(paper.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)[0, 0])
        ink = gray < INK_DARKNESS * paper_gray
        core = _ring_mask(out.shape[:2], symbol, (0.0, ERASE_CORE))
        halo = _ring_mask(out.shape[:2], symbol, (0.0, ERASE_HALO)) & ~ink
        out[core | halo] = paper
        scale = symbol.diameter * PLACEHOLDER_SCALE
        (width, height), _ = cv2.getTextSize(PLACEHOLDER, cv2.FONT_HERSHEY_SIMPLEX, scale, PLACEHOLDER_THICKNESS)
        cv2.putText(out, PLACEHOLDER, (int(symbol.x - width / 2), int(symbol.y + height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), PLACEHOLDER_THICKNESS, cv2.LINE_AA)
    return out


def restore_symbols(lines: list[list[Word]], symbols: list[Symbol], scale: float) -> list[list[Word]]:
    """Replace placeholders read inside words covering a symbol; insert the symbols Tesseract missed.

    Symbol positions are in crop pixels and word boxes in read-image pixels, `scale` times larger.
    A placeholder only counts inside a word over a symbol: rules text also has real X's.
    """
    remaining = sorted(symbols, key=lambda s: s.x)
    out = []
    for line in lines:
        restored = []
        for word in line:
            covered = [s for s in remaining
                       if word.left <= s.x * scale <= word.left + word.width
                       and abs(s.y * scale - (word.top + word.height / 2)) <= SAME_LINE * s.diameter * scale]
            if not covered:
                restored.append(word)
                continue
            # Tesseract sometimes reads one placeholder as "Xx": extra ones in the word are dropped too.
            text, scores = "", []
            for char in word.text:
                if char not in PLACEHOLDER_READINGS:
                    text += char
                elif covered:
                    symbol = covered.pop(0)
                    remaining.remove(symbol)
                    text += f"{{{symbol.name}}}"
                    scores.append(symbol.score)
            # Tesseract is unsure of the drawn placeholders; the template match vouches for the word.
            confidence = max([word.confidence or 0.0, *scores])
            restored.append(Word(text, confidence, word.left, word.top, word.width, word.height))
        out.append(restored)
    return insert_symbols(out, remaining, scale)


def _center(line: list[Word]) -> float:
    return sum(w.top + w.height / 2 for w in line) / len(line)


def insert_symbols(lines: list[list[Word]], symbols: list[Symbol], scale: float) -> list[list[Word]]:
    """Add each symbol as a word to the line it sits on, or to a new line if it is far from all of them."""
    out = [list(line) for line in lines]
    for symbol in sorted(symbols, key=lambda s: s.y):
        size = int(symbol.diameter * scale)
        y = symbol.y * scale
        word = Word(f"{{{symbol.name}}}", symbol.score, int(symbol.left * scale), int(y - size / 2), size, size)
        nearest = min(out, key=lambda line: abs(_center(line) - y), default=None)
        if nearest is not None and abs(_center(nearest) - y) <= SAME_LINE * size:
            nearest.append(word)
        else:
            out.append([word])
    return [sorted(line, key=lambda w: w.left) for line in sorted(out, key=_center)]


def join_words(texts: list[str]) -> str:
    """Join a line's words, writing symbols next to each other and to adjacent punctuation."""
    out = ""
    for text in texts:
        previous_symbol = out.endswith("}")
        glued = out and (
            (previous_symbol and text.startswith("{"))
            or (previous_symbol and text[0] in NO_SPACE_BEFORE)
            or (text.startswith("{") and out[-1] in NO_SPACE_AFTER)
        )
        out += text if glued or not out else " " + text
    return re.sub(r"(?<=\}):(?=\S)", ": ", out)  # "{T}:Add" when Tesseract joined the colon to the next word
