"""Download mana symbols from Scryfall and save them as PNG templates."""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import cairosvg
import requests

HEADERS = {
    "User-Agent": f"Banhammer/0.1 (university project; {os.environ.get('BANHAMMER_CONTACT', 'contact-not-set')})",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}
# symbols that can show up in a mana cost (no tap, energy, etc.)
COST = re.compile(r"^\{(?:[WUBRGCSXYZ]|\d{1,2}|[WUBRG]/[WUBRG](?:/P)?|2/[WUBRG]|[WUBRGC]/P)\}$")


def filename(symbol: str) -> str:
    return symbol.strip("{}").replace("/", "_")


SIZE = 64


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/mana_templates"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    symbols = requests.get("https://api.scryfall.com/symbology", headers=HEADERS, timeout=30).json()["data"]
    for s in symbols:
        if not COST.match(s["symbol"]):
            continue
        time.sleep(0.1)
        svg = requests.get(s["svg_uri"], headers=HEADERS, timeout=30).content
        cairosvg.svg2png(bytestring=svg, write_to=str(args.out / f"{filename(s['symbol'])}.png"),
                         output_width=SIZE, output_height=SIZE)
        print("saved", s["symbol"])


if __name__ == "__main__":
    main()
