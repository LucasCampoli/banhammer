"""Download sample cards from Scryfall into images/samples/ (gitignored)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

API = "https://api.scryfall.com/cards/search"
BASE_QUERY = "legal:standard frame:2015 -is:fullart -is:borderless -is:showcase"
TYPES = ["creature", "instant", "sorcery", "enchantment", "artifact"]
DELAY = 0.1  # Scryfall asks for 50-100 ms between requests
FIELDS = ["id", "name", "mana_cost", "type_line", "oracle_text", "power", "toughness"]

HEADERS = {
    "User-Agent": f"Banhammer/0.1 (university project; {os.environ.get('BANHAMMER_CONTACT', 'contact-not-set')})",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}


def get(session: requests.Session, url: str, **kw) -> requests.Response:
    time.sleep(DELAY)
    r = session.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r


def fetch_type(session: requests.Session, card_type: str, n: int) -> list[dict]:
    params = {"q": f"{BASE_QUERY} t:{card_type}", "unique": "cards", "order": "random"}
    out: list[dict] = []
    url, p = API, params
    while url and len(out) < n:
        data = get(session, url, params=p).json()
        for c in data.get("data", []):
            if "card_faces" in c or "image_uris" not in c:
                continue  # skip double-faced cards for now
            out.append(c)
            if len(out) == n:
                break
        url, p = data.get("next_page"), None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("images/samples"))
    ap.add_argument("--per-type", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    truth, seen = [], set()
    with requests.Session() as s:
        for t in TYPES:
            for c in fetch_type(s, t, args.per_type):
                if c["id"] in seen:
                    continue
                seen.add(c["id"])
                img = get(s, c["image_uris"]["large"]).content
                (args.out / f"{c['id']}.jpg").write_bytes(img)
                truth.append({k: c.get(k) for k in FIELDS})
                print(f"[{t}] {c['name']}")
    (args.out / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    print(f"Saved {len(truth)} cards to {args.out}")


if __name__ == "__main__":
    main()
