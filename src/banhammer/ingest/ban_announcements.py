"""Why each banned card was banned, from Wizards' banned and restricted announcements.

The index only says a card is banned. The announcements say why (win rates, turn-three kills,
metagame share) and when. This module downloads every announcement listed in the site map
(cached, one request per second), finds the paragraphs that name each card banned in Modern or
Standard, and asks the LLM for a one- or two-sentence reason. The result goes to
data/processed/ban_reasons.jsonl, keyed by oracle id, for the judge to read next to the
"(banned in ...)" mark (config `ban_reasons: true`).

    python -m banhammer.ingest.ban_announcements [--config configs/no_rag.yaml]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from banhammer.providers import get_llm
from banhammer.shared.config import ROOT, load_env, load_yaml
from banhammer.shared.documents import read_jsonl

SITEMAP = "https://magic.wizards.com/en/sitemap.xml"
USER_AGENT = "Mozilla/5.0 (compatible; research-script)"
CACHE = ROOT / "data" / "raw" / "ban-announcements"
OUT = ROOT / "data" / "processed" / "ban_reasons.jsonl"
FORMATS = ("modern", "standard")
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                     "august", "september", "october", "november", "december"], 1)}


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def announcement_urls(sitemap: str) -> list[str]:
    """Banned and restricted announcements, without the Commander and Arena-only ones."""
    urls = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    keep = []
    for url in urls:
        slug = url.rsplit("/", 1)[-1]
        if "banned" in slug and ("restricted" in slug or "pioneer-banned" in slug) \
                and not any(skip in slug for skip in ("commander", "arena", "history", "all-banned")):
            keep.append(url)
    return sorted(set(keep))


def announcement_date(url: str) -> str | None:
    """The date in an announcement's address, as YYYY-MM-DD."""
    slug = url.rsplit("/", 1)[-1]
    found = re.search(r"(\d{4})-(\d{2})-(\d{2})", slug)
    if found:
        return "-".join(found.groups())
    found = re.search(r"(" + "|".join(MONTHS) + r")-(\d{1,2})-(\d{4})", slug)
    if found:
        month, day, year = found.groups()
        return f"{year}-{MONTHS[month]:02d}-{int(day):02d}"
    return None


def paragraphs(page: str) -> list[str]:
    """The article's paragraphs as plain text; inline tags (card links) are joined, not split."""
    page = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", page, flags=re.S | re.I)
    page = re.sub(r"</?(p|div|h[1-6]|li|br|tr|section|article)[^>]*>", "\n", page, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", page))
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if len(line.strip()) > 60]


def load_pages(urls: list[str]) -> dict[str, list[str]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    pages = {}
    for url in urls:
        path = CACHE / (url.rsplit("/", 1)[-1] + ".html")
        if not path.exists():
            path.write_text(fetch(url), encoding="utf-8")
            time.sleep(1)
        pages[url] = paragraphs(path.read_text(encoding="utf-8"))
    return pages


class BanReason(BaseModel):
    found: bool = Field(description="Whether the excerpts explain why this card was banned.")
    reason: str = Field("", description="Why it was banned, in one or two sentences of at most 45 words, "
                                        "concrete (win rate, how fast the deck won, metagame share, the "
                                        "interaction that made it a problem). Empty if not found.")


_SYSTEM = """You read excerpts of Magic: The Gathering banned and restricted announcements.
Say why the given card was banned in the given format, using only what the excerpts say."""


def banned_cards(cards_path: Path) -> list[dict]:
    rows = []
    for doc in read_jsonl(cards_path):
        meta = doc.metadata
        for fmt in meta.get("banned") or []:
            if fmt in FORMATS:
                rows.append({"oracle_id": meta["oracle_id"], "name": meta["name"], "format": fmt})
    return rows


def mentions(name: str, pages: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """Announcements that name the card, oldest first, with the paragraphs that do."""
    names = {name, name.split(",")[0]} | {part.strip() for part in name.split("//")}
    found = []
    for url, paras in pages.items():
        hits = [p for p in paras if any(n in p for n in names)]
        if hits:
            found.append((url, hits))
    return sorted(found, key=lambda item: announcement_date(item[0]) or "9999")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhammer.ingest.ban_announcements", description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "no_rag.yaml")
    args = parser.parse_args()
    load_env()
    llm = get_llm(load_yaml(args.config)["llm"]).with_structured_output(BanReason)
    urls = announcement_urls(fetch(SITEMAP))
    print(f"{len(urls)} announcements")
    pages = load_pages(urls)
    rows = []
    for card in banned_cards(ROOT / "data" / "processed" / "cards.jsonl"):
        found = mentions(card["name"], pages)
        reason, url = None, None
        if found:
            excerpt = "\n\n".join(f"[{announcement_date(u)}] " + "\n".join(paras[:6]) for u, paras in found[:3])
            answer = llm.invoke([SystemMessage(content=_SYSTEM),
                                 HumanMessage(content=f"Card: {card['name']}\nFormat: {card['format']}\n\n{excerpt}")])
            answer = answer if isinstance(answer, BanReason) else BanReason.model_validate(answer)
            if answer.found and answer.reason:
                reason, url = answer.reason, found[0][0]
        rows.append({**card, "reason": reason, "date": announcement_date(url) if url else None, "source": url})
        print(f"{card['format']:<9} {card['name']:<35} {'ok' if reason else '--'} {(reason or '')[:70]}")
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"{sum(1 for r in rows if r['reason'])}/{len(rows)} with a reason. Wrote {OUT}")


if __name__ == "__main__":
    main()
