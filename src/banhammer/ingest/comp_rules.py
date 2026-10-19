"""Comprehensive Rules: one document per numbered rule and subrule."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from langchain_core.documents import Document

from banhammer.shared.documents import clean_metadata
from banhammer.ingest.http import download, headers

RULES_PAGE = "https://magic.wizards.com/en/rules"
_TXT_LINK = re.compile(r'href="([^"]+\.txt[^"]*)"', re.IGNORECASE)
_RULE = re.compile(r"^(\d{3}\.\d+[a-z]?)(?:\.)?\s+(.*)$")
_SECTION = re.compile(r"^(\d{3})\.\s+(\S.*)$")


class CompRules:
    namespace = "rules"
    filename = "comp-rules.txt"

    def download(self, dest: Path) -> None:
        response = requests.get(RULES_PAGE, headers=headers("text/html"), timeout=60)
        response.raise_for_status()
        download(rules_txt_url(response.text), dest, accept="text/plain")

    def normalize(self, raw: Path) -> list[Document]:
        return rules_to_documents(raw.read_text(encoding="utf-8-sig"))


def rules_txt_url(html: str, base: str = RULES_PAGE) -> str:
    links = _TXT_LINK.findall(html)
    if not links:
        raise RuntimeError(f"no .txt link found on {base}")
    for link in links:
        lowered = link.lower()
        if "comprule" in lowered or "magiccomp" in lowered:
            return urljoin(base, link)
    return urljoin(base, links[0])


def rules_to_documents(text: str) -> list[Document]:
    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    start = _nth(lines, "1. Game Concepts", 2)
    end = _nth(lines, "Glossary", 2)
    if start is None:
        start = _nth(lines, "1. Game Concepts", 1) or 0
    body = lines[start:end] if end is not None else lines[start:]

    section = ""
    titles: dict[str, str] = {}
    current: dict | None = None
    docs: list[Document] = []

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        rule = current["rule"]
        body_text = current["text"]
        if current["examples"]:
            body_text = body_text + "\n" + "\n".join(current["examples"])
        if not rule[-1].isalpha():
            titles[rule] = current["text"]
        parent = rule[:-1] if rule[-1].isalpha() else rule
        page = f"{rule} {body_text}".strip()
        docs.append(Document(
            id=rule,
            page_content=page,
            metadata=clean_metadata({
                "rule": rule,
                "parent": parent,
                "title": titles.get(parent, ""),
                "section": section,
                "text": page,
            }),
        ))
        current = None

    for raw_line in body:
        line = raw_line.strip()
        if not line:
            continue
        rule_match = _RULE.match(line)
        if rule_match:
            flush()
            current = {"rule": rule_match.group(1), "text": rule_match.group(2).strip(), "examples": []}
            continue
        section_match = _SECTION.match(line)
        if section_match:
            flush()
            section = f"{section_match.group(1)}. {section_match.group(2).strip()}"
            continue
        if current is None:
            continue
        if line.startswith("Example:"):
            current["examples"].append(line)
        else:
            current["text"] = f"{current['text']} {line}".strip()
    flush()
    return docs


def _nth(lines: list[str], prefix: str, occurrence: int) -> int | None:
    seen = 0
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            seen += 1
            if seen == occurrence:
                return index
    return None
