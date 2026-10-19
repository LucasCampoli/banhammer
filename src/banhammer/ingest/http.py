"""HTTP helpers for the data sources."""
from __future__ import annotations

import os
from pathlib import Path

import requests

from banhammer import __version__


def headers(accept: str = "application/json;q=0.9,*/*;q=0.8") -> dict[str, str]:
    contact = os.environ.get("BANHAMMER_CONTACT", "contact-not-set")
    return {
        "User-Agent": f"Banhammer/{__version__} (personal project; {contact})",
        "Accept": accept,
    }


def download(url: str, dest: Path, *, accept: str = "*/*") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=headers(accept), stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    temporary.replace(dest)
