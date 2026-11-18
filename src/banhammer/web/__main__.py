"""Local web UI: python -m banhammer.web, then open http://127.0.0.1:8000"""
from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from banhammer.ingest.scryfall_cards import open_text
from banhammer.models import FORMATS, CardInput
from banhammer.pipeline import build_graph, evaluate_card
from banhammer.shared.config import ROOT
from banhammer.shared.report import cited_sources

log = logging.getLogger("banhammer.web")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
# Scryfall objects that are not playable cards.
NON_CARD_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem", "vanguard", "planar", "scheme"}
SEARCH_LIMIT = 8
MAX_BODY_BYTES = 64_000


def image_url(raw: dict) -> str | None:
    """Scryfall image of the card, or of its front face for double-faced cards."""
    faces = raw.get("card_faces") or [{}]
    uris = raw.get("image_uris") or faces[0].get("image_uris") or {}
    return uris.get("normal")


def load_catalog(bulk_path: Path) -> tuple[list[tuple[str, dict]], dict[str, str]]:
    """Every playable card as (casefolded name, CardInput fields + image) for the name search,
    and the image of every card by oracle id, for the cards a report cites."""
    catalog = []
    images = {}
    with open_text(bulk_path) as handle:
        for line in handle:
            raw = json.loads(line)
            if raw.get("layout") in NON_CARD_LAYOUTS:
                continue
            card = CardInput.model_validate(raw)
            image = image_url(raw)
            if card.oracle_id and image:
                images[card.oracle_id] = image
            catalog.append((card.name.casefold(), {**card.model_dump(mode="json"), "image": image}))
    catalog.sort(key=lambda entry: entry[0])
    return catalog, images


def search_catalog(catalog: list[tuple[str, dict]], query: str) -> list[dict]:
    """Names that start with the query first, then names that merely contain it."""
    needle = query.strip().casefold()
    if not needle:
        return []
    prefix = [card for name, card in catalog if name.startswith(needle)]
    contains = [card for name, card in catalog if needle in name and not name.startswith(needle)]
    return (prefix + contains)[:SEARCH_LIMIT]


def make_handler(graph, catalog: list[tuple[str, dict]], images: dict[str, str]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/api/cards":
                query = parse_qs(url.query).get("q", [""])[0]
                self._send_json({"cards": search_catalog(catalog, query)})
            elif url.path in STATIC_FILES:
                name, content_type = STATIC_FILES[url.path]
                self._send(HTTPStatus.OK, (STATIC_DIR / name).read_bytes(), content_type)
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/evaluate":
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send_json({"error": "request too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                fmt = body.get("format", "modern")
                if fmt not in FORMATS:
                    raise ValueError(f"format must be one of {', '.join(FORMATS)}")
                card = CardInput.model_validate(body.get("card") or {})
                if not card.name.strip():
                    raise ValueError("the card needs a name")
            except (ValueError, ValidationError) as error:  # JSONDecodeError is a ValueError
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            try:
                report, result = evaluate_card(graph, card, fmt)
            except Exception as error:  # the pipeline calls remote APIs; report any failure to the page
                log.exception("evaluation of %r failed", card.name)
                self._send_json({"error": f"the pipeline failed: {error}"}, HTTPStatus.BAD_GATEWAY)
                return
            sources = cited_sources(report, result.get("hits") or [])
            for source_id, source in sources.items():
                source["image"] = images.get(source_id)  # card hits are keyed by oracle id
            self._send_json({
                "card": card.model_dump(mode="json"),
                "image": images.get(card.oracle_id or ""),
                "format": fmt,
                "report": report.model_dump(mode="json"),
                "sources": sources,
                "dropped_claims": result.get("dropped_claims", 0),
            })

        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhammer.web", description="Serve the local web UI.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--bulk", type=Path, default=ROOT / "data" / "raw" / "oracle-cards.jsonl.gz",
                        help="Scryfall oracle bulk file used by the card search")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(f"Loading cards from {args.bulk} ...")
    catalog, images = load_catalog(args.bulk)
    graph = build_graph(args.config)
    # Bound to localhost only: the page triggers paid API calls.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(graph, catalog, images))
    print(f"{len(catalog)} cards loaded. Open http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
