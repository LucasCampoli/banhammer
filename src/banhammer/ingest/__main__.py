"""Download and normalize the card catalog and the Comprehensive Rules."""
from __future__ import annotations

import argparse
from pathlib import Path

from banhammer.shared.config import ROOT, load_env, load_yaml
from banhammer.shared.documents import write_jsonl
from banhammer.ingest import SOURCES


def run(config_path: Path, force: bool = False, only: str | None = None) -> None:
    load_env()
    cfg = load_yaml(config_path)
    raw_dir = ROOT / cfg.get("raw_dir", "data/raw")
    processed_dir = ROOT / cfg.get("processed_dir", "data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = [only] if only else list(cfg["sources"])
    for name in selected:
        source = SOURCES[name]
        raw_path = raw_dir / source.filename
        out_path = processed_dir / f"{source.namespace}.jsonl"
        if raw_path.exists() and not force:
            print(f"{name}: raw file exists, skipping download ({raw_path})")
        else:
            print(f"{name}: downloading")
            source.download(raw_path)
        if out_path.exists() and not force:
            print(f"{name}: processed file exists, skipping normalize ({out_path})")
            continue
        docs = source.normalize(raw_path)
        write_jsonl(out_path, docs)
        print(f"{name}: wrote {len(docs)} records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhammer.ingest", description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ingest.yaml")
    parser.add_argument("--force", action="store_true", help="download and normalize even if the files exist")
    parser.add_argument("--only", choices=sorted(SOURCES))
    args = parser.parse_args()
    run(args.config, force=args.force, only=args.only)


if __name__ == "__main__":
    main()
