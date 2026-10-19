#!/usr/bin/env python
"""Download the UCI Cleveland heart disease file into data/raw/.

    python scripts/download_data.py                # download (overwrites)
    python scripts/download_data.py --skip-if-exists

If the UCI server is unreachable, download the file manually from
https://archive.ics.uci.edu/dataset/45/heart+disease (processed.cleveland.data)
and place it at data/raw/processed.cleveland.data.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartrisk import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--url", default=config.UCI_URL)
    args = parser.parse_args()

    dest = config.RAW_PATH
    if args.skip_if_exists and dest.exists():
        print(f"{dest} already present - skipping download")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(args.url, timeout=60) as resp:
            body = resp.read()
    except Exception as exc:  # network errors, DNS, HTTP errors
        print(f"ERROR: could not download {args.url}: {exc}", file=sys.stderr)
        print("Place processed.cleveland.data in data/raw/ manually (see docstring).", file=sys.stderr)
        return 1

    dest.write_bytes(body)
    n_lines = body.count(b"\n")
    print(f"Saved {dest} ({len(body)} bytes, {n_lines} rows)")
    if n_lines < 300:
        print("WARNING: expected about 303 rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
