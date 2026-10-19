#!/usr/bin/env python
"""Clean data/raw/processed.cleveland.data -> data/processed/heart_clean.csv."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartrisk import config, data  # noqa: E402


def main() -> int:
    if not config.RAW_PATH.exists():
        print(f"ERROR: {config.RAW_PATH} not found. Run scripts/download_data.py first.", file=sys.stderr)
        return 2
    raw = data.load_raw()
    clean = data.clean(raw)
    out = data.save_clean(clean)
    print(f"raw rows: {len(raw)}  ->  clean rows: {len(clean)}")
    print("missing values kept for in-pipeline imputation:",
          {k: int(v) for k, v in clean.isna().sum().items() if v})
    print("class balance:", clean[config.TARGET].value_counts().sort_index().to_dict())
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
