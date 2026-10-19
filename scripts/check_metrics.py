#!/usr/bin/env python
"""CI quality gate: fail the pipeline if the trained model is not good enough."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ROOT / "reports" / "metrics.json"))
    parser.add_argument("--min-auc", type=float, default=0.80)
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        print(f"FAIL: {path} not found - did training run?", file=sys.stderr)
        return 2
    m = json.loads(path.read_text())
    cv_auc, test_auc = m["cv_roc_auc"], m["test"]["roc_auc"]
    print(f"best={m['best_model']}  nested-CV AUC={cv_auc:.3f}  test AUC={test_auc:.3f}  "
          f"(gate: >= {args.min_auc})")
    if cv_auc < args.min_auc or test_auc < args.min_auc:
        print("FAIL: model below quality gate", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
