#!/usr/bin/env python3
"""CLI entry point — reproduce_command for submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python rank.py` from project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.ranker import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob AI hackathon job description"
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .json")
    parser.add_argument("--out", required=True, help="Output submission CSV path")
    parser.add_argument(
        "--config",
        default="config/jd_requirements.yaml",
        help="Path to JD requirements YAML (default: config/jd_requirements.yaml)",
    )
    parser.add_argument("--top-n", type=int, default=100, help="Number of rows in submission")
    args = parser.parse_args()

    result = run_pipeline(
        candidates_path=args.candidates,
        out_path=args.out,
        config_path=args.config,
        top_n=args.top_n,
    )
    print(
        f"Wrote {len(result.rows)} rows to {args.out} "
        f"({result.candidates_processed} processed, "
        f"{result.candidates_excluded} excluded, "
        f"{result.elapsed_seconds:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
