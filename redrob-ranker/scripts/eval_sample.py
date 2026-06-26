#!/usr/bin/env python3
"""Evaluate ranker on a small sample — prints top-K for manual review (not spec CSV)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader.jd_config import load_jd_config
from src.io.candidates import load_candidates
from src.pipeline.ranker import rank_candidates
from src.scoring.combiner import score_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample pool evaluation (manual review)")
    parser.add_argument(
        "--candidates",
        default=str(ROOT.parent / "sample_candidates.json"),
        help="Path to sample JSON/JSONL",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_jd_config(args.config)
    candidates = load_candidates(args.candidates)
    top_k = min(args.top_k, len(candidates))

    ranked, excluded = rank_candidates(candidates, config, top_n=top_k)

    print(f"Pool size: {len(candidates)} | Excluded: {excluded} | Showing top {top_k}\n")
    print(f"{'Rank':<5} {'ID':<14} {'FINAL':<7} {'FIT':<7} {'AVAIL':<7} {'INT':<7} Title")
    print("-" * 80)
    for i, (cand, sc) in enumerate(ranked, start=1):
        title = cand.get("profile", {}).get("current_title", "")[:28]
        print(
            f"{i:<5} {sc.candidate_id:<14} {sc.final_score:<7.3f} "
            f"{sc.fit:<7.3f} {sc.availability:<7.3f} {sc.integrity:<7.3f} {title}"
        )
        if sc.integrity_issues:
            print(f"      issues: {', '.join(sc.integrity_issues)}")

    negatives = []
    for cand in candidates:
        sc = score_candidate(cand, config)
        title = (cand.get("profile", {}).get("current_title") or "").lower()
        if any(k in title for k in ("marketing", "hr manager", "content writer", "graphic")):
            negatives.append((sc.final_score, sc.candidate_id, title))
    if negatives:
        print("\nNegative-title candidates (should rank below top):")
        for fs, cid, title in sorted(negatives, reverse=True)[:5]:
            print(f"  {cid} {fs:.3f} — {title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
