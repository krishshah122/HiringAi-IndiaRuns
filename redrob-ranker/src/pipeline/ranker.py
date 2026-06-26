"""End-to-end ranking pipeline."""

from __future__ import annotations

import csv
import heapq
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config_loader.jd_config import JDConfig, load_jd_config
from src.io.candidates import iter_candidates, load_candidates
from src.reasoning.generator import generate_reasoning
from src.scoring.combiner import CandidateScore, score_candidate
from src.semantic.loader import SemanticLoader


@dataclass
class RankingResult:
    rows: list[dict[str, Any]]
    elapsed_seconds: float
    candidates_processed: int
    candidates_excluded: int


def rank_candidates(
    candidates: list[dict[str, Any]],
    config: JDConfig,
    top_n: int = 100,
) -> tuple[list[tuple[dict[str, Any], CandidateScore]], int]:
    """Return top-N (candidate, score) pairs and exclusion count using Two-Stage Re-ranking."""
    from src.text.profile_text import build_jd_text, build_rich_profile_text

    scored_entries: list[list[Any]] = []
    excluded = 0

    for candidate in candidates:
        scored = score_candidate(candidate, config)
        if scored.excluded:
            excluded += 1
            continue
        scored_entries.append(
            [scored.final_score, scored.candidate_id, candidate, scored]
        )

    if scored_entries:
        # Step 1: Get the top K candidates for re-ranking (K = 500)
        k = min(500, len(scored_entries))
        scored_entries.sort(key=lambda x: (-x[0], x[1]))
        top_k = scored_entries[:k]
        
        # Step 2: Build rich profile texts and JD text for these top K candidates
        rich_texts = [build_rich_profile_text(item[2]) for item in top_k]
        jd_text = build_jd_text(config)
        
        # Step 3: Compute high-fidelity semantic similarities on-the-fly
        loader = SemanticLoader.get_instance()
        rich_sims = loader.compute_rich_similarities(rich_texts, jd_text)
        
        # Step 4: Re-calculate the final score for each of the top K candidates
        weights = config.weights
        total_w = sum(weights.get(key, 0.0) for key in ("title_career", "skills", "experience", "semantic", "location"))
        if total_w <= 0:
            total_w = 1.0
            
        for i, item in enumerate(top_k):
            candidate = item[2]
            scored = item[3]
            
            # Update the semantic feature with the rich semantic similarity
            scored.features["semantic"] = rich_sims[i]
            
            # Recompute fit score
            fit = 0.0
            for key in ("title_career", "skills", "experience", "semantic", "location"):
                w = weights.get(key, 0.0) / total_w
                fit += w * scored.features.get(key, 0.0)
                
            scored.fit = max(0.0, min(1.0, fit))
            
            # Recompute final score
            integrity_factor = scored.integrity
            if scored.integrity < float(config.integrity.get("penalty_threshold", 0.60)):
                integrity_factor = scored.integrity * 0.85
                
            scored.final_score = scored.fit * scored.availability * integrity_factor
            
            # Update the entry score in the list
            item[0] = scored.final_score

    # Re-sort using updated scores and break ties by candidate_id ascending (spec §3)
    scored_entries.sort(key=lambda x: (-x[0], x[1]))
    top = scored_entries[:top_n]
    return [(cand, sc) for _, _, cand, sc in top], excluded


def _assign_monotonic_scores(raw_scores: list[float]) -> list[float]:
    """Map to [0.15, 1.0] strictly decreasing by rank (index order already tie-broken)."""
    n = len(raw_scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    out: list[float] = []
    for i in range(n):
        # Strictly decreasing; step ensures no equal adjacent scores after rounding
        val = round(1.0 - i * (0.85 / (n - 1)), 4)
        out.append(val)
    return out


def build_submission_rows(
    ranked: list[tuple[dict[str, Any], CandidateScore]],
    config: JDConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_idx, (candidate, scored) in enumerate(ranked, start=1):
        # Subtract a tiny epsilon based on rank to guarantee strict monotonicity and prevent any ties
        score_val = round(scored.final_score - 1e-6 * rank_idx, 5)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "rank": rank_idx,
                "score": score_val,
                "reasoning": generate_reasoning(candidate, rank_idx, scored, config),
            }
        )
    return rows


def write_submission_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    candidates_path: str | Path,
    out_path: str | Path,
    config_path: str | Path | None = None,
    top_n: int = 100,
) -> RankingResult:
    t0 = time.perf_counter()
    config = load_jd_config(config_path)

    # Initialize semantic embeddings if they exist
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
    SemanticLoader.initialize(artifacts_dir)

    if Path(candidates_path).suffix == ".json":
        candidates = load_candidates(candidates_path)
    else:
        candidates = list(iter_candidates(candidates_path))

    if len(candidates) < top_n:
        top_n = len(candidates)

    ranked, excluded = rank_candidates(candidates, config, top_n=top_n)
    rows = build_submission_rows(ranked, config)
    write_submission_csv(rows, out_path)
    elapsed = time.perf_counter() - t0

    return RankingResult(
        rows=rows,
        elapsed_seconds=elapsed,
        candidates_processed=len(candidates),
        candidates_excluded=excluded,
    )
