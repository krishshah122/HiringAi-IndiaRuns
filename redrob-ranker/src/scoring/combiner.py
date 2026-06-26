"""Combine FIT, AVAILABILITY, INTEGRITY into FINAL score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.availability.multiplier import compute_availability
from src.config_loader.jd_config import JDConfig
from src.fit.scorer import compute_fit
from src.integrity.checker import compute_integrity, should_exclude


@dataclass
class CandidateScore:
    candidate_id: str
    final_score: float
    fit: float
    availability: float
    integrity: float
    features: dict[str, float]
    integrity_issues: list[str]
    excluded: bool


def score_candidate(candidate: dict[str, Any], config: JDConfig) -> CandidateScore:
    cid = candidate.get("candidate_id", "")
    integrity, issues = compute_integrity(candidate, config)
    excluded = should_exclude(integrity, config)

    fit, features = compute_fit(candidate, config)
    availability = compute_availability(candidate, config)

    if excluded:
        final = 0.0
    else:
        integrity_factor = integrity
        if integrity < float(config.integrity.get("penalty_threshold", 0.60)):
            integrity_factor = integrity * 0.85
        final = fit * availability * integrity_factor

    return CandidateScore(
        candidate_id=cid,
        final_score=final,
        fit=fit,
        availability=availability,
        integrity=integrity,
        features=features,
        integrity_issues=issues,
        excluded=excluded,
    )
