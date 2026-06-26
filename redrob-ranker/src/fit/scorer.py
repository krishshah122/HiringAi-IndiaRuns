"""FIT score — weighted combination of feature sub-scores."""

from __future__ import annotations

from typing import Any

from src.config_loader.jd_config import JDConfig
from src.features.extractor import extract_features


def compute_fit(candidate: dict[str, Any], config: JDConfig) -> tuple[float, dict[str, float]]:
    features = extract_features(candidate, config)
    weights = config.weights
    total_w = sum(weights.get(k, 0.0) for k in ("title_career", "skills", "experience", "semantic", "location"))
    if total_w <= 0:
        total_w = 1.0

    fit = 0.0
    for key in ("title_career", "skills", "experience", "semantic", "location"):
        w = weights.get(key, 0.0) / total_w
        fit += w * features.get(key, 0.0)

    return max(0.0, min(1.0, fit)), features
