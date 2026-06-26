from src.features.extractor import score_title_career, score_trusted_skills
from src.fit.scorer import compute_fit
from src.scoring.combiner import score_candidate


def _find_by_title(candidates, needle: str):
    for c in candidates:
        title = (c.get("profile", {}).get("current_title") or "").lower()
        if needle in title:
            return c
    return None


def test_negative_title_scores_low(sample_candidates, config):
    bad = _find_by_title(sample_candidates, "marketing") or _find_by_title(
        sample_candidates, "hr"
    )
    if bad is None:
        return
    assert score_title_career(bad, config) < 0.20


def test_ml_title_scores_higher_than_marketing(sample_candidates, config):
    bad = _find_by_title(sample_candidates, "marketing") or _find_by_title(
        sample_candidates, "content writer"
    )
    good = _find_by_title(sample_candidates, "engineer") or _find_by_title(
        sample_candidates, "scientist"
    )
    if bad is None or good is None:
        return
    assert score_title_career(good, config) > score_title_career(bad, config)


def test_fit_in_unit_interval(sample_candidates, config):
    for cand in sample_candidates[:15]:
        fit, features = compute_fit(cand, config)
        assert 0.0 <= fit <= 1.0
        for v in features.values():
            assert 0.0 <= v <= 1.0
