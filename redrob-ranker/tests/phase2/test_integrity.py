from src.integrity.checker import compute_integrity, should_exclude
from src.scoring.combiner import score_candidate


def test_honeypot_low_integrity(honeypot_candidate, config):
    integrity, issues = compute_integrity(honeypot_candidate, config)
    assert integrity < 0.60
    assert "expert_zero_duration" in issues or "education_timeline" in issues


def test_honeypot_excluded(honeypot_candidate, config):
    integrity, _ = compute_integrity(honeypot_candidate, config)
    assert should_exclude(integrity, config) or integrity < 0.60


def test_clean_sample_has_reasonable_integrity(sample_candidates, config):
    for cand in sample_candidates[:10]:
        integrity, _ = compute_integrity(cand, config)
        assert integrity >= 0.40
