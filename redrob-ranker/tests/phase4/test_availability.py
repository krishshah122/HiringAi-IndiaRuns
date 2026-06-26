from src.availability.multiplier import compute_availability


def test_active_candidate_availability_near_one(sample_candidates, config):
    cand = sample_candidates[0]
    avail = compute_availability(cand, config)
    lo = config.availability.get("min_multiplier", 0.75)
    hi = config.availability.get("max_multiplier", 1.10)
    assert lo <= avail <= hi


def test_stale_profile_penalized(config):
    cand = {
        "redrob_signals": {
            "last_active_date": "2024-01-01",
            "recruiter_response_rate": 0.05,
            "open_to_work_flag": False,
            "notice_period_days": 90,
        }
    }
    avail = compute_availability(cand, config)
    assert avail < 0.90
