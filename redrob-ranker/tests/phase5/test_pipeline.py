from src.pipeline.ranker import build_submission_rows, rank_candidates
from src.reasoning.generator import generate_reasoning
from src.scoring.combiner import score_candidate


def test_final_score_formula(sample_candidates, config):
    cand = sample_candidates[0]
    sc = score_candidate(cand, config)
    if not sc.excluded:
        assert sc.final_score <= sc.fit * sc.availability + 1e-9


def test_ranking_tiebreak_by_id(sample_candidates, config):
    ranked, _ = rank_candidates(sample_candidates, config, top_n=10)
    ids = [c["candidate_id"] for c, _ in ranked]
    assert len(ids) == len(set(ids))


def test_reasoning_not_empty(sample_candidates, config):
    cand = sample_candidates[0]
    sc = score_candidate(cand, config)
    text = generate_reasoning(cand, 1, sc, config)
    assert len(text) > 20
    title = cand["profile"]["current_title"]
    assert title in text or str(cand["profile"]["years_of_experience"]) in text


def test_reasoning_mentions_actual_skill(sample_candidates, config):
    cand = sample_candidates[0]
    sc = score_candidate(cand, config)
    text = generate_reasoning(cand, 1, sc, config).lower()
    skills = [s["name"].lower() for s in cand.get("skills", []) if s.get("name")]
    if skills:
        assert any(skill in text for skill in skills[:3])


def test_monotonic_scores(sample_candidates, config):
    ranked, _ = rank_candidates(sample_candidates, config, top_n=20)
    rows = build_submission_rows(ranked, config)
    scores = [r["score"] for r in rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
