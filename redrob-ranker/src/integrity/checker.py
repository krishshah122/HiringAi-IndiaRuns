"""INTEGRITY scoring — honeypot and trap detection (rules only)."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from src.config_loader.jd_config import JDConfig
from src.text.profile_text import build_career_text, build_profile_text


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _months_between(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def _title_relevance(title: str, config: JDConfig) -> float:
    title_l = title.lower()
    for neg in config.role.get("negative_title_keywords", []):
        if neg in title_l:
            return 0.0
    for pos in config.role.get("positive_title_keywords", []):
        if pos in title_l:
            return 1.0
    return 0.35


def _career_ml_evidence(career_text: str, config: JDConfig) -> float:
    hits = sum(1 for sig in config.ml_career_signals if sig in career_text)
    return min(1.0, hits / 4.0)


def compute_integrity(candidate: dict[str, Any], config: JDConfig) -> tuple[float, list[str]]:
    """Return (integrity score 0-1, list of issue codes)."""
    score = 1.0
    issues: list[str] = []

    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    yoe = float(profile.get("years_of_experience", 0) or 0)



    expert_zero_duration = 0
    expert_low_trust = 0
    for skill in skills:
        prof = (skill.get("proficiency") or "").lower()
        duration = int(skill.get("duration_months") or 0)
        endorsements = int(skill.get("endorsements") or 0)
        if prof == "expert" and duration == 0:
            expert_zero_duration += 1
            score -= 0.25
            issues.append("expert_zero_duration")
        if prof == "expert" and endorsements == 0 and duration < 6:
            expert_low_trust += 1
            score -= 0.08

    if expert_zero_duration >= 2:
        score -= 0.20
        issues.append("honeypot_skill_fraud")

    expert_count = sum(1 for s in skills if (s.get("proficiency") or "").lower() == "expert")
    stuffer_threshold = int(config.integrity.get("expert_skill_stuffer_count", 8))
    if expert_count >= stuffer_threshold:
        low_trust_experts = sum(
            1
            for s in skills
            if (s.get("proficiency") or "").lower() == "expert"
            and int(s.get("endorsements") or 0) < 3
            and int(s.get("duration_months") or 0) < 12
        )
        if low_trust_experts >= 4:
            score -= 0.25
            issues.append("skill_inflation")

    for edu in education:
        start_y = int(edu.get("start_year") or 0)
        end_y = int(edu.get("end_year") or 0)
        if end_y and start_y and end_y < start_y:
            score -= 0.35
            issues.append("education_timeline")

    career_months = 0
    today = date(2026, 6, 26)
    for role in career:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date")) or today
        if start and end and end < start:
            score -= 0.30
            issues.append("career_date_inversion")
        if start and end:
            career_months += _months_between(start, end)

    if career_months > 0 and yoe > 0:
        implied_years = career_months / 12.0
        if yoe > implied_years + 3.5:
            score -= 0.20
            issues.append("experience_inflation")

    summary = (profile.get("summary") or "").lower()
    if re.search(r"\b(\d{2,})\s+years?\s+(of\s+)?(nlp|ml|ai|machine learning)", summary):
        m = re.search(r"\b(\d{2,})\s+years?", summary)
        if m and int(m.group(1)) > yoe + 2:
            score -= 0.15
            issues.append("narrative_experience_mismatch")

    title = profile.get("current_title", "")
    title_rel = _title_relevance(title, config)
    career_text = build_career_text(candidate)
    ml_evidence = _career_ml_evidence(career_text, config)
    ai_skill_count = sum(
        1
        for s in skills
        if any(sig in (s.get("name") or "").lower() for sig in config.ml_career_signals)
    )
    if ai_skill_count >= 6 and title_rel < 0.4 and ml_evidence < 0.25:
        score -= 0.20
        issues.append("keyword_stuffer")

    score = max(0.0, min(1.0, score))
    return score, issues


def should_exclude(integrity: float, config: JDConfig) -> bool:
    return integrity < float(config.integrity.get("exclude_threshold", 0.30))
