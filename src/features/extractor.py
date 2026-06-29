"""Structured feature extraction for FIT scoring."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from src.config_loader.jd_config import JDConfig
from src.text.profile_text import build_career_text, build_profile_text
from src.semantic.loader import SemanticLoader


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(t in text for t in terms)


def _skill_trust(skill: dict[str, Any], config: JDConfig) -> float:
    prof = (skill.get("proficiency") or "beginner").lower()
    pw = config.proficiency_weights.get(prof, 0.25)
    endorsements = int(skill.get("endorsements") or 0)
    duration = int(skill.get("duration_months") or 0)
    if prof == "expert" and duration == 0:
        return 0.0
    duration_factor = min(1.0, duration / 12.0)
    return pw * (1.0 + 0.15 * math.log1p(endorsements)) * (0.35 + 0.65 * duration_factor)


def _normalize_skill_name(name: str) -> str:
    return name.lower().strip()


def _skill_matches(skill_name: str, targets: set[str], synonyms: dict[str, list[str]]) -> bool:
    s = _normalize_skill_name(skill_name)
    if s in targets:
        return True
    for base, alts in synonyms.items():
        if base in targets and (s == base or s in alts):
            return True
        if s == base:
            for alt in alts:
                if alt in targets:
                    return True
    return any(t in s or s in t for t in targets if len(t) > 3)


def score_title_career(candidate: dict[str, Any], config: JDConfig) -> float:
    profile = candidate.get("profile", {})
    title = (profile.get("current_title") or "").lower()
    headline = (profile.get("headline") or "").lower()
    career_text = build_career_text(candidate)

    title_score = 0.35
    for neg in config.role.get("negative_title_keywords", []):
        if neg in title or neg in headline:
            return 0.05
    for pos in config.role.get("positive_title_keywords", []):
        if pos in title or pos in headline:
            title_score = 0.95
            break

    ml_hits = sum(1 for sig in config.ml_career_signals if sig in career_text)
    prod_hits = sum(1 for sig in config.production_signals if sig in career_text)
    career_score = min(1.0, ml_hits / 5.0) * 0.65 + min(1.0, prod_hits / 2.0) * 0.35

    consulting_only = True
    has_product = False
    for role in candidate.get("career_history", []):
        company = (role.get("company") or "").lower()
        industry = (role.get("industry") or "").lower()
        is_consulting = any(cf.lower() in company for cf in config.consulting_firms) or any(
            ind.lower() in industry for ind in config.it_services_industries
        )
        if not is_consulting:
            consulting_only = False
            has_product = True

    penalty = 0.0
    if consulting_only and not has_product:
        penalty = 0.35

    cv_only = sum(1 for sig in config.cv_only_signals if sig in career_text)
    ml_in_career = sum(1 for sig in config.ml_career_signals if sig in career_text)
    if cv_only >= 2 and ml_in_career == 0:
        penalty = max(penalty, 0.25)

    return max(0.0, min(1.0, title_score * 0.45 + career_score * 0.55 - penalty))


def score_trusted_skills(candidate: dict[str, Any], config: JDConfig) -> float:
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    must_hits = 0.0
    nice_hits = 0.0
    for skill in skills:
        name = skill.get("name", "")
        trust = _skill_trust(skill, config)
        if _skill_matches(name, config.expanded_must_have, config.skill_synonyms):
            must_hits += trust
        elif _skill_matches(name, config.expanded_nice_have, config.skill_synonyms):
            nice_hits += trust * 0.5

    must_norm = min(1.0, must_hits / 4.0)
    nice_norm = min(1.0, nice_hits / 2.0)
    return max(0.0, min(1.0, must_norm * 0.80 + nice_norm * 0.20))


def score_experience(candidate: dict[str, Any], config: JDConfig) -> float:
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0) or 0)
    ideal_min = float(config.experience.get("ideal_min", 5))
    ideal_max = float(config.experience.get("ideal_max", 9))
    soft_min = float(config.experience.get("soft_min", 4))
    soft_max = float(config.experience.get("soft_max", 12))

    if ideal_min <= yoe <= ideal_max:
        return 1.0
    if yoe < soft_min or yoe > soft_max:
        return 0.25
    if yoe < ideal_min:
        return 0.5 + 0.5 * (yoe - soft_min) / max(0.1, ideal_min - soft_min)
    return 0.5 + 0.5 * (soft_max - yoe) / max(0.1, soft_max - ideal_max)


def score_location(candidate: dict[str, Any], config: JDConfig) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    preferred_country = (config.location.get("preferred_country") or "India").lower()
    cities = [c.lower() for c in config.location.get("preferred_cities", [])]

    score = 0.2
    if preferred_country in country or country == preferred_country:
        score = 0.55
    if any(city in location for city in cities):
        score = 0.90
    if signals.get("willing_to_relocate"):
        score = max(score, 0.75)

    notice = int(signals.get("notice_period_days") or 90)
    preferred_notice = int(config.availability.get("preferred_notice_days", 30))
    if notice <= preferred_notice:
        score = min(1.0, score + 0.10)
    elif notice > 60:
        score = max(0.0, score - 0.10)

    work_mode = (signals.get("preferred_work_mode") or "").lower()
    if work_mode in {"hybrid", "flexible", "onsite"}:
        score = min(1.0, score + 0.05)
    return max(0.0, min(1.0, score))


def score_semantic(candidate: dict[str, Any], config: JDConfig) -> float:
    """Uses Phase 6 embeddings if available, fallback to lexical overlap."""
    loader = SemanticLoader.get_instance()
    cid = candidate.get("candidate_id")
    if cid:
        sim = loader.get_similarity(cid)
        if sim is not None:
            return sim

    profile_text = build_profile_text(candidate)
    must_hits = sum(1 for s in config.expanded_must_have if s in profile_text)
    prod_hits = sum(1 for s in config.production_signals if s in profile_text)
    return max(0.0, min(1.0, min(1.0, must_hits / 6.0) * 0.7 + min(1.0, prod_hits / 3.0) * 0.3))


def extract_features(candidate: dict[str, Any], config: JDConfig) -> dict[str, float]:
    return {
        "title_career": score_title_career(candidate, config),
        "skills": score_trusted_skills(candidate, config),
        "experience": score_experience(candidate, config),
        "semantic": score_semantic(candidate, config),
        "location": score_location(candidate, config),
    }
