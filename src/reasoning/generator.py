"""Fact-based reasoning strings for submission CSV."""

from __future__ import annotations

import hashlib
from typing import Any

from src.config_loader.jd_config import JDConfig
from src.scoring.combiner import CandidateScore
from src.text.profile_text import build_career_text


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _top_skills(candidate: dict[str, Any], limit: int = 3) -> list[str]:
    skills = candidate.get("skills", [])
    ranked = sorted(
        skills,
        key=lambda s: (
            {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
                _normalize(s.get("proficiency")), 0
            ),
            int(s.get("endorsements") or 0),
        ),
        reverse=True,
    )
    return [s.get("name", "") for s in ranked[:limit] if s.get("name")]


def _skill_matches(name: str, targets: set[str], synonyms: dict[str, list[str]]) -> bool:
    s = _normalize(name)
    if not s:
        return False
    if s in targets:
        return True
    for base, alts in synonyms.items():
        if base in targets and (s == base or s in alts):
            return True
        if s == base and any(alt in targets for alt in alts):
            return True
    return any(t in s or s in t for t in targets if len(t) > 3)


def _collect_skill_matches(candidate: dict[str, Any], config: JDConfig) -> tuple[list[str], list[str]]:
    must: list[str] = []
    nice: list[str] = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if _skill_matches(name, config.expanded_must_have, config.skill_synonyms):
            must.append(name)
        elif _skill_matches(name, config.expanded_nice_have, config.skill_synonyms):
            nice.append(name)
    return must, nice


def _positive_title_matches(candidate: dict[str, Any], config: JDConfig) -> list[str]:
    profile = candidate.get("profile", {})
    title = _normalize(profile.get("current_title"))
    headline = _normalize(profile.get("headline"))
    career_text = build_career_text(candidate).lower()
    matches: list[str] = []
    for keyword in config.role.get("positive_title_keywords", []):
        if keyword and (keyword in title or keyword in headline or keyword in career_text):
            if keyword not in matches:
                matches.append(keyword)
    return matches


def _format_skill_summary(
    must_skills: list[str], nice_skills: list[str], top_skills: list[str], original_skills: list[str]
) -> str:
    # Always include a mention of key skills listed in their profile (to satisfy tests and ensure facts are accurate)
    profile_skills = original_skills[:3]
    profile_phrase = f"Key profile skills include {', '.join(profile_skills)}." if profile_skills else ""

    if must_skills:
        core = ", ".join(must_skills[:3])
        rest = f" with JD-aligned strengths in {', '.join(nice_skills[:2])}" if nice_skills else ""
        jd_phrase = f"Core JD skills include {core}{rest}."
    elif nice_skills:
        jd_phrase = f"Relevant JD-aligned skills include {', '.join(nice_skills[:3])}."
    else:
        jd_phrase = "The profile shows a broad technical background."

    if profile_phrase:
        return f"{profile_phrase} {jd_phrase}"
    return jd_phrase



def _format_experience_phrase(score: float, yoe: float, config: JDConfig) -> str:
    ideal_min = float(config.experience.get("ideal_min", 5))
    ideal_max = float(config.experience.get("ideal_max", 9))
    if score >= 0.95:
        return "Experience is well aligned with the JD's preferred seniority."
    if ideal_min <= yoe <= ideal_max:
        return "Experience is within the JD's preferred range."
    if yoe < ideal_min:
        return f"At {int(yoe)} years of experience, the profile is slightly junior for the ideal JD band."
    return f"At {int(yoe)} years of experience, the profile is slightly more senior than the ideal JD band."


def _format_semantic_phrase(score: float) -> str:
    if score >= 0.75:
        return "Embedding-based semantic fit is strong and supports the profile's JD relevance."
    if score >= 0.45:
        return "Semantic fit is solid, indicating the candidate's profile aligns reasonably with the JD."
    return "Semantic similarity is modest, so the profile's fit is mainly driven by explicit skills and career signals."


def _format_location_phrase(score: float, location: str, rr: float) -> str:
    if score >= 0.85:
        return f"Based in {location} with strong local/logistics fit and good platform responsiveness."
    if score >= 0.55:
        return f"Based in {location} with acceptable fit for location and availability."
    return f"Located in {location}, though location/notice constraints are less ideal."


def _format_concerns(
    score: CandidateScore,
    rr: float,
    notice: int,
    rank: int,
    title: str,
) -> str:
    concerns: list[str] = []
    if notice >= 60:
        concerns.append(f"a long notice period of {notice} days")
    if rr < 0.25:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if score.features.get("title_career", 0) < 0.40:
        concerns.append("limited direct product search or ranking career evidence")
    if score.features.get("skills", 0) < 0.35:
        concerns.append("weak coverage of core JD skills")
    if score.features.get("semantic", 0) < 0.35:
        concerns.append("modest semantic fit to the job description")

    if not concerns:
        return ""

    concern = concerns[0]
    if rank <= 20:
        return f"Still, one concern is {concern}."
    return f"However, there is a concern around {concern}."


def generate_reasoning(
    candidate: dict[str, Any],
    rank: int,
    score: CandidateScore,
    config: JDConfig,
) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    cid = candidate.get("candidate_id", "")

    title = profile.get("current_title", "Professional")
    yoe = float(profile.get("years_of_experience") or 0)
    location = profile.get("location", "Unknown location")
    rr = float(signals.get("recruiter_response_rate") or 0)
    notice = int(signals.get("notice_period_days") or 0)

    top_skills = _top_skills(candidate)
    must_skills, nice_skills = _collect_skill_matches(candidate, config)
    positive_titles = _positive_title_matches(candidate, config)

    original_skills = [s.get("name", "") for s in candidate.get("skills", []) if s.get("name")]
    skill_phrase = _format_skill_summary(must_skills, nice_skills, top_skills, original_skills)
    experience_phrase = _format_experience_phrase(score.features.get("experience", 0.0), yoe, config)
    semantic_phrase = _format_semantic_phrase(score.features.get("semantic", 0.0))
    location_phrase = _format_location_phrase(score.features.get("location", 0.0), location, rr)
    concern_phrase = _format_concerns(score, rr, notice, rank, title)

    h = int(hashlib.md5(cid.encode("utf-8")).hexdigest(), 16)
    intro_options = [
        f"{title} with {int(yoe)} years of experience.",
        f"Current role: {title}, with {int(yoe)} years on record.",
        f"This candidate is a {title} and brings {int(yoe)} years of experience.",
    ]
    jd_focus = ""
    if positive_titles:
        jd_focus = f" The profile matches the JD focus on {positive_titles[0]}."

    intro = intro_options[h % len(intro_options)] + jd_focus

    if rank <= 20:
        tone = "Strong overall fit" if score.final_score > 0.60 else "Good overall fit"
    elif rank <= 60:
        tone = "Moderate fit"
    else:
        tone = "Adjoint fit"

    reasoning = (
        f"{intro} {skill_phrase} {experience_phrase} {semantic_phrase} {location_phrase}. {concern_phrase}".strip()
    )
    reasoning = reasoning.replace("  ", " ")
    return reasoning
