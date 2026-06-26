"""Build canonical text representations for profiles."""

from __future__ import annotations

from typing import Any


def build_profile_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("current_title", ""),
        profile.get("summary", ""),
    ]
    
    # We only include job titles to save massive amounts of CPU time, dropping heavy descriptions
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
        
    return " ".join(p for p in parts if p).lower()


def build_career_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("company", ""))
        parts.append(role.get("description", ""))
    return " ".join(p for p in parts if p).lower()


def build_jd_text(config: Any) -> str:
    parts = []
    parts.extend(config.role.get("positive_title_keywords", []))
    parts.extend(config.must_have_skills)
    parts.extend(config.nice_to_have_skills)
    parts.extend(config.ml_career_signals)
    parts.extend(config.production_signals)
    return " ".join(parts).lower()


def build_rich_profile_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("current_title", ""),
        profile.get("summary", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("company", ""))
        parts.append(role.get("industry", ""))
        desc = (role.get("description") or "").strip()
        if desc:
            parts.append(desc[:300])
            
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
        
    return " ".join(p for p in parts if p).lower()
