"""Load and validate jd_requirements.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class JDConfig:
    raw: dict[str, Any]
    fit_mode: str
    role: dict[str, Any]
    experience: dict[str, Any]
    location: dict[str, Any]
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    skill_synonyms: dict[str, list[str]]
    consulting_firms: list[str]
    it_services_industries: list[str]
    production_signals: list[str]
    ml_career_signals: list[str]
    cv_only_signals: list[str]
    weights: dict[str, float]
    integrity: dict[str, float]
    availability: dict[str, float]
    proficiency_weights: dict[str, float]
    expanded_must_have: set[str] = field(default_factory=set)
    expanded_nice_have: set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JDConfig":
        synonyms = data.get("skill_synonyms", {})
        must = [s.lower() for s in data.get("must_have_skills", [])]
        nice = [s.lower() for s in data.get("nice_to_have_skills", [])]
        expanded_must: set[str] = set(must)
        expanded_nice: set[str] = set(nice)
        for base, alts in synonyms.items():
            base_l = base.lower()
            if base_l in must:
                expanded_must.update(a.lower() for a in alts)
            if base_l in nice:
                expanded_nice.update(a.lower() for a in alts)
        return cls(
            raw=data,
            fit_mode=data.get("fit_mode", "rules"),
            role=data.get("role", {}),
            experience=data.get("experience", {}),
            location=data.get("location", {}),
            must_have_skills=must,
            nice_to_have_skills=nice,
            skill_synonyms={k.lower(): [a.lower() for a in v] for k, v in synonyms.items()},
            consulting_firms=[c.lower() for c in data.get("consulting_firms", [])],
            it_services_industries=[i.lower() for i in data.get("it_services_industries", [])],
            production_signals=[s.lower() for s in data.get("production_signals", [])],
            ml_career_signals=[s.lower() for s in data.get("ml_career_signals", [])],
            cv_only_signals=[s.lower() for s in data.get("cv_only_signals", [])],
            weights=data.get("weights", {}),
            integrity=data.get("integrity", {}),
            availability=data.get("availability", {}),
            proficiency_weights=data.get("proficiency_weights", {}),
            expanded_must_have=expanded_must,
            expanded_nice_have=expanded_nice,
        )


def load_jd_config(path: str | Path | None = None) -> JDConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "jd_requirements.yaml"
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return JDConfig.from_dict(data)
