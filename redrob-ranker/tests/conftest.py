"""Shared pytest fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader.jd_config import load_jd_config


@pytest.fixture
def config():
    return load_jd_config(ROOT / "config" / "jd_requirements.yaml")


@pytest.fixture
def sample_candidates():
    path = PARENT / "sample_candidates.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def honeypot_candidate():
    return {
        "candidate_id": "CAND_9999999",
        "profile": {
            "current_title": "AI Engineer",
            "headline": "AI Engineer",
            "summary": "15 years NLP expert",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 4.0,
        },
        "career_history": [
            {
                "company": "StartupX",
                "title": "AI Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 48,
                "description": "NLP",
            }
        ],
        "education": [{"institution": "X", "start_year": 2020, "end_year": 2018}],
        "skills": [
            {"name": "RAG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "LoRA", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ],
        "redrob_signals": {
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 15,
            "verified_email": True,
            "verified_phone": True,
        },
    }
