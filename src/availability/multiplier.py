"""AVAILABILITY multiplier from redrob_signals."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from src.config_loader.jd_config import JDConfig


def _days_since(iso_date: str | None, ref: date | None = None) -> int:
    if not iso_date:
        return 9999
    ref = ref or date(2026, 6, 26)
    try:
        d = datetime.strptime(iso_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return 9999
    return max(0, (ref - d).days)


def compute_availability(candidate: dict[str, Any], config: JDConfig) -> float:
    signals = candidate.get("redrob_signals", {})
    mult = 1.0

    stale_days = int(config.availability.get("stale_active_days", 180))
    if _days_since(signals.get("last_active_date")) > stale_days:
        mult -= 0.15

    response_rate = float(signals.get("recruiter_response_rate") or 0)
    low_rr = float(config.availability.get("low_response_rate", 0.20))
    if response_rate < low_rr:
        mult -= 0.12
    elif response_rate >= 0.6:
        mult += 0.04

    if not signals.get("open_to_work_flag", False):
        mult -= 0.08

    notice = int(signals.get("notice_period_days") or 90)
    preferred = int(config.availability.get("preferred_notice_days", 30))
    if notice <= preferred:
        mult += 0.04
    elif notice > 60:
        mult -= 0.06

    if int(signals.get("saved_by_recruiters_30d") or 0) > 0:
        mult += 0.03
    if int(signals.get("profile_views_received_30d") or 0) >= 10:
        mult += 0.02

    icr = float(signals.get("interview_completion_rate") or 0)
    if icr > 0 and icr < 0.4:
        mult -= 0.05

    if signals.get("verified_email") and signals.get("verified_phone"):
        mult += 0.02

    assessments = signals.get("skill_assessment_scores") or {}
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        if avg_assessment >= 60:
            mult += 0.03
        elif avg_assessment < 35:
            mult -= 0.03

    lo = float(config.availability.get("min_multiplier", 0.75))
    hi = float(config.availability.get("max_multiplier", 1.10))
    return max(lo, min(hi, mult))
