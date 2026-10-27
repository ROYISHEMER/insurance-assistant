"""Helpers and constants shared by the rule modules."""
from __future__ import annotations

from ..models import Decision, Fact, Intent, Outcome, Policy

CAR, HEALTH, TRAVEL, HOME = "רכב", "בריאות", "נסיעות", "דירה"
PRODUCTS = (CAR, HEALTH, TRAVEL, HOME)
ACTIVE, CANCELLED, FROZEN = "פעילה", "מבוטלת", "מוקפאת"

CANCEL = "01_תנאי_ביטול.md"
REFUNDS = "02_החזרים.md"
CLAIMS = "03_טיפול_בתביעות.md"
EXCEPTIONS = "04_חריגים.md"

MISSING_VALUE = "חסר"


def format_money(amount: float) -> str:
    return f"{amount:,.0f}" if float(amount).is_integer() else f"{amount:,.2f}"


def data_source(policy: Policy) -> str:
    return f"{policy.source}, פוליסה {policy.policy_id}"


def start_decision(policy: Policy, intent: Intent, outcome: Outcome, code: str, headline: str) -> Decision:
    """Every policy decision starts with a status fact and one warning per data issue."""
    return Decision(
        outcome=outcome,
        intent=intent,
        code=code,
        headline=headline,
        policy_id=policy.policy_id,
        facts=[Fact("סטטוס הפוליסה", policy.status or MISSING_VALUE, data_source(policy))],
        warnings=[f"בעיה בנתוני הפוליסה: {issue}" for issue in policy.data_issues],
    )
