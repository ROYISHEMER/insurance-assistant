"""Core data types shared by every layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class StartupError(Exception):
    """The data source or the procedure documents are unusable. The message is shown as is."""


@dataclass
class Policy:
    policy_id: str
    customer_id: str
    customer_name: str
    product: str
    start_date: date | None
    monthly_premium: float | None
    status: str
    last_payment_date: date | None
    open_claims: int | None
    notes: str = ""
    data_issues: list[str] = field(default_factory=list)
    source: str = ""
    conflicting_records: bool = False


class Outcome(Enum):
    ANSWER = "ANSWER"
    ANSWER_WITH_GAPS = "ANSWER_WITH_GAPS"
    ESCALATE = "ESCALATE"
    CANNOT_ANSWER = "CANNOT_ANSWER"


class Intent(Enum):
    CANCELLATION = "cancellation"
    CLAIM_STATUS = "claim_status"
    CLAIM_PAYMENT_DATE = "claim_payment_date"
    CLAIM_DOCUMENTS = "claim_documents"
    REFUND_TIMING = "refund_timing"
    POLICY_INFO = "policy_info"
    GENERAL_POLICY = "general_policy"
    UNKNOWN = "unknown"


@dataclass
class Fact:
    label: str
    value: str
    source: str


@dataclass
class Decision:
    outcome: Outcome
    intent: Intent
    code: str
    headline: str
    policy_id: str | None = None
    facts: list[Fact] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    say_to_customer: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)


DEFAULT_AS_OF = date(2026, 10, 25)  # the CSV export date; never today's date
