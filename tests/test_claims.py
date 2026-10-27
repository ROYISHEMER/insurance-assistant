import re
from dataclasses import replace

import pytest

from insurance_assistant.models import Intent, Outcome
from insurance_assistant.repository import CsvPolicyRepository, ValidatedPolicyRepository
from insurance_assistant.rules.claims import (
    claim_documents,
    claim_payment_date,
    claim_status,
    policy_info,
    refund_timing,
)


@pytest.fixture(scope="module")
def repo():
    return ValidatedPolicyRepository(CsvPolicyRepository())


def assert_cited(decision):
    for fact in decision.facts:
        assert fact.source and ("§" in fact.source or "policies.csv" in fact.source), fact


def test_payment_date_with_open_claim(repo):
    d = claim_payment_date(repo.get("4425"))
    assert (d.outcome, d.code, d.intent) == (Outcome.CANNOT_ANSWER, "CLAIM_PAYMENT_UNKNOWN", Intent.CLAIM_PAYMENT_DATE)
    assert d.numbers == {}
    stages = [f for f in d.facts if f.source == "03_טיפול_בתביעות.md §2"]
    assert len(stages) == 4
    assert "3 ימי עסקים" in stages[0].value and "10 ימי עסקים" in stages[1].value
    assert "אין לוח זמנים" in stages[2].value and "אין לוח זמנים" in stages[3].value
    assert any("השלב הנוכחי" in m for m in d.missing)
    assert any("לוח זמנים להחלטה ולתשלום" in m for m in d.missing)
    assert any("המלצת המעריך" in w and "§3" in w and "§6" in w for w in d.warnings)
    assert_cited(d)


def test_payment_date_never_states_a_date_or_amount(repo):
    d = claim_payment_date(repo.get("4428"))
    assert "₪" not in " ".join(d.say_to_customer)
    assert not re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", " ".join(d.say_to_customer))


@pytest.mark.parametrize("ask", [claim_payment_date, claim_status])
def test_no_open_claim(repo, ask):
    d = ask(repo.get("4471"))
    assert (d.outcome, d.code) == (Outcome.ANSWER, "NO_OPEN_CLAIM")
    assert d.missing == []


def test_claim_status_open(repo):
    d = claim_status(repo.get("4428"))
    assert (d.outcome, d.code, d.intent) == (Outcome.ANSWER_WITH_GAPS, "CLAIM_OPEN_STAGE_UNKNOWN", Intent.CLAIM_STATUS)
    assert any(f.source == "03_טיפול_בתביעות.md §3" for f in d.facts)
    assert any("המלצת המעריך" in w for w in d.warnings)
    assert d.numbers == {}
    assert_cited(d)


@pytest.mark.parametrize("ask", [claim_payment_date, claim_status])
def test_unknown_claims(repo, ask):
    policy = replace(repo.get("4471"), open_claims=None)
    d = ask(policy)
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "CLAIM_COUNT_UNKNOWN")
    assert d.missing


@pytest.mark.parametrize("product,expected", [
    ("רכב", "הצעת מחיר מוסך"),
    ("דירה", "דוח שמאי אם הנזק מעל 10,000 ₪"),
    ("בריאות", "הפניית רופא"),
    ("נסיעות", "כרטיסי טיסה"),
])
def test_claim_documents(product, expected):
    d = claim_documents(product, "4471")
    assert (d.outcome, d.code, d.policy_id) == (Outcome.ANSWER, "CLAIM_DOCS", "4471")
    sources = [f.source for f in d.facts]
    assert sources.count("03_טיפול_בתביעות.md §1") == 2
    doc_fact = next(f for f in d.facts if f.source == "03_טיפול_בתביעות.md §4")
    assert expected in doc_fact.value
    assert any("האתר, האפליקציה" in f.value for f in d.facts)


@pytest.mark.parametrize("product", ["אופנוע", None, ""])
def test_claim_documents_unknown_product(product):
    d = claim_documents(product)
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "NO_PRODUCT")
    assert d.missing


def test_refund_timing():
    d = refund_timing("4471")
    assert (d.outcome, d.code, d.policy_id) == (Outcome.ANSWER, "REFUND_TIMING", "4471")
    assert all(f.source == "02_החזרים.md §4" for f in d.facts)
    values = " ".join(f.value for f in d.facts)
    assert "14 ימי עסקים" in values and "אימות זהות" in values
    assert refund_timing().policy_id is None


def test_policy_info_4471(repo):
    d = policy_info(repo.get("4471"))
    assert (d.outcome, d.code) == (Outcome.ANSWER, "POLICY_INFO")
    assert len(d.facts) == 10
    values = {f.label: f.value for f in d.facts}
    assert values["מספר פוליסה"] == "4471"
    assert values["שם הלקוח"] == "רונן אביטל"
    assert values["תאריך תחילה"] == "2023-02-10"
    assert values["פרמיה חודשית"] == "410 ₪"
    assert values["תביעות פתוחות"] == "0"
    assert values["הערות"] == "אין הערות"
    assert all(f.source == "policies.csv, פוליסה 4471" for f in d.facts)
    assert d.warnings == []


def test_policy_info_missing_values_and_issues(repo):
    policy = replace(repo.get("4421"), monthly_premium=None, open_claims=None)
    d = policy_info(policy)
    values = {f.label: f.value for f in d.facts}
    assert values["פרמיה חודשית"] == "חסר" and values["תביעות פתוחות"] == "חסר"
    assert len(d.warnings) == len(policy.data_issues) == 2
