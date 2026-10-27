"""Properties that must hold for every policy in the export."""
from collections import Counter
from datetime import date

import pytest

from insurance_assistant.models import DEFAULT_AS_OF, Outcome
from insurance_assistant.repository import CsvPolicyRepository, ValidatedPolicyRepository
from insurance_assistant.rules.cancellation import evaluate_cancellation, has_open_or_unknown_claims

MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")


@pytest.fixture(scope="module")
def decisions():
    csv_repo = CsvPolicyRepository()
    repo = ValidatedPolicyRepository(csv_repo)
    results = []
    for pid in csv_repo.policy_ids():
        policy = repo.get(pid)
        results.append((policy, evaluate_cancellation(policy, DEFAULT_AS_OF, repo)))
    counts = Counter(d.code for _, d in results)
    print("\nCancellation codes over all CSV policies:", dict(sorted(counts.items())))
    return results


def test_every_policy_evaluates(decisions):
    assert len(decisions) == 61


def test_open_or_unknown_claims_escalate_without_money(decisions):
    for policy, d in decisions:
        if policy.status == "פעילה" and has_open_or_unknown_claims(policy):
            assert d.outcome is Outcome.ESCALATE, policy.policy_id
        if d.outcome is Outcome.ESCALATE:
            assert not any(k in d.numbers for k in MONEY_KEYS), policy.policy_id


def test_car_estimate_fee_is_150(decisions):
    for policy, d in decisions:
        if policy.product == "רכב" and d.code == "CANCEL_ESTIMATE":
            assert d.numbers["fee"] == 150, policy.policy_id


def test_health_never_has_grant(decisions):
    for policy, d in decisions:
        if policy.product == "בריאות":
            assert "grant" not in d.numbers and "conditional_estimate" not in d.numbers, policy.policy_id


def test_conditional_estimate_consistency(decisions):
    for policy, d in decisions:
        n = d.numbers
        if "conditional_estimate" in n:
            assert n["conditional_estimate"] == round(n["grant"] - n["fee"], 2), policy.policy_id
            assert n["conditional_estimate"] >= 0
            assert n["arrears"] is False


def test_effective_date_not_before_as_of(decisions):
    for policy, d in decisions:
        if "effective_date" in d.numbers:
            assert date.fromisoformat(d.numbers["effective_date"]) >= DEFAULT_AS_OF, policy.policy_id


def test_status_codes(decisions):
    for policy, d in decisions:
        if policy.status == "מוקפאת":
            assert d.code == "FROZEN"
        if policy.status == "מבוטלת":
            assert d.code == "ALREADY_CANCELLED"


def test_answer_has_nothing_missing(decisions):
    for policy, d in decisions:
        if d.outcome is Outcome.ANSWER:
            assert d.missing == [], policy.policy_id


def test_every_fact_has_a_source(decisions):
    for policy, d in decisions:
        assert d.facts and d.facts[0].label == "סטטוס הפוליסה"
        for fact in d.facts:
            assert fact.source and ("§" in fact.source or "policies.csv" in fact.source), fact


def test_script_never_says_at_least(decisions):
    for _, d in decisions:
        assert not any("לפחות" in s for s in d.say_to_customer)
