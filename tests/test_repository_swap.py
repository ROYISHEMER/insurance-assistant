"""Switching the data source (CSV -> API) must not change any decision."""
import json
from dataclasses import asdict, replace

import pytest

from insurance_assistant.assistant import answer
from insurance_assistant.repository import CsvPolicyRepository

QUESTIONS = [
    "4,471 רוצה ביטול",
    "הלקוח 4471 קנה ב-2023 ורוצה לבטל",
    "4461 רוצה לבטל",
    "מור שפירא רוצה לבטל",
    "C-1072 רוצה לבטל",
    "C-1072 רוצה לבטל את ביטוח הרכב",
    "C-1088 וגם C-1079 רוצים לבטל",
    "ניר נחום רוצה לבטל את 4471",
    "רונן אביטל וניר נחום רוצים לבטל, פוליסה 4471",
    "4428 רוצה לבטל",
    "4471 לא מבקש לבטל. רק הסבר כללי: איך מחשבים עלות ביטול?",
    "כמה דמי ביטול בפוליסה 4425?",
    "4418 ו-4433 רוצה לבטל",
    "מתי תשולם התביעה של 4425",
    "פרטי פוליסה 4421",
    "4419 רוצה לבטל",
    "4401 רוצה לבטל",
    "4455 רוצה לבטל",
    "4471 שלום",
]


class InMemoryRepository:
    """Stands in for the production API: same policies, different source."""

    source_name = "policy-api"

    def __init__(self, policies):
        self._policies = [replace(p, source="", data_issues=list(p.data_issues)) for p in policies]

    def get(self, policy_id):
        return next((p for p in self._policies if p.policy_id == str(policy_id)), None)

    def by_customer(self, customer_id):
        return [p for p in self._policies if p.customer_id == customer_id]

    def find_names_in(self, text):
        return [p for p in self._policies if p.customer_name and p.customer_name in text]


@pytest.fixture(scope="module")
def repos():
    csv_repo = CsvPolicyRepository()
    memory = InMemoryRepository([csv_repo.get(pid) for pid in csv_repo.policy_ids()])
    return csv_repo, memory


def normalized(response, source_name):
    text = json.dumps(asdict(response.decision), ensure_ascii=False, default=str)
    return text.replace(source_name, "<source>"), response.text.replace(source_name, "<source>")


@pytest.mark.parametrize("question", QUESTIONS)
def test_same_decision_for_any_source(repos, question):
    csv_repo, memory = repos
    from_csv = answer(question, csv_repo, llm=None)
    from_api = answer(question, memory, llm=None)
    assert normalized(from_csv, "policies.csv") == normalized(from_api, "policy-api")
    assert from_csv.path == from_api.path


def test_source_name_is_used(repos):
    _, memory = repos
    d = answer("4471 רוצה לבטל", memory).decision
    assert any(f.source == "policy-api, פוליסה 4471" for f in d.facts)
