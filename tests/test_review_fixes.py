"""Fixes from the final independent review (A1-A7, B1-B2)."""
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from insurance_assistant import assistant
from insurance_assistant.assistant import answer
from insurance_assistant.extract import extract, keyword_intents, keyword_signals, name_appears_in
from insurance_assistant.guardrails import mentions_regulator
from insurance_assistant.models import Intent, Outcome, StartupError
from insurance_assistant.policy_docs import POLICY_DOC_FILES, check_policy_docs, load_policy_docs
from insurance_assistant.repository import CsvPolicyRepository, ValidatedPolicyRepository
from insurance_assistant.rules.cancellation import evaluate_cancellation

ROOT = Path(__file__).resolve().parent.parent
MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")


class FakeLLM:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def complete(self, system, user, max_tokens, json_schema=None, timeout=None):
        self.calls += 1
        return self.responses[min(self.calls, len(self.responses)) - 1]


def general(text):
    return FakeLLM(json.dumps({"intents": ["general_policy"], "mentions_regulator": False}), text)


@pytest.fixture(scope="module")
def repo():
    return CsvPolicyRepository()


def no_money(decision):
    return not any(k in decision.numbers for k in MONEY_KEYS)


# --- A1: general answers are grounded in the documents only ------------------------------------------

def test_number_from_question_is_not_grounding(repo):
    llm = general("מגיע ללקוח החזר של 5000 ₪ (02_החזרים.md §2).")
    d = answer("מה הנוהל להחזר על חיוב כפול של 5000 ₪ בפוליסה 4471?", repo, llm).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "UNGROUNDED")
    assert any("5000" in w for w in d.warnings)


def test_policy_id_from_question_may_appear(repo):
    llm = general("בפוליסה 4471 החזר על חיוב כפול מגיע לפי הנוהל (02_החזרים.md §1).")
    d = answer("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", repo, llm).decision
    assert d.code == "GENERAL_POLICY" and d.policy_id == "4471"


def test_customer_id_digits_may_appear(repo):
    llm = general("ללקוח C-1088: החזר על חיוב כפול מגיע לפי הנוהל (02_החזרים.md §1).")
    assert answer("מה הנוהל להחזר על חיוב כפול ללקוח C-1088?", repo, llm).decision.code == "GENERAL_POLICY"


# --- A2: regulator terms with Hebrew prefixes -----------------------------------------------------------

@pytest.mark.parametrize("text", ["יגיש תלונה לממונה על הביטוח", "והמפקח יחליט", "פנה לרשות שוק ההון",
                                  "כתב לרגולטור", "שהממונה על הביטוח", "במפקח על הביטוח", "כשמפקח"])
def test_regulator_prefixes(text):
    assert mentions_regulator(text)


@pytest.mark.parametrize("text", ["רוצה לבטל", "תלונה לממונה"])
def test_regulator_needs_a_term(text):
    assert not mentions_regulator(text)


def test_regulator_inflection_escalates(repo):
    d = answer("4471 רוצה לבטל ויגיש תלונה לממונה על הביטוח", repo).decision
    assert (d.outcome, d.code) == (Outcome.ESCALATE, "REGULATOR") and no_money(d)


# --- A3: refund / owed default only without a stripped negated cancellation --------------------------------

def test_refund_after_negated_cancel_is_not_cancellation(repo):
    s = keyword_signals("4413 לא רוצה לבטל את הפוליסה, מבקש החזר בגלל חיוב כפול")
    assert s.negated_cancel and s.refund and keyword_intents(s) == (Intent.UNKNOWN, [])
    d = answer("4413 לא רוצה לבטל את הפוליסה, מבקש החזר בגלל חיוב כפול", repo).decision
    assert d.code == "UNCLEAR_REQUEST" and no_money(d)


def test_refund_alone_still_defaults_to_cancellation():
    assert keyword_intents(keyword_signals("כמה החזר יקבל"))[0] is Intent.CANCELLATION
    assert keyword_intents(keyword_signals("ולא לבטל, מה מגיע לו"))[0] is Intent.UNKNOWN


# --- A4: "אלא" after a negated cancellation ------------------------------------------------------------------

def test_ella_corrects_the_object(repo):
    s = keyword_signals("4428 לא רוצה לבטל את התביעה אלא את הפוליסה")
    assert s.escalation_only_cancel and not s.cancel  # phase 7b: escalation-only
    d = answer("4428 לא רוצה לבטל את התביעה אלא את הפוליסה", repo).decision
    assert (d.outcome, d.code, d.policy_id) == (Outcome.ESCALATE, "OPEN_CLAIM", "4428")


@pytest.mark.parametrize("text", ["לא רוצה לבטל. אלא שאלה אחרת", "לא רוצה לבטל, רק שואל על התביעה",
                                  "לא רוצה לבטל את התביעה; אלא לשאול"])
def test_ella_only_in_the_same_sentence(text):
    assert not keyword_signals(text).cancel


# --- A5: numbers in other contexts ----------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["050-4471-123", "4471-123", "12-4471", "טלפון 050-4471-123"])
def test_hyphenated_digit_groups_are_not_ids(repo, text):
    e = extract(text, repo)
    assert e.policy_id is None and e.unknown_policy_id is None


def test_hebrew_vav_hyphen_still_works(repo):
    e = extract("4418 ו-4433", repo)
    assert e.policy_id == "4418" and e.other_policy_ids == ["4433"]


def test_phone_number_question(repo):
    assert answer("לקוח חדש רוצה לבטל. הטלפון שלו 050-4471-123", repo).decision.code == "NO_POLICY"


def test_unknown_next_to_known_is_not_discarded(repo):
    d = answer("מספר הפוליסה הוא 4461; 4471 הוא מספר הפנייה. הלקוח רוצה לבטל", repo).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "POLICY_NOT_FOUND") and no_money(d)
    assert "4461" in d.headline and any("4461" in m and "4471" in m for m in d.missing)


@pytest.mark.parametrize("text", ["חיוב של 5000 ₪ בפוליסה 4471", "₪5000 בפוליסה 4471", "5000 ש\"ח ל-4471",
                                  "5000 שקל ל-4471"])
def test_amounts_are_not_policy_ids(repo, text):
    e = extract(text, repo)
    assert e.policy_id == "4471" and e.unknown_policy_id is None


def test_year_next_to_known_id_is_ignored(repo):
    assert answer("4471 מ-2023 רוצה לבטל", repo).decision.code == "CANCEL_ESTIMATE"


# --- A6: names on word boundaries --------------------------------------------------------------------------------

@pytest.mark.parametrize("text,found", [
    ("רונן אביטל רוצה לבטל", True), ("לרונן אביטל", True), ("ושלרונן אביטל", False), ("ולרונן אביטל", True),
    ("(רונן אביטל)", True), ("רונן אביטלמן רוצה לבטל", False), ("ארונן אביטל", False), ("Xרונן אביטל", False),
])
def test_name_appears_in(text, found):
    assert name_appears_in("רונן אביטל", text) is found


def test_longer_surname_is_no_match(repo):
    assert extract("רונן אביטלמן רוצה לבטל", repo).name_matches == {}
    assert answer("רונן אביטלמן רוצה לבטל", repo).decision.code == "NO_POLICY"


# --- A7: product mismatch ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["C-1088 רוצה לבטל את ביטוח הדירה", "רונן אביטל רוצה לבטל את ביטוח הדירה"])
def test_product_mismatch(repo, question):
    d = answer(question, repo).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "PRODUCT_MISMATCH") and no_money(d)
    assert "רכב (4471)" in d.headline and [f.label for f in d.facts] == ["פוליסה 4471"]


def test_explicit_policy_id_ignores_product_hint(repo):
    assert answer("4471 רוצה לבטל את ביטוח הדירה", repo).decision.code == "CANCEL_ESTIMATE"


def test_matching_product_still_narrows(repo):
    assert answer("C-1072 רוצה לבטל את ביטוח הדירה", repo).decision.policy_id == "4433"


# --- B1: cited sections must exist ----------------------------------------------------------------------------------

def copy_docs(tmp_path):
    folder = tmp_path / "policies"
    folder.mkdir()
    for name in POLICY_DOC_FILES:
        (folder / name).write_bytes((ROOT / "data" / "policies" / name).read_bytes())
    return folder


@pytest.mark.parametrize("name,old,section", [
    ("01_תנאי_ביטול.md", "## 7.", "§7"),
    ("02_החזרים.md", "**ג.", "§2ג"),
    ("03_טיפול_בתביעות.md", "## 6.", "§6"),
    ("04_חריגים.md", "## ה.", "§ה"),
])
def test_missing_cited_section_fails_startup(tmp_path, name, old, section):
    folder = copy_docs(tmp_path)
    path = folder / name
    path.write_text(path.read_text(encoding="utf-8").replace(old, "טקסט"), encoding="utf-8")
    with pytest.raises(StartupError) as info:
        check_policy_docs(folder)
    assert name in str(info.value) and section in str(info.value)


def test_general_answer_never_assumes_a_section(repo, monkeypatch):
    docs = dict(load_policy_docs())
    docs["02_החזרים.md"] = docs["02_החזרים.md"].split("## 3.")[0]  # §3 gone
    monkeypatch.setattr(assistant, "_docs", lambda directory: docs)
    d = answer("איך מחושב החזר?", repo, general("ההחזר מורכב משלושה רכיבים (02_החזרים.md §2).")).decision
    assert d.code == "GENERAL_POLICY"
    assert "02_החזרים.md §3" not in [f.source for f in d.facts]


# --- B2: last payment after as_of -------------------------------------------------------------------------------------

def test_future_last_payment_means_unknown_arrears():
    repo = ValidatedPolicyRepository(CsvPolicyRepository())
    d = evaluate_cancellation(repo.get("4471"), date(2026, 10, 1), repo)  # last payment 2026-10-05
    assert d.numbers["arrears"] is None and "conditional_estimate" not in d.numbers
    assert any("last_payment_date" in w and "2026-10-05" in w for w in d.warnings)
    assert any("תאריך תשלום אחרון תקין" in m for m in d.missing)


def test_same_day_payment_is_fine():
    repo = ValidatedPolicyRepository(CsvPolicyRepository())
    policy = replace(repo.get("4471"), last_payment_date=date(2026, 10, 25))
    assert evaluate_cancellation(policy, date(2026, 10, 25), repo).numbers["arrears"] is False
