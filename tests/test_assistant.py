import json
import math
from dataclasses import replace
from datetime import date

import pytest

from insurance_assistant import assistant
from insurance_assistant.assistant import (
    BANNER_KEYWORDS,
    BANNER_LLM_UNAVAILABLE,
    GENERAL_SYSTEM_PROMPT,
    answer,
    render,
)
from insurance_assistant.llm import LLMError
from insurance_assistant.models import Decision, Fact, Intent, Outcome
from insurance_assistant.repository import CsvPolicyRepository

MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")


class FakeLLM:
    """Returns the given responses in order; an Exception instance is raised instead."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user, max_tokens, json_schema=None, timeout=None):
        self.calls.append(dict(system=system, user=user, max_tokens=max_tokens,
                               json_schema=json_schema, timeout=timeout))
        response = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response


def intents(*values, regulator=False):
    return json.dumps({"intents": [v.value for v in values], "mentions_regulator": regulator})


class FakeRepository:
    source_name = "fake-api"

    def __init__(self, *policies):
        self.policies = list(policies)

    def get(self, policy_id):
        return next((p for p in self.policies if p.policy_id == policy_id), None)

    def by_customer(self, customer_id):
        return [p for p in self.policies if p.customer_id == customer_id]

    def find_names_in(self, text):
        return [p for p in self.policies if p.customer_name in text]


@pytest.fixture(scope="module")
def repo():
    return CsvPolicyRepository()


def ask(repo, question, llm=None, **kwargs):
    return answer(question, repo, llm, **kwargs)


# --- regulator ------------------------------------------------------------------------------

def test_regulator_escalates_without_llm(repo):
    llm = FakeLLM(intents(Intent.CANCELLATION))
    r = ask(repo, "4471 רוצה לבטל ויתלונן ברשות שוק-ההון", llm)
    d = r.decision
    assert (d.outcome, d.code) == (Outcome.ESCALATE, "REGULATOR")
    assert llm.calls == [] and r.banner is None
    assert d.numbers == {} or "additional_request" not in d.numbers
    assert not any(k in d.numbers for k in MONEY_KEYS)
    assert any(f.source == "04_חריגים.md §ה" for f in d.facts)
    assert any("מנהל המשמרת" in s for s in d.say_to_customer)


def test_regulator_from_llm_flag(repo):
    r = ask(repo, "4471 רוצה לבטל ויגיש תלונה", FakeLLM(intents(Intent.CANCELLATION, regulator=True)))
    assert r.decision.code == "REGULATOR"


# --- policy resolution -------------------------------------------------------------------------

@pytest.mark.parametrize("question,code,policy_id", [
    ("4,471 רוצה ביטול", "CANCEL_ESTIMATE", "4471"),
    ("הלקוח 4471 קנה ב-2023 ורוצה לבטל", "CANCEL_ESTIMATE", "4471"),
    ("4461 רוצה לבטל", "POLICY_NOT_FOUND", None),
    ("מור שפירא רוצה לבטל", "AMBIGUOUS_CUSTOMER", None),
    ("C-1072 רוצה לבטל", "MULTIPLE_POLICIES", None),
    ("C-1072 רוצה לבטל את ביטוח הרכב", "CANCEL_ESTIMATE", "4418"),
    ("C-1088 וגם C-1079 רוצים לבטל", "MULTIPLE_CUSTOMERS", None),
    ("ניר נחום רוצה לבטל את 4471", "CONFLICTING_IDENTIFIERS", None),
    ("רונן אביטל וניר נחום רוצים לבטל, פוליסה 4471", "CONFLICTING_IDENTIFIERS", None),
    ("4471 של C-1072 רוצה לבטל", "CONFLICTING_IDENTIFIERS", None),
    ("C-9999 רוצה לבטל", "CUSTOMER_NOT_FOUND", None),
    ("C-1088 ו-C-9999 רוצים לבטל", "MULTIPLE_CUSTOMERS", None),
    ("דנה רוזן ומיכל רוזן רוצות לבטל", "MULTIPLE_CUSTOMERS", None),
    ("מור שפירא, פוליסה 4413, רוצה לבטל", "CANCEL_ESTIMATE", "4413"),
    ("רונן אביטל רוצה לבטל", "CANCEL_ESTIMATE", "4471"),
    ("C-1088 רוצה לבטל", "CANCEL_ESTIMATE", "4471"),
])
def test_resolution(repo, question, code, policy_id):
    d = ask(repo, question).decision
    assert (d.code, d.policy_id) == (code, policy_id)


def test_ambiguous_and_multiple_list_policies(repo):
    d = ask(repo, "מור שפירא רוצה לבטל").decision
    assert {f.label for f in d.facts} == {"פוליסה 4409", "פוליסה 4413"}
    d = ask(repo, "C-1072 רוצה לבטל").decision
    assert {f.label for f in d.facts} == {"פוליסה 4418", "פוליסה 4433"}


def test_conflicting_records():
    base = CsvPolicyRepository().get("4471")
    fake = FakeRepository(replace(base, conflicting_records=True))
    d = ask(fake, "4471 רוצה לבטל").decision
    assert (d.outcome, d.code, d.policy_id) == (Outcome.CANNOT_ANSWER, "CONFLICTING_RECORDS", "4471")


# --- open-claim escalation ----------------------------------------------------------------------

def test_cancel_keyword_unlisted_escalates_open_claim(repo):
    d = ask(repo, "4428 רוצה לבטל", FakeLLM(intents(Intent.CLAIM_STATUS))).decision
    assert (d.outcome, d.code) == (Outcome.ESCALATE, "OPEN_CLAIM")
    assert not any(k in d.numbers for k in MONEY_KEYS)


def test_cancel_keyword_unlisted_without_claim_never_cancels(repo):
    d = ask(repo, "4471 רוצה לבטל", FakeLLM(intents(Intent.CLAIM_STATUS))).decision
    assert d.code == "NO_OPEN_CLAIM" and d.intent is Intent.CLAIM_STATUS


def test_possible_cancellation_escalates_open_claim(repo):
    llm = FakeLLM(intents(Intent.CLAIM_STATUS, Intent.CANCELLATION))
    d = ask(repo, "מה מצב התביעה של 4425? אולי הוא יפסיק את הביטוח", llm).decision
    assert d.code == "OPEN_CLAIM"


def test_possible_cancellation_without_claim_is_dropped(repo):
    llm = FakeLLM(intents(Intent.POLICY_INFO, Intent.CANCELLATION))
    d = ask(repo, "פרטי 4471, אולי הוא יפסיק את הביטוח", llm).decision
    assert d.code == "POLICY_INFO"


def test_degraded_mode_over_escalates(repo):
    r = ask(repo, "כמה דמי ביטול בפוליסה 4425?", None)
    assert r.decision.code == "OPEN_CLAIM"
    unknown = replace(CsvPolicyRepository().get("4471"), open_claims=None)
    assert ask(FakeRepository(unknown), "כמה דמי ביטול בפוליסה 4471?").decision.code == "UNKNOWN_CLAIMS"


# --- missing policy ----------------------------------------------------------------------------

def test_no_policy(repo):
    d = ask(repo, "הלקוח רוצה לבטל").decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "NO_POLICY")


def test_policy_info_without_policy_becomes_general(repo):
    answer_text = "ביטול נכנס לתוקף בסוף חודש החיוב (01_תנאי_ביטול.md §1)."
    llm = FakeLLM(intents(Intent.POLICY_INFO), answer_text)
    d = ask(repo, "מה הפרטים של הביטול?", llm).decision
    assert d.code == "GENERAL_POLICY" and len(llm.calls) == 2


def test_policy_info_without_policy_and_no_llm(repo):
    assert ask(repo, "פרטי הפוליסה").decision.code == "NO_POLICY"


# --- dispatch ----------------------------------------------------------------------------------

@pytest.mark.parametrize("question,code", [
    ("מתי תשולם התביעה של 4425", "CLAIM_PAYMENT_UNKNOWN"),
    ("מה מצב התביעה של 4428", "CLAIM_OPEN_STAGE_UNKNOWN"),
    ("פרטי פוליסה 4471", "POLICY_INFO"),
    ("אילו מסמכים צריך לתביעה בפוליסה 4471", "CLAIM_DOCS"),
    ("אילו מסמכים צריך לתביעה בביטוח דירה", "CLAIM_DOCS"),
    ("אילו מסמכים צריך לתביעה", "NO_PRODUCT"),
    ("מתי יגיע ההחזר", "REFUND_TIMING"),
    ("4471 שלום", "UNCLEAR_REQUEST"),
    ("שלום", "OUT_OF_SCOPE"),
])
def test_dispatch(repo, question, code):
    assert ask(repo, question).decision.code == code


def test_claim_documents_use_policy_product(repo):
    d = ask(repo, "אילו מסמכים צריך לתביעה בפוליסה 4471").decision
    assert any("הצעת מחיר מוסך" in f.value for f in d.facts)


def test_unclear_request_headline(repo):
    d = ask(repo, "4471 שלום").decision
    assert d.headline == "לא זוהתה הבקשה לגבי פוליסה 4471 – בקש מהלקוח לפרט"
    assert d.policy_id == "4471"


def test_date_out_of_range(repo, monkeypatch):
    def overflow(*args):
        raise OverflowError

    monkeypatch.setattr(assistant, "evaluate_cancellation", overflow)
    d = ask(repo, "4471 רוצה לבטל").decision
    assert (d.outcome, d.code, d.policy_id) == (Outcome.CANNOT_ANSWER, "DATE_OUT_OF_RANGE", "4471")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_calculation_error(repo, monkeypatch, bad):
    def broken(policy, as_of, repo):
        return Decision(Outcome.ANSWER_WITH_GAPS, Intent.CANCELLATION, "CANCEL_ESTIMATE", "x",
                        policy_id=policy.policy_id, numbers={"fee": 150, "grant": bad})

    monkeypatch.setattr(assistant, "evaluate_cancellation", broken)
    r = ask(repo, "4471 רוצה לבטל")
    assert (r.decision.code, r.decision.policy_id) == ("CALCULATION_ERROR", "4471")
    assert not any(k in r.decision.numbers for k in MONEY_KEYS)
    assert "150" not in r.text


# --- general answers ---------------------------------------------------------------------------

GOOD_GENERAL = ("דמי ביטול הם 50 ₪ לפוליסה בוותק של פחות מ-12 חודשים (01_תנאי_ביטול.md §3), "
                "ובפוליסת רכב 150 ₪ בכל מקרה (04_חריגים.md §א).")


def test_general_question_with_policy(repo):
    llm = FakeLLM(intents(Intent.GENERAL_POLICY), GOOD_GENERAL)
    r = ask(repo, "4471 לא מבקש לבטל. רק הסבר כללי: איך מחשבים עלות ביטול?", llm)
    d = r.decision
    assert (d.outcome, d.code, d.policy_id) == (Outcome.ANSWER, "GENERAL_POLICY", "4471")
    assert not any(k in d.numbers for k in MONEY_KEYS)
    assert d.facts[0].label == "תשובת מודל שפה – יש לאמת מול הטקסט המצוטט"
    sources = [f.source for f in d.facts]
    assert "01_תנאי_ביטול.md §3" in sources and "04_חריגים.md §א" in sources
    assert any(f.label == "חריגים הגוברים על הנהלים הבסיסיים" for f in d.facts)
    second = llm.calls[1]
    assert second["system"].startswith(GENERAL_SYSTEM_PROMPT + "\n\n")
    assert "## 6. פוליסה עם תביעה פתוחה" in second["system"] and "## ה. פניות רגולטור" in second["system"]
    assert second["timeout"] == 8.0 and second["max_tokens"] == 1024


def test_general_answer_cites_verbatim_subsection(repo):
    text = "מענק ותק ניתן כאחוז מהפרמיה השנתית (02_החזרים.md §2ג)."
    d = ask(repo, "איך מחשבים מענק ותק?", FakeLLM(intents(Intent.GENERAL_POLICY), text)).decision
    facts = {f.source: f.value for f in d.facts[1:]}
    assert facts["02_החזרים.md §2ג"].startswith("**ג. בתוספת מענק ותק**")
    assert "240 ₪" in facts["02_החזרים.md §2ג"]
    assert facts["02_החזרים.md §3"].startswith("## 3. ביטוח בריאות")
    assert "חריגים הגוברים על הנהלים הבסיסיים" in [f.label for f in d.facts]


def test_general_claims_answer_has_no_exceptions_doc(repo):
    text = "יש להגיש מסמכים רפואיים (03_טיפול_בתביעות.md §4)."
    d = ask(repo, "איך מגישים תביעה בריאות באופן כללי?", FakeLLM(intents(Intent.GENERAL_POLICY), text)).decision
    assert d.code == "GENERAL_POLICY"
    assert "חריגים הגוברים על הנהלים הבסיסיים" not in [f.label for f in d.facts]


@pytest.mark.parametrize("text,code", [
    ("לא מופיע במדיניות", "NOT_IN_POLICY"),
    ("דמי ביטול הם 75 ₪ (01_תנאי_ביטול.md §3).", "UNGROUNDED"),
    ("דמי ביטול הם 50 ₪.", "UNGROUNDED"),
    ("דמי ביטול הם 50 ₪ (01_תנאי_ביטול.md §9).", "UNGROUNDED"),
    ("דמי ביטול הם 50 ₪ (05_אחר.md §1).", "UNGROUNDED"),
    ("מענק (02_החזרים.md §2ד).", "UNGROUNDED"),
])
def test_general_answer_checks(repo, text, code):
    d = ask(repo, "איך מחשבים דמי ביטול?", FakeLLM(intents(Intent.GENERAL_POLICY), text)).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, code)


def test_llm_failure_on_general_answer(repo, capsys):
    llm = FakeLLM(intents(Intent.GENERAL_POLICY), LLMError("secret-detail-503"))
    r = ask(repo, "4471 שואל באופן כללי איך עובד ביטול?", llm)
    d = r.decision
    assert (d.code, d.policy_id) == ("LLM_UNAVAILABLE", "4471")
    assert r.banner == BANNER_LLM_UNAVAILABLE and r.degraded
    assert r.text.startswith(BANNER_LLM_UNAVAILABLE)
    assert "נסה שוב בעוד רגע, או בדוק ישירות במסמכי המדיניות" in d.missing
    assert "secret-detail-503" not in r.text and "secret-detail-503" not in json.dumps(
        [d.headline, d.missing, d.warnings, [f.value for f in d.facts]], ensure_ascii=False)
    assert "secret-detail-503" in capsys.readouterr().err


# --- banners -------------------------------------------------------------------------------------

def test_banner_on_llm_failure(repo):
    r = ask(repo, "4471 רוצה לבטל", FakeLLM("not json"))
    assert r.banner == BANNER_KEYWORDS and r.text.startswith(BANNER_KEYWORDS)
    assert r.decision.code == "CANCEL_ESTIMATE" and "(keywords)" in r.path


def test_no_banner_without_llm(repo):
    r = ask(repo, "4471 רוצה לבטל")
    assert r.banner is None and not r.degraded


def test_no_banner_with_working_llm(repo):
    r = ask(repo, "4471 רוצה לבטל", FakeLLM(intents(Intent.CANCELLATION)))
    assert r.banner is None and "(llm)" in r.path


# --- additional requests ----------------------------------------------------------------------------

def test_additional_secondary_intent(repo):
    llm = FakeLLM(intents(Intent.CANCELLATION, Intent.CLAIM_STATUS))
    d = ask(repo, "4471 רוצה לבטל ומה מצב התביעה", llm).decision
    assert d.code == "CANCEL_ESTIMATE" and d.numbers["additional_request"] is True
    assert any("סטטוס תביעה" in w for w in d.warnings)


def test_additional_other_policy(repo):
    d = ask(repo, "4418 ו-4433 רוצה לבטל").decision
    assert d.policy_id == "4418" and d.numbers["additional_request"] is True
    assert any("4433" in w for w in d.warnings)


def test_additional_request_not_flagged_on_escalation(repo):
    llm = FakeLLM(intents(Intent.CANCELLATION, Intent.CLAIM_STATUS))
    d = ask(repo, "4428 רוצה לבטל ומה מצב התביעה", llm).decision
    assert d.outcome is Outcome.ESCALATE and "additional_request" not in d.numbers
    assert any("סטטוס תביעה" in w for w in d.warnings)


def test_no_additional_request(repo):
    assert "additional_request" not in ask(repo, "4471 רוצה לבטל").decision.numbers


def test_additional_skipped_for_multiple_customers(repo):
    d = ask(repo, "4471 ו-4418 רוצים לבטל").decision
    assert d.code == "MULTIPLE_CUSTOMERS" and d.warnings == []


# --- open-claim safety net ----------------------------------------------------------------------------

OPEN_WARNING = "לפוליסה תביעה פתוחה – אם הלקוח מבקש לבטל, יש להעביר למנהל משמרת (01_תנאי_ביטול.md §6)."
UNKNOWN_WARNING = "לא ידוע אם לפוליסה תביעה פתוחה – אם הלקוח מבקש לבטל, יש להעביר למנהל משמרת (01_תנאי_ביטול.md §6)."


def test_safety_net_open_claim(repo):
    d = ask(repo, "פרטי פוליסה 4425").decision
    assert d.code == "POLICY_INFO" and OPEN_WARNING in d.warnings


def test_safety_net_unknown_claims():
    unknown = replace(CsvPolicyRepository().get("4471"), open_claims=None)
    d = ask(FakeRepository(unknown), "פרטי פוליסה 4471").decision
    assert UNKNOWN_WARNING in d.warnings


def test_safety_net_not_on_escalation_or_clean_policy(repo):
    d = ask(repo, "4425 רוצה לבטל").decision
    assert d.outcome is Outcome.ESCALATE and OPEN_WARNING not in d.warnings
    assert OPEN_WARNING not in ask(repo, "פרטי פוליסה 4471").decision.warnings


# --- rendering and response ------------------------------------------------------------------------------

def test_render_sections_in_order():
    d = Decision(Outcome.ANSWER_WITH_GAPS, Intent.CANCELLATION, "X", "כותרת", policy_id="4471",
                 facts=[Fact("א", "1", "src §1")], warnings=["w"], missing=["m"], assumptions=["a"],
                 say_to_customer=["s"])
    text = render(d)
    assert text.splitlines()[0] == "[תשובה חלקית – חסרים נתונים] כותרת (פוליסה 4471)"
    assert "- א: 1 [src §1]" in text
    order = [text.index(t) for t in ("עובדות:", "אזהרות:", "חסר:", "הנחות:", "מה לומר ללקוח:")]
    assert order == sorted(order)


def test_render_skips_empty_sections():
    text = render(Decision(Outcome.CANNOT_ANSWER, Intent.UNKNOWN, "X", "כותרת"))
    assert text == "[אין תשובה] כותרת"


@pytest.mark.parametrize("outcome,label", [
    (Outcome.ANSWER, "תשובה"), (Outcome.ESCALATE, "העברה למנהל משמרת"), (Outcome.CANNOT_ANSWER, "אין תשובה"),
])
def test_render_labels(outcome, label):
    assert render(Decision(outcome, Intent.UNKNOWN, "X", "h")).startswith(f"[{label}] h")


def test_response_fields(repo):
    r = ask(repo, "4471 רוצה לבטל", as_of=date(2026, 10, 25))
    assert r.path == "intent:cancellation(keywords) → policy:4471 → rules:cancellation → template"
    assert isinstance(r.latency_ms, int) and r.latency_ms >= 0
    assert r.text == render(r.decision)


def test_repo_is_always_validated():
    raw = replace(CsvPolicyRepository().get("4471"), monthly_premium=-5.0)
    d = ask(FakeRepository(raw), "4471 רוצה לבטל").decision
    assert "grant" not in d.numbers
    assert any("monthly_premium" in w for w in d.warnings)
