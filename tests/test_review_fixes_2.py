"""Second round of review fixes (phase 7b): each fixed at the level of the principle."""
import json

import pytest

from eval import stability
from insurance_assistant.assistant import answer
from insurance_assistant.extract import extract, keyword_signals, name_appears_in, normalize_question
from insurance_assistant.grounding import currency_amounts
from insurance_assistant.guardrails import mentions_regulator
from insurance_assistant.models import Intent, Outcome
from insurance_assistant.repository import CsvPolicyRepository

MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")


class FakeLLM:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def complete(self, system, user, max_tokens, json_schema=None, timeout=None):
        self.calls += 1
        return self.responses[min(self.calls, len(self.responses)) - 1]


def intents(*values):
    return json.dumps({"intents": [v.value for v in values], "mentions_regulator": False})


@pytest.fixture(scope="module")
def repo():
    return CsvPolicyRepository()


def no_money(d):
    return not any(k in d.numbers for k in MONEY_KEYS)


# --- 1. regulator terms as plain substrings -------------------------------------------------------

@pytest.mark.parametrize("text", [
    "ושהממונה על הביטוח יטפל", "וכשהמפקח יגיע", "לרשות שוק ההון",
    "וכשלממונה על הביטוח", "שוק־ההון", "הרגולטורים", "אמפקח",  # extra: plural, glued letter
])
def test_regulator_substrings(text):
    assert mentions_regulator(text)


def test_regulator_many_prefixes_escalates(repo):
    llm = FakeLLM(intents(Intent.CANCELLATION))
    d = answer("4471 רוצה לבטל ואומר שיתלונן ושהממונה על הביטוח יטפל בזה", repo, llm).decision
    assert (d.outcome, d.code) == (Outcome.ESCALATE, "REGULATOR") and no_money(d)
    assert llm.calls == 0


# --- 2. currency-marked amounts in general answers --------------------------------------------------

def general(text):
    return FakeLLM(intents(Intent.GENERAL_POLICY), text)


@pytest.mark.parametrize("question,text", [
    ("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", "מגיע ללקוח החזר של 4471 ₪ (02_החזרים.md §2)."),
    ("מה הנוהל להחזר על חיוב כפול ללקוח C-1088?", "מגיע ללקוח החזר של 1,088 ₪ (02_החזרים.md §2)."),
    ("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", "מגיע ללקוח החזר של ₪4471 (02_החזרים.md §2)."),
    ("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", "ההחזר: NIS 4471 (02_החזרים.md §2)."),
    ("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", 'ההחזר 4471 ש"ח (02_החזרים.md §2).'),
])
def test_currency_amount_ignores_identifier_exception(repo, question, text):
    d = answer(question, repo, general(text)).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "UNGROUNDED")


def test_bare_identifier_is_still_allowed(repo):
    d = answer("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", repo,
               general("בפוליסה 4471 החזר על חיוב כפול מגיע לפי הנוהל (02_החזרים.md §1).")).decision
    assert d.code == "GENERAL_POLICY"


def test_document_amount_with_currency_is_allowed(repo):
    d = answer("כמה דמי ביטול ברכב?", repo, general("בפוליסת רכב דמי הביטול הם 150 ₪ (04_חריגים.md §א).")).decision
    assert d.code == "GENERAL_POLICY"


def test_currency_amounts():
    assert currency_amounts('4471 ₪, ₪ 12, NIS 5, 7 nis, 1,088 ש"ח, 9 ש״ח, 3 שקלים, 4 שח') == \
        {"4471", "12", "5", "7", "1088", "9", "3", "4"}
    assert currency_amounts("פוליסה 4471, 14 ימים, NISSAN 5") == set()


# --- 3. "אלא" without its own cancel verb only escalates -----------------------------------------------

@pytest.mark.parametrize("question", [
    "4413 לא רוצה לבטל אלא לקבל החזר על חיוב כפול",
    "4471 לא מעוניין לבטל אלא לדעת מה מגיע לו על חיוב שגוי",
    "4471 לא רוצה לבטל את התביעה אלא את הפוליסה",  # extra: same phrasing, no open claim
])
@pytest.mark.parametrize("llm_intent", [None, Intent.CANCELLATION, Intent.REFUND_TIMING])
def test_ella_never_produces_a_cancellation_answer(repo, question, llm_intent):
    llm = FakeLLM(intents(llm_intent)) if llm_intent else None
    d = answer(question, repo, llm).decision
    assert d.code not in ("CANCEL_ESTIMATE", "COOLING_OFF", "TRAVEL_BEFORE_TRIP") and no_money(d)


@pytest.mark.parametrize("llm_intent", [None, Intent.CANCELLATION, Intent.CLAIM_STATUS])
def test_ella_escalates_on_open_claim(repo, llm_intent):
    llm = FakeLLM(intents(llm_intent)) if llm_intent else None
    d = answer("4428 לא רוצה לבטל את התביעה אלא את הפוליסה", repo, llm).decision
    assert (d.outcome, d.code) == (Outcome.ESCALATE, "OPEN_CLAIM")


def test_ella_with_its_own_cancel_verb_is_a_normal_cancel(repo):
    s = keyword_signals("4471 לא רוצה לבטל את התביעה אלא לבטל את הביטוח")
    assert s.cancel and not s.escalation_only_cancel
    assert answer("4471 לא רוצה לבטל את התביעה אלא לבטל את הביטוח", repo).decision.code == "CANCEL_ESTIMATE"


def test_ella_in_another_sentence_is_not_a_signal():
    s = keyword_signals("4471 לא רוצה לבטל. אלא מה, הוא שואל על התביעה")
    assert not s.escalation_only_cancel and not s.cancel


# --- 4. number context ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["050־4471־123", "050–4471–123", "050—4471—123", "050−4471−123"])
def test_dashes_between_digits_become_hyphens(repo, text):
    assert normalize_question(text) == "050-4471-123"
    e = extract(f"טלפון {text}, הלקוח רוצה לבטל", repo)
    assert e.policy_id is None and e.unknown_policy_id is None


def test_maqaf_between_words_is_untouched():
    assert normalize_question("שוק־ההון ו־4471") == "שוק־ההון ו־4471"


@pytest.mark.parametrize("text", ["NIS 4471", 'ש"ח 4471', "ש״ח 4471", "4471 nis", "4471 NIS", "₪ 4471", "4471שקל"])
def test_currency_tokens_mark_amounts(repo, text):
    e = extract(f"לקוח רוצה לבטל; החיוב שלו {text}", repo)
    assert e.policy_id is None and e.unknown_policy_id is None


def test_two_spaces_before_currency_is_not_an_amount(repo):
    assert extract("פוליסה 4471  ₪", repo).policy_id == "4471"


@pytest.mark.parametrize("question", ["טלפון 050־4471־123, הלקוח רוצה לבטל", "לקוח רוצה לבטל; החיוב שלו NIS 4471"])
def test_number_context_questions(repo, question):
    assert answer(question, repo).decision.code == "NO_POLICY"


# --- 5. name boundaries --------------------------------------------------------------------------------

@pytest.mark.parametrize("name,text", [
    ("איתי מזרחי", "איתי מזרחי-פרץ רוצה לבטל"),
    ("גיל אזולאי", "גיל אזולאי־לוי רוצה לבטל"),
    ("רונן אביטל", "רונן אביטלX רוצה לבטל"),
    ("רונן אביטל", "רונן אביטל–כהן"),
    ("רונן אביטל", "רונן אביטל—Smith"),
])
def test_names_followed_by_letters_or_dash_letter(name, text):
    assert not name_appears_in(name, text)


@pytest.mark.parametrize("text", ["רונן אביטל - רוצה לבטל", "רונן אביטל-4471", "רונן אביטל, רוצה לבטל", "רונן אביטל."])
def test_names_followed_by_non_letters_still_match(text):
    assert name_appears_in("רונן אביטל", text)


def test_compound_surname_question(repo):
    assert answer("איתי מזרחי-פרץ רוצה לבטל", repo).decision.code == "NO_POLICY"


# --- 6. every mentioned product must belong to the customer ----------------------------------------------

@pytest.mark.parametrize("question", [
    "C-1088 מבקש לבטל את ביטוח הבריאות ולא את הרכב",
    "C-1088 רוצה לבטל את הדירה ולא את הרכב",
    "רונן אביטל רוצה לבטל את הדירה ולא את הרכב",  # extra: resolved by name
])
def test_any_foreign_product_is_a_mismatch(repo, question):
    d = answer(question, repo).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "PRODUCT_MISMATCH") and no_money(d)


def test_several_owned_products_are_multiple_policies(repo):
    assert answer("C-1072 רוצה לבטל את הדירה או את הרכב", repo).decision.code == "MULTIPLE_POLICIES"


def test_explicit_policy_id_is_not_checked_against_products(repo):
    assert answer("4471 רוצה לבטל את הרכב ולא את הדירה", repo).decision.code == "CANCEL_ESTIMATE"


def test_products_mentioned_are_all_collected(repo):
    e = extract("C-1088 רוצה לבטל את הדירה ולא את הרכב", repo)
    assert e.products_mentioned == ["רכב", "דירה"] and e.product_hint is None


# --- 7. stability prints the number of runs ------------------------------------------------------------------

def test_stability_prints_runs(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(stability, "get_llm", lambda: object())
    monkeypatch.setattr(stability, "load_cases", lambda: [{"id": "x", "set": "core", "q": "q", "outcome": "ANSWER"}])
    monkeypatch.setattr(stability, "run_case", lambda case, repo, llm: {"passed": True, "reasons": [], "response": None})
    monkeypatch.setattr(stability, "classify", lambda case, results: ("STABLE-CORRECT", []))
    monkeypatch.setattr(stability, "adversarial", lambda repo, llm: 0)
    assert stability.main(["--runs", "3"]) == 0
    out = capsys.readouterr().out
    assert "1 cases × 3 runs each" in out and "Summary (3 runs per case)" in out
