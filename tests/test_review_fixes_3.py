"""Third review round (phase 7c): one normalization, any prefix run, two ids, explicit cancellation only."""
import json

import pytest

from insurance_assistant.assistant import answer
from insurance_assistant.extract import extract, keyword_signals, name_appears_in, normalize_question
from insurance_assistant.models import Intent, Outcome
from insurance_assistant.repository import CsvPolicyRepository
from insurance_assistant.textnorm import normalize_unicode

MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")
ESTIMATES = ("CANCEL_ESTIMATE", "COOLING_OFF", "TRAVEL_BEFORE_TRIP")


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


# --- 1. one normalization before any matching --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("א ב", "א ב"), ("א ב", "א ב"), ("א\tב", "א ב"), ("א\n\nב", "א ב"), ("א  ב", "א ב"),
    ("א‑ב", "א-ב"), ("א־ב", "א-ב"), ("א–ב", "א-ב"), ("א—ב", "א-ב"), ("1−2", "1-2"),
    ("א‐ב", "א-ב"),
])
def test_normalize_unicode(raw, expected):
    assert normalize_unicode(raw) == expected


def test_normalize_question_applies_it():
    assert normalize_question("050‑٤٤٧١ ₪") == "050-4471 ₪"


@pytest.mark.parametrize("text", ["שילם 4413 ₪", "שילם 4413 ₪", "שילם 4413\t₪", "שילם ₪ 4413",
                                  "שילם NIS  4413"])
def test_currency_with_any_whitespace(repo, text):
    e = extract(f"לקוח רוצה לבטל; {text}", repo)
    assert e.policy_id is None and e.unknown_policy_id is None


@pytest.mark.parametrize("text", ["050 - 4413 - 123", "050 -4413- 123", "050- 4413 -123", "050 – 4413 – 123"])
def test_spaced_hyphen_digit_groups(repo, text):
    e = extract(f"טלפון {text}, הלקוח רוצה לבטל", repo)
    assert e.policy_id is None and e.unknown_policy_id is None


def test_hyphen_after_hebrew_word_still_reads_the_id(repo):
    assert extract("פוליסה - 4413 רוצה לבטל", repo).policy_id == "4413"


@pytest.mark.parametrize("text", ["גיל אזולאי‑כהן", "גיל אזולאי‐כהן", "גיל אזולאי‒כהן"])
def test_unicode_hyphen_surnames(text):
    assert not name_appears_in("גיל אזולאי", text)


def test_regulator_and_negation_after_normalization(repo):
    assert extract("4471 יפנה לרשות שוק‑ההון", repo).regulator
    assert not keyword_signals(normalize_question("לא רוצה לבטל, רק שואל")).cancel


def test_general_answer_is_normalized_before_grounding(repo):
    llm = FakeLLM(intents(Intent.GENERAL_POLICY), "מגיע ללקוח החזר של 4471 ₪ (02_החזרים.md §2).")
    d = answer("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", repo, llm).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "UNGROUNDED")


@pytest.mark.parametrize("question", ["לקוח רוצה לבטל; שילם 4413 ₪", "טלפון 050 - 4413 - 123, הלקוח רוצה לבטל",
                                      "גיל אזולאי‑כהן רוצה לבטל"])
def test_normalization_questions(repo, question):
    assert answer(question, repo).decision.code == "NO_POLICY"


# --- 2. any run of prefix letters ------------------------------------------------------------------------

@pytest.mark.parametrize("text,product", [
    ("וכשהדירה תבוטל", "דירה"), ("ולכשהרכב", "רכב"), ("ושבבריאות", "בריאות"), ("וכשבחו\"ל", "נסיעות"),
])
def test_products_with_many_prefixes(repo, text, product):
    assert extract(text, repo).products_mentioned == [product]


@pytest.mark.parametrize("text", ["הדירהX", "רכב-שטח", "דירות", "ברכבת", "אדירה"])
def test_products_need_a_word_boundary(repo, text):
    assert extract(text, repo).products_mentioned == []


@pytest.mark.parametrize("text,found", [
    ("וכשרונן אביטל יתקשר", True), ("ושכשלרונן אביטל", True), ("ארונן אביטל", False), ("וכשרונן אביטל-לוי", False),
])
def test_names_with_many_prefixes(text, found):
    assert name_appears_in("רונן אביטל", text) is found


def test_product_many_prefixes_question(repo):
    d = answer("C-1088 רוצה לבטל וכשהדירה תבוטל תעדכנו אותי", repo).decision
    assert (d.code, no_money(d)) == ("PRODUCT_MISMATCH", True)


# --- 3. two known policy ids -> ask which one ---------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "C-1072 לא את 4418; הוא רוצה לבטל את 4433",
    "4418 רוצה לבטל וגם 4433",
    "4433 או 4418? הלקוח רוצה לבטל",
])
@pytest.mark.parametrize("with_llm", [False, True])
def test_two_policy_ids_never_compute(repo, question, with_llm):
    llm = FakeLLM(intents(Intent.CANCELLATION)) if with_llm else None
    d = answer(question, repo, llm).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "MULTIPLE_POLICIES") and no_money(d)
    assert {f.label for f in d.facts} == {"פוליסה 4418", "פוליסה 4433"}
    assert "additional_request" not in d.numbers


def test_same_id_twice_is_one_policy(repo):
    assert answer("4471 רוצה לבטל, כן 4471", repo).decision.code == "CANCEL_ESTIMATE"


def test_ids_of_two_customers_stay_multiple_customers(repo):
    assert answer("4471 ו-4418 רוצים לבטל", repo).decision.code == "MULTIPLE_CUSTOMERS"


# --- 4. keyword mode: cancellation only from an explicit verb on the policy --------------------------------

@pytest.mark.parametrize("question", [
    "4471 מבקש החזר עקב חיוב כפול",
    "4471 רוצה לבטל את הוראת הקבע",
    "4471 רוצה לבטל את החיוב",
    "4471 רוצה לבטל את התשלום",
    "4471 רוצה לבטל את הכרטיס",
    "4471 רוצה לבטל את התביעה",
    "4471 רוצה לבטל הוראת קבע",
    "4471 מבקש ביטול החיוב האחרון",
    "4471 מה מגיע לו",
])
def test_no_cancellation_without_an_explicit_policy_verb(repo, question):
    d = answer(question, repo).decision
    assert d.code not in ESTIMATES and no_money(d)


@pytest.mark.parametrize("question", ["4471 רוצה לבטל את הפוליסה", "4471 רוצה לבטל את הביטוח", "בקשת ביטול ל4471",
                                      "לקוח 4471 רוצה לבטל; החיוב שלו גבוה"])
def test_explicit_policy_cancellation_still_works(repo, question):
    assert answer(question, repo).decision.code == "CANCEL_ESTIMATE"


def test_non_policy_object_still_escalates_on_open_claim(repo):
    d = answer("4428 רוצה לבטל את הוראת הקבע", repo).decision  # raw keyword: deliberate over-escalation
    assert (d.outcome, d.code) == (Outcome.ESCALATE, "OPEN_CLAIM")


def test_signals_keep_raw_keyword():
    s = keyword_signals("רוצה לבטל את הוראת הקבע")
    assert not s.cancel and s.cancel_keyword_raw
