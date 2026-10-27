import json
from types import SimpleNamespace

import pytest

from insurance_assistant.extract import (
    INTENT_SCHEMA,
    INTENT_SYSTEM_PROMPT,
    Signals,
    extract,
    keyword_evidence,
    keyword_intents,
    keyword_signals,
    normalize_question,
    pick_llm_intents,
    strip_negations,
)
from insurance_assistant.grounding import numbers_in, ungrounded_numbers
from insurance_assistant.guardrails import mentions_regulator, normalize_text
from insurance_assistant.llm import AnthropicLLM, LLMError, get_llm
from insurance_assistant.models import Intent
from insurance_assistant.repository import CsvPolicyRepository, ValidatedPolicyRepository

C = Intent.CANCELLATION
STATUS = Intent.CLAIM_STATUS
PAYMENT = Intent.CLAIM_PAYMENT_DATE
DOCS = Intent.CLAIM_DOCUMENTS
REFUND = Intent.REFUND_TIMING
INFO = Intent.POLICY_INFO
GENERAL = Intent.GENERAL_POLICY
UNKNOWN = Intent.UNKNOWN


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def complete(self, system, user, max_tokens, json_schema=None, timeout=None):
        self.calls.append(dict(system=system, user=user, max_tokens=max_tokens, json_schema=json_schema))
        if self.error:
            raise self.error
        return self.response


def llm_says(*intents, regulator=False):
    return FakeLLM(json.dumps({"intents": [i.value for i in intents], "mentions_regulator": regulator}))


@pytest.fixture(scope="module")
def repo():
    return ValidatedPolicyRepository(CsvPolicyRepository())


# --- guardrails -----------------------------------------------------------------------

def test_normalize_text():
    assert normalize_text("שוק-ההון") == "שוק ההון"
    assert normalize_text('חו"ל  ו־׳א׳') == "חול ו א"


@pytest.mark.parametrize("text", [
    "הלקוח אמר שיתלונן ברשות שוק ההון",
    "יפנה לרשות שוק-ההון",
    "יפנה לרשות שוק־ההון",
    "יפנה לרשות שוק–ההון",
    "יפנה לרשות שוק—ההון",
    "הוא כבר כתב לרגולטור",
    "תלונה לממונה: הממונה על-הביטוח",
    "יפנה למפקח על הביטוח",
    "יפנה לפיקוח על הביטוח",
    "הוא אמר שידבר עם המפקח",
])
def test_regulator_variants(text):
    assert mentions_regulator(text)


def test_no_regulator():
    assert not mentions_regulator("לקוח 4471 רוצה לבטל, מה מגיע לו?")


def test_regulator_keyword_skips_llm(repo):
    llm = llm_says(C)
    e = extract("לקוח 4471 רוצה לבטל ויפנה לרשות שוק־ההון", repo, llm)
    assert llm.calls == []
    assert (e.regulator, e.via, e.llm_failed) == (True, "keywords", False)
    assert e.policy_id == "4471"


def test_llm_regulator_is_ored_in(repo):
    e = extract("לקוח 4471 רוצה לבטל ואמר שיגיש תלונה", repo, llm_says(C, regulator=True))
    assert e.regulator and e.via == "llm"


# --- grounding ---------------------------------------------------------------------------

def test_numbers_in():
    assert numbers_in("החזר של 1,234 ₪ ו-342.00 ₪, פוליסה 4471") == {"1234", "342", "4471"}


def test_ungrounded_numbers():
    assert ungrounded_numbers("כ-342 ₪ תוך 14 ימים", "342 ... 14 ימי עסקים") == []
    assert ungrounded_numbers("כ-350 ₪ עד 15.11", "342") == ["15.11", "350"]


# --- llm client -----------------------------------------------------------------------------

def fake_client(stop_reason="end_turn", content=None, error=None):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        blocks = content if content is not None else [SimpleNamespace(type="text", text='{"a": 1}')]
        return SimpleNamespace(stop_reason=stop_reason, content=blocks)

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def test_llm_request_shape(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    client, calls = fake_client()
    llm = AnthropicLLM(client=client)
    assert llm.model == "claude-haiku-5-5"
    assert llm.complete("sys", "q", 1024, json_schema=INTENT_SCHEMA, timeout=2.5) == '{"a": 1}'
    sent = calls[0]
    assert "temperature" not in sent
    assert sent["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": INTENT_SCHEMA}}
    assert sent["timeout"] == 2.5 and sent["system"] == "sys" and sent["max_tokens"] == 1024
    llm.complete("sys", "q", 10)
    assert calls[1]["output_config"] == {"effort": "low"} and "timeout" not in calls[1]


def test_llm_model_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    assert AnthropicLLM(client=fake_client()[0]).model == "claude-sonnet-5-5"


@pytest.mark.parametrize("kwargs", [
    dict(error=TimeoutError()),
    dict(error=RuntimeError("boom")),
    dict(stop_reason="refusal"),
    dict(stop_reason="max_tokens"),
    dict(stop_reason="tool_use"),
    dict(content=[]),
    dict(content=[SimpleNamespace(type="text", text="  ")]),
    dict(content=[SimpleNamespace(type="thinking", thinking="...")]),
    dict(content=[SimpleNamespace(type="text")]),  # malformed block
])
def test_llm_errors(kwargs):
    client, _ = fake_client(**kwargs)
    with pytest.raises(LLMError):
        AnthropicLLM(client=client).complete("s", "u", 10)


def test_get_llm_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_llm() is None


def test_get_llm_init_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(AnthropicLLM, "__init__", lambda self, **kw: (_ for _ in ()).throw(RuntimeError()))
    assert get_llm() is None


# --- normalization and identifiers -----------------------------------------------------------

def test_normalize_question():
    assert normalize_question("פוליסה 4,471") == "פוליסה 4471"
    assert normalize_question("٤٤٧١ / ۴۴۷۱") == "4471 / 4471"
    assert normalize_question("1,2345") == "1,2345"


@pytest.mark.parametrize("question", ["פוליסה 4,471", "פוליסה ٤٤٧١", "פוליסה ۴۴۷۱", "פוליסה ٤,٤٧١"])
def test_policy_id_normalized(repo, question):
    assert extract(question, repo).policy_id == "4471"


@pytest.mark.parametrize("question", ["הלקוח מ-2024 רוצה לבטל", "פוליסה משנת 1999", "ב2100"])
def test_years_skipped(repo, question):
    e = extract(question, repo)
    assert e.policy_id is None and e.unknown_policy_id is None


@pytest.mark.parametrize("question", ["A4448", "4448B", "abc4448xyz", "XC-1088", "פוליסה A-4448"])
def test_latin_letter_ids_rejected(repo, question):
    e = extract(question, repo)
    assert (e.policy_id, e.unknown_policy_id, e.customer_id) == (None, None, None)


@pytest.mark.parametrize("prefix", list("לבהומכש"))
def test_hebrew_prefixes_accepted(repo, prefix):
    assert extract(f"מה קורה {prefix}4471?", repo).policy_id == "4471"


def test_customer_ids(repo):
    e = extract("לקוח C-1088 וגם c1072", repo)
    assert e.customer_id == "C-1088" and e.other_customer_ids == ["C-1072"]
    assert e.policy_id is None and e.unknown_policy_id is None  # 1088 is not read as a policy


def test_unknown_policy_id(repo):
    assert extract("פוליסה 9999", repo).unknown_policy_id == "9999"
    e = extract("פוליסה 9999 או 4471", repo)
    assert e.policy_id == "4471" and e.unknown_policy_id == "9999"  # kept: the pipeline asks which is right


def test_several_policy_ids(repo):
    e = extract("4418 ו-4433", repo)
    assert e.policy_id == "4418" and e.other_policy_ids == ["4433"]


def test_name_matches_per_name(repo):
    e = extract("מור שפירא ודנה רוזן", repo)
    assert e.name_matches == {"מור שפירא": {"C-1059", "C-1063"}, "דנה רוזן": {"C-1057"}}


@pytest.mark.parametrize("question,hint", [
    ("ביטוח הרכב שלו", "רכב"),
    ("נזק בדירה", "דירה"),
    ("ולבריאות", "בריאות"),
    ('ביטוח חו"ל', "נסיעות"),
    ("ביטוח חו״ל", "נסיעות"),
    ("לפני הנסיעה", "נסיעות"),
    ("רכב ודירה", None),
    ("ברכבת", None),
    ("שאלה כללית", None),
])
def test_product_hint(repo, question, hint):
    assert extract(question, repo).product_hint == hint


# --- negation and keyword signals ----------------------------------------------------------------

def test_negated_cancel_is_stripped():
    stripped = strip_negations("הוא לא רוצה לבטל, רק שואל על התביעה")
    assert "בטל" not in stripped and ", רק שואל על התביעה" in stripped
    s = keyword_signals("הוא לא רוצה לבטל, רק שואל על התביעה")
    assert not s.cancel and not s.cancel_keyword_raw and s.claim


def test_ela_lebatel_is_not_negation():
    s = keyword_signals("הוא לא שואל על התביעה אלא לבטל")
    assert s.cancel


def test_negation_never_strips_the_object():
    stripped = strip_negations("לא רוצה לבטל. את התביעה צריך לבדוק")
    assert "את התביעה" in stripped and "בטל" not in stripped


@pytest.mark.parametrize("text", ["ולא לבטל את הפוליסה", "אינה מעוניינת בביטול", "אין צריך לבטל", "הוא לא מבקש ביטול"])
def test_other_negations(text):
    assert not keyword_signals(text).cancel


@pytest.mark.parametrize("text", ["כמה דמי ביטול?", "מה תנאי הביטול", "מה עלות ביטול", "מתי ההחזר אחרי ביטול",
                                  "לאחר הביטול", "מה מדיניות הביטול"])
def test_cancel_noun_phrases_excluded(text):
    s = keyword_signals(text)
    assert not s.cancel and s.cancel_keyword_raw


def test_keyword_evidence():
    s = keyword_signals("מתי תשולם התביעה?")
    assert keyword_evidence(PAYMENT, s) and not keyword_evidence(STATUS, s)
    assert keyword_evidence(STATUS, keyword_signals("מה מצב התביעה"))
    assert keyword_evidence(REFUND, keyword_signals("מתי יגיע ההחזר"))
    assert not keyword_evidence(REFUND, keyword_signals("כמה החזר"))
    assert keyword_evidence(DOCS, keyword_signals("אילו מסמכים"))
    assert keyword_evidence(INFO, keyword_signals("פרטי הפוליסה"))
    assert not keyword_evidence(GENERAL, keyword_signals("פרטי הנוהל"))
    owed = keyword_signals("מה מגיע לו")
    assert not any(keyword_evidence(i, owed) for i in Intent)


@pytest.mark.parametrize("text,primary,secondary", [
    ("רוצה לבטל, ומתי תשולם התביעה?", C, [PAYMENT]),
    ("מתי תשולם התביעה ומתי יגיע ההחזר", PAYMENT, []),
    ("מתי יגיע ההחזר", REFUND, []),
    ("אילו מסמכים לתביעה", DOCS, []),  # documents before claim_status
    ("מה מצב התביעה", STATUS, []),
    ("מה מגיע לו", C, []),
    ("כמה החזר יקבל", C, []),
    ("פרטי הפוליסה", INFO, []),
    ("שלום", UNKNOWN, []),
])
def test_keyword_fallback(text, primary, secondary):
    assert keyword_intents(keyword_signals(text)) == (primary, secondary)


# --- LLM intent call -----------------------------------------------------------------------------

def test_llm_call_shape(repo):
    llm = llm_says(STATUS)
    e = extract("מה מצב התביעה של 4425", repo, llm)
    assert (e.primary_intent, e.via, e.llm_failed) == (STATUS, "llm", False)
    call = llm.calls[0]
    assert call["system"] == INTENT_SYSTEM_PROMPT and call["max_tokens"] == 1024
    assert call["json_schema"] == INTENT_SCHEMA
    assert INTENT_SCHEMA["required"] == ["intents", "mentions_regulator"]
    assert INTENT_SCHEMA["additionalProperties"] is False
    assert set(INTENT_SCHEMA["properties"]["intents"]["items"]["enum"]) == {i.value for i in Intent}


@pytest.mark.parametrize("llm", [
    FakeLLM("not json"),
    FakeLLM('{"intents": [], "mentions_regulator": false}'),
    FakeLLM('{"intents": ["cancel"], "mentions_regulator": false}'),
    FakeLLM('{"intents": ["cancellation"], "mentions_regulator": "no"}'),
    FakeLLM('{"intents": ["cancellation"]}'),
    FakeLLM('{"intents": ["cancellation"], "mentions_regulator": false, "x": 1}'),
    FakeLLM('{"intents": "cancellation", "mentions_regulator": false}'),
    FakeLLM('["cancellation"]'),
    FakeLLM(error=LLMError("timeout")),
])
def test_invalid_llm_output_falls_back(repo, llm):
    e = extract("לקוח 4471 רוצה לבטל", repo, llm)
    assert (e.via, e.llm_failed, e.primary_intent) == ("keywords", True, C)


def test_no_llm_uses_keywords(repo):
    e = extract("לקוח 4471 רוצה לבטל", repo, None)
    assert (e.via, e.llm_failed, e.primary_intent) == ("keywords", False, C)


def test_cancel_keyword_raw(repo):
    assert extract("כמה דמי ביטול לפוליסה 4471", repo).cancel_keyword_raw
    assert not extract("לא רוצה לבטל, מה מצב התביעה", repo).cancel_keyword_raw


# --- pick_llm_intents ------------------------------------------------------------------------------

def sig(**kw):
    return Signals(**kw)


def test_pick_drops_duplicates_and_unknown():
    p = pick_llm_intents([UNKNOWN, STATUS, STATUS], sig(claim=True), identified=True)
    assert (p.primary, p.secondary) == (STATUS, [])
    assert pick_llm_intents([UNKNOWN], sig(), identified=False).primary is UNKNOWN


def test_pick_not_identified_first_is_primary():
    p = pick_llm_intents([GENERAL, C], sig(cancel=True), identified=False)
    assert (p.primary, p.secondary) == (GENERAL, [C])
    assert not p.possible_cancellation and not p.keyword_cancel_unlisted


def test_pick_not_identified_later_needs_evidence():
    p = pick_llm_intents([GENERAL, C, DOCS], sig(), identified=False)
    assert (p.primary, p.secondary) == (GENERAL, [])


def test_pick_not_identified_keywords_never_reorder():
    p = pick_llm_intents([STATUS], sig(cancel=True, claim=True), identified=False)
    assert (p.primary, p.secondary, p.keyword_cancel_unlisted) == (STATUS, [], False)


def test_pick_identified_cancellation_first():
    p = pick_llm_intents([C, STATUS], sig(claim=True), identified=True)
    assert (p.primary, p.secondary) == (C, [STATUS])


def test_pick_identified_cancellation_later_with_signal():
    p = pick_llm_intents([STATUS, C], sig(cancel=True, claim=True), identified=True)
    assert (p.primary, p.secondary) == (C, [STATUS])


def test_pick_identified_cancellation_later_without_signal():
    p = pick_llm_intents([STATUS, C], sig(claim=True), identified=True)
    assert (p.primary, p.secondary, p.possible_cancellation) == (STATUS, [], True)


def test_pick_identified_cancellation_unlisted_with_signal():
    p = pick_llm_intents([STATUS], sig(cancel=True, claim=True), identified=True)
    assert (p.primary, p.secondary, p.keyword_cancel_unlisted) == (STATUS, [], True)
    assert not p.possible_cancellation


def test_pick_identified_secondary_needs_evidence():
    p = pick_llm_intents([C, STATUS, DOCS], sig(documents=True), identified=True)
    assert (p.primary, p.secondary) == (C, [DOCS])


def test_identified_via_customer_or_name(repo):
    question = "רוצה לבטל, ושאל על התביעה"
    assert extract("C-1088 " + question, repo, llm_says(STATUS)).keyword_cancel_unlisted
    assert extract("רונן אביטל " + question, repo, llm_says(STATUS)).keyword_cancel_unlisted
    assert not extract("C-9999 " + question, repo, llm_says(STATUS)).keyword_cancel_unlisted
