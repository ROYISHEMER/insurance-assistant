"""Turn a rep's free-text question into identifiers (always by code) and intents (LLM or keywords)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .guardrails import mentions_regulator
from .llm import LLMError
from .models import Intent
from .repository import PolicyRepository

# --- normalization ---------------------------------------------------------------

_DIGITS = str.maketrans(
    {**{chr(0x0660 + i): str(i) for i in range(10)}, **{chr(0x06F0 + i): str(i) for i in range(10)}}
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize_question(question: str) -> str:
    """Arabic-Indic and Persian digits -> 0-9; drop a thousands comma between digits."""
    return _THOUSANDS.sub("", question.translate(_DIGITS))


# --- identifiers -------------------------------------------------------------------

_CUSTOMER_ID = re.compile(r"(?<![A-Za-z0-9])[Cc]-?(\d{4})(?![\dA-Za-z])")
# A Latin letter touching the number (A4448, 4448B, XC-1088) means it is not a policy id;
# Hebrew prefix letters glued before it (ל4471, ב4471, ...) are fine.
# Part of a hyphenated digit group (a phone number like 050-4471-123) is not a policy id either.
_POLICY_ID = re.compile(r"(?<![\dA-Za-z])(?<![A-Za-z]-)(?<!\d-)\d{4}(?![\dA-Za-z])(?!-\d)")
_YEARS = range(1900, 2101)
_PREFIX = r"(?<![א-ת])ו?ש?[הבלמכ]?"
_PRODUCTS = {
    "רכב": re.compile(_PREFIX + r"רכב(?![א-ת])"),
    "דירה": re.compile(_PREFIX + r"דירה(?![א-ת])"),
    "בריאות": re.compile(_PREFIX + r"בריאות(?![א-ת])"),
    "נסיעות": re.compile(_PREFIX + r"(?:נסיעות|נסיעה|חו\"ל|חו״ל)(?![א-ת])"),
}


@dataclass
class Identifiers:
    policy_id: str | None = None
    other_policy_ids: list[str] = field(default_factory=list)
    unknown_policy_id: str | None = None
    customer_id: str | None = None
    other_customer_ids: list[str] = field(default_factory=list)
    name_matches: dict[str, set[str]] = field(default_factory=dict)
    product_hint: str | None = None


_AMOUNT_AFTER = re.compile(r"\s*(?:₪|ש\"ח|ש״ח|שח(?![א-ת])|שקל|NIS(?![A-Za-z]))")
_AMOUNT_BEFORE = re.compile(r"₪\s*$")


def _is_amount(text: str, match: re.Match) -> bool:
    """"5000 ₪" / "₪5000" is money, not a policy id."""
    return bool(_AMOUNT_AFTER.match(text, match.end()) or _AMOUNT_BEFORE.search(text[:match.start()]))


def name_appears_in(name: str, text: str) -> bool:
    """The full name, not followed by a Hebrew letter, preceded by a non-letter or up to two prefix letters."""
    if not name:
        return False
    pattern = r"(?<![^\W\d_])[ושהבלמכ]{0,2}" + re.escape(name) + r"(?![א-ת])"
    return re.search(pattern, text) is not None


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def extract_identifiers(text: str, repo: PolicyRepository) -> Identifiers:
    ids = Identifiers()
    customers = _unique([f"C-{m}" for m in _CUSTOMER_ID.findall(text)])
    if customers:
        ids.customer_id, ids.other_customer_ids = customers[0], customers[1:]
    without_customers = _CUSTOMER_ID.sub(" ", text)

    known, unknown = [], []
    candidates = [m.group() for m in _POLICY_ID.finditer(without_customers) if not _is_amount(without_customers, m)]
    for candidate in _unique(candidates):
        if int(candidate) in _YEARS:
            continue
        (known if repo.get(candidate) is not None else unknown).append(candidate)
    if known:
        ids.policy_id, ids.other_policy_ids = known[0], known[1:]
    if unknown:
        # Kept even next to a known id: the pipeline asks the rep which number is the policy.
        ids.unknown_policy_id = unknown[0]

    # Whole-name matches only, whatever the repository's own matching does.
    for policy in repo.find_names_in(text):
        if name_appears_in(policy.customer_name, text):
            ids.name_matches.setdefault(policy.customer_name, set()).add(policy.customer_id)

    products = [name for name, pattern in _PRODUCTS.items() if pattern.search(text)]
    if len(products) == 1:
        ids.product_hint = products[0]
    return ids


# --- keyword signals ------------------------------------------------------------------

_NEGATED_VERB = re.compile(
    r"(?<![א-ת])ו?(?:לא|אינו|אינה|אין)\s+"
    r"(?:רוצה|רוצים|מעוניין|מעוניינת|מבקש|מבקשת|צריך|צריכה)\s+[^\s.,;?!]+"
)
_NEGATED_CANCEL = re.compile(r"(?<![א-ת])ו?לא\s+לבטל(?![א-ת])")
_CANCEL_CONTEXT = re.compile(r"(?:אחרי|לאחר|דמי|תנאי|מדיניות|עלות)\s+ה?ביטול")
_WHEN = ("מתי", "תוך כמה", "כמה זמן")
_PAID = ("ישולם", "תשולם", "שולם", "ישלמו", "משלמים", "תשלום", "הכסף")


_ELLA = re.compile(r"(?<![א-ת])אלא(?![א-ת])")
_SENTENCE_END = re.compile(r"[.;?!\n]")


def strip_negations(text: str) -> str:
    """Remove only the negated verb phrase; never crosses punctuation, never strips the object."""
    return _NEGATED_CANCEL.sub(" ", _NEGATED_VERB.sub(" ", text))


def _negated_cancellations(text: str) -> list[re.Match]:
    verb_phrases = [m for m in _NEGATED_VERB.finditer(text) if _has_cancel_word(m.group())]
    return verb_phrases + list(_NEGATED_CANCEL.finditer(text))  # positions refer to the original text


def _corrected_by_ella(text: str, negations: list[re.Match]) -> bool:
    """"לא רוצה לבטל את התביעה אלא את הפוליסה": "אלא" later in the same sentence corrects the object."""
    for m in negations:
        end = _SENTENCE_END.search(text, m.end())
        if _ELLA.search(text, m.end(), end.start() if end else len(text)):
            return True
    return False


def _has_cancel_word(text: str) -> bool:
    return "בטל" in text or "ביטול" in text


@dataclass
class Signals:
    claim: bool = False
    when: bool = False
    paid: bool = False
    refund: bool = False
    documents: bool = False
    details: bool = False
    owed: bool = False
    cancel: bool = False
    cancel_keyword_raw: bool = False
    negated_cancel: bool = False

    @property
    def payment(self) -> bool:
        return self.when or self.paid


def keyword_signals(text: str) -> Signals:
    stripped = strip_negations(text)
    negations = _negated_cancellations(text)
    corrected = _corrected_by_ella(text, negations)
    return Signals(
        claim="תביע" in stripped,
        when=any(w in stripped for w in _WHEN),
        paid=any(w in stripped for w in _PAID),
        refund="החזר" in stripped,
        documents="מסמכ" in stripped,
        details="פרטי" in stripped or "סטטוס" in stripped,
        owed="מגיע לו" in stripped or "מגיע לה" in stripped,
        cancel=corrected or _has_cancel_word(_CANCEL_CONTEXT.sub(" ", stripped)),
        cancel_keyword_raw=corrected or _has_cancel_word(stripped),
        negated_cancel=bool(negations),
    )


def keyword_evidence(intent: Intent, s: Signals) -> bool:
    """Whether the keywords support an intent. "מגיע לו" is evidence of nothing."""
    return {
        Intent.CANCELLATION: s.cancel,
        Intent.CLAIM_PAYMENT_DATE: s.claim and s.payment,
        Intent.CLAIM_STATUS: s.claim and not s.payment,
        Intent.REFUND_TIMING: s.refund and s.when,
        Intent.CLAIM_DOCUMENTS: s.documents,
        Intent.POLICY_INFO: s.details,
    }.get(intent, False)


_FALLBACK_ORDER = (Intent.CLAIM_PAYMENT_DATE, Intent.REFUND_TIMING, Intent.CLAIM_DOCUMENTS, Intent.CLAIM_STATUS)


def keyword_intents(s: Signals) -> tuple[Intent, list[Intent]]:
    other = next((i for i in _FALLBACK_ORDER if keyword_evidence(i, s)), None)
    if s.cancel:
        return Intent.CANCELLATION, [other] if other else []
    if other:
        return other, []
    # "החזר" / "מגיע לו" mean cancellation only if the rep did not just say the customer is NOT cancelling.
    if (s.owed or s.refund) and not s.negated_cancel:
        return Intent.CANCELLATION, []
    if s.details:
        return Intent.POLICY_INFO, []
    return Intent.UNKNOWN, []


# --- LLM intents ----------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You classify questions that insurance call-center reps type (in Hebrew) about a customer.
Return JSON only: {"intents": [<one or more values>], "mentions_regulator": <bool>}.
List every separate request the customer makes, in the order asked; usually there is exactly one.

Intents:
- cancellation: the customer wants to cancel, or asks what they get / pay if they cancel.
  General rule: any wish to end, stop, leave or not continue the insurance, in any wording, is cancellation.
- claim_status: status of an existing claim.
- claim_payment_date: when an existing claim will be paid.
- claim_documents: which documents are needed to file a claim.
- refund_timing: when / how a refund is transferred.
- policy_info: details of a specific customer's policy (status, premium, dates, ...). Only when the question
  names a specific policy or customer (policy number, customer number or customer name).
- general_policy: a general question about company procedures, not about a specific customer's numbers.
  Questions about what the insurance covers, or how a procedure works, are general_policy.
  General rule: asking how much something costs or how it is calculated, without asking to do it,
  is general_policy.
- unknown: anything else.

Classify by what the customer actually wants now. Negated requests do not count:
"הוא לא רוצה לבטל, רק שואל על התביעה" is claim_status, not cancellation.
A mention that only sets context is not a request: "מתי מקבלים את ההחזר אחרי ביטול?" is refund_timing only.
mentions_regulator is true if the customer mentions complaining to, or contacting, the insurance
regulator (רשות שוק ההון / הממונה / המפקח על הביטוח), even indirectly."""

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intents": {"type": "array", "items": {"type": "string", "enum": [i.value for i in Intent]}},
        "mentions_regulator": {"type": "boolean"},
    },
    "required": ["intents", "mentions_regulator"],
    "additionalProperties": False,
}


def parse_intent_response(text: str) -> tuple[list[Intent], bool]:
    """Raises ValueError on anything that does not match INTENT_SCHEMA exactly."""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"intents", "mentions_regulator"}:
        raise ValueError("unexpected keys")
    intents, regulator = data["intents"], data["mentions_regulator"]
    if not isinstance(intents, list) or not intents or not isinstance(regulator, bool):
        raise ValueError("bad types")
    if not all(isinstance(i, str) for i in intents):
        raise ValueError("non-string intent")
    return [Intent(i) for i in intents], regulator


@dataclass
class IntentPick:
    primary: Intent
    secondary: list[Intent] = field(default_factory=list)
    possible_cancellation: bool = False
    keyword_cancel_unlisted: bool = False


def pick_llm_intents(llm_intents: list[Intent], signals: Signals, identified: bool) -> IntentPick:
    intents = [i for i in _unique(llm_intents) if i is not Intent.UNKNOWN]
    if not intents:
        return IntentPick(Intent.UNKNOWN, keyword_cancel_unlisted=identified and signals.cancel)
    first, rest = intents[0], intents[1:]

    def with_evidence(candidates: list[Intent]) -> list[Intent]:
        return [i for i in candidates if keyword_evidence(i, signals)]

    if not identified:
        # The LLM's first intent is primary; keywords never reorder it.
        return IntentPick(first, with_evidence(rest))

    listed = Intent.CANCELLATION in intents
    if first is Intent.CANCELLATION:
        return IntentPick(first, with_evidence(rest))
    others = [i for i in intents if i is not Intent.CANCELLATION]
    if listed and signals.cancel:
        return IntentPick(Intent.CANCELLATION, with_evidence(others))
    if listed:
        # Listed later without keyword support: escalate only on open/unknown claims (pipeline).
        return IntentPick(first, with_evidence(others[1:]), possible_cancellation=True)
    # Not listed: a cancel keyword alone never creates a cancellation answer.
    return IntentPick(first, with_evidence(rest), keyword_cancel_unlisted=signals.cancel)


# --- extraction ---------------------------------------------------------------------------

@dataclass
class Extraction:
    primary_intent: Intent
    secondary_intents: list[Intent] = field(default_factory=list)
    policy_id: str | None = None
    other_policy_ids: list[str] = field(default_factory=list)
    unknown_policy_id: str | None = None
    customer_id: str | None = None
    other_customer_ids: list[str] = field(default_factory=list)
    name_matches: dict[str, set[str]] = field(default_factory=dict)
    product_hint: str | None = None
    regulator: bool = False
    via: str = "keywords"  # "llm" or "keywords"
    llm_failed: bool = False
    cancel_keyword_raw: bool = False
    possible_cancellation: bool = False
    keyword_cancel_unlisted: bool = False


def _is_identified(ids: Identifiers, repo: PolicyRepository) -> bool:
    return bool(ids.policy_id or ids.name_matches or (ids.customer_id and repo.by_customer(ids.customer_id)))


def extract(question: str, repo: PolicyRepository, llm=None) -> Extraction:
    text = normalize_question(question)
    ids = extract_identifiers(text, repo)
    signals = keyword_signals(text)

    def build(pick: IntentPick, **flags) -> Extraction:
        return Extraction(
            primary_intent=pick.primary,
            secondary_intents=pick.secondary,
            policy_id=ids.policy_id,
            other_policy_ids=ids.other_policy_ids,
            unknown_policy_id=ids.unknown_policy_id,
            customer_id=ids.customer_id,
            other_customer_ids=ids.other_customer_ids,
            name_matches=ids.name_matches,
            product_hint=ids.product_hint,
            cancel_keyword_raw=signals.cancel_keyword_raw,
            possible_cancellation=pick.possible_cancellation,
            keyword_cancel_unlisted=pick.keyword_cancel_unlisted,
            **flags,
        )

    def by_keywords(**flags) -> Extraction:
        return build(IntentPick(*keyword_intents(signals)), via="keywords", **flags)

    # The regulator check runs first and by code: the LLM is not called at all.
    if mentions_regulator(text):
        return by_keywords(regulator=True)
    if llm is None:
        return by_keywords()
    try:
        response = llm.complete(INTENT_SYSTEM_PROMPT, question, max_tokens=1024, json_schema=INTENT_SCHEMA)
        llm_intents, llm_regulator = parse_intent_response(response)
    except (LLMError, ValueError, TypeError):
        return by_keywords(llm_failed=True)
    pick = pick_llm_intents(llm_intents, signals, _is_identified(ids, repo))
    return build(pick, via="llm", regulator=llm_regulator)
