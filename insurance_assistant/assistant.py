"""The pipeline: extraction -> policy resolution -> safety checks -> deterministic rules -> template."""
from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from .extract import Extraction, extract, normalize_question
from .grounding import currency_amounts, numbers_in, ungrounded_numbers, without_thousands
from .guardrails import mentions_regulator
from .llm import LLMError
from .models import DEFAULT_AS_OF, Decision, Fact, Intent, Outcome, Policy
from .policy_docs import PROJECT_ROOT, load_policy_docs, split_sections
from .repository import PolicyRepository, RepositoryUnavailable, ValidatedPolicyRepository
from .rules.cancellation import claims_escalation, evaluate_cancellation, has_open_or_unknown_claims
from .rules.claims import claim_documents, claim_payment_date, claim_status, policy_info, refund_timing
from .rules.common import CANCEL, EXCEPTIONS, REFUNDS, data_source, start_decision
from .textnorm import normalize_unicode

GENERAL_SYSTEM_PROMPT = """\
You answer insurance call-center reps' general questions about company procedures.
Answer in Hebrew, briefly, ONLY from the procedure documents below. Documents marked as overriding
(04_חריגים.md) take precedence over the others.
Cite every statement as <file name> §<section>, e.g. (01_תנאי_ביטול.md §3).
Do not calculate amounts or dates for a specific customer and do not add numbers that are not in the documents.
If the answer is not in the documents, reply exactly: לא מופיע במדיניות
Keep the answer under 120 words.
Do not repeat policy or customer numbers from the question."""
NOT_IN_POLICY = "לא מופיע במדיניות"
GENERAL_TIMEOUT = 8.0
# Thinking tokens count toward max_tokens (SDK 2.3.0 docs); 1024 truncated a live Hebrew answer.
GENERAL_MAX_TOKENS = 4096

LABELS = {
    Outcome.ANSWER: "תשובה",
    Outcome.ANSWER_WITH_GAPS: "תשובה חלקית – חסרים נתונים",
    Outcome.ESCALATE: "העברה למנהל משמרת",
    Outcome.CANNOT_ANSWER: "אין תשובה",
}
BANNER_LLM_UNAVAILABLE = "מצב מוגבל: שירות השפה לא זמין כרגע – לא ניתן לענות על שאלה כללית"
BANNER_KEYWORDS = "מצב מוגבל: זיהוי לפי מילות מפתח בלבד – ודא שהבנת את בקשת הלקוח"
INTENT_NAMES = {
    Intent.CANCELLATION: "ביטול פוליסה",
    Intent.CLAIM_STATUS: "סטטוס תביעה",
    Intent.CLAIM_PAYMENT_DATE: "מועד תשלום תביעה",
    Intent.CLAIM_DOCUMENTS: "מסמכים לתביעה",
    Intent.REFUND_TIMING: "מועד העברת החזר",
    Intent.POLICY_INFO: "פרטי פוליסה",
    Intent.GENERAL_POLICY: "שאלה כללית על הנהלים",
    Intent.UNKNOWN: "בקשה לא מזוהה",
}
# The number is taken as a whole token, decimal part included: "לקוח 1088.50" is the number 1088.5, not id 1088.
_ID_CONTEXT = re.compile(
    r"(?:(?<![^\W\d_])[ושהבלמכ]*(?:פוליסה|פוליסת|לקוח) ?|(?<![A-Za-z])C-)(\d+(?:\.\d+)?)(?![\d.]?\d)"
)
NEEDS_POLICY = (Intent.CANCELLATION, Intent.CLAIM_STATUS, Intent.CLAIM_PAYMENT_DATE, Intent.POLICY_INFO)
_CITATION = re.compile(r"([^\s()\[\]§,:]+\.md)\s*§\s*(\d+[א-ת]?|[א-ת]{1,2})(?![א-ת])")


@dataclass
class Response:
    decision: Decision
    text: str
    latency_ms: int
    path: str
    banner: str | None = None

    @property
    def degraded(self) -> bool:
        return self.banner is not None


# --- one read per policy per question --------------------------------------------------------

class _OneReadPerCall:
    """Within one answer() call every policy is read once, and the same Policy object is used everywhere
    (resolution, claims safety, rules, safety net). A second read that disagrees means the source changed
    under us: RepositoryUnavailable, so no amounts are computed from mixed data."""

    def __init__(self, repo: PolicyRepository):
        self._repo = repo
        self.source_name = repo.source_name
        self._policies: dict[str, Policy | None] = {}
        self._customers: dict[str, list[Policy]] = {}

    def _keep(self, policy: Policy) -> Policy:
        seen = self._policies.get(policy.policy_id)
        if seen is None:  # first read (or an earlier "not found", now superseded by a real record)
            self._policies[policy.policy_id] = policy
            return policy
        if seen != policy and not (seen.conflicting_records and policy.conflicting_records):
            raise RepositoryUnavailable(f"policy {policy.policy_id} changed between two reads in one question")
        return seen if not policy.conflicting_records else policy

    def get(self, policy_id: str) -> Policy | None:
        key = str(policy_id).strip()
        if key not in self._policies:
            policy = self._repo.get(key)
            self._policies[key] = None if policy is None else self._keep(policy)
        return self._policies[key]

    def by_customer(self, customer_id: str) -> list[Policy]:
        key = str(customer_id).strip()
        if key not in self._customers:
            self._customers[key] = [self._keep(p) for p in self._repo.by_customer(key)]
        return self._customers[key]

    def find_names_in(self, text: str) -> list[Policy]:
        return [self._keep(p) for p in self._repo.find_names_in(text)]


# --- small builders ---------------------------------------------------------------------

def _cannot(intent: Intent, code: str, headline: str, missing: str | None = None,
            policy: Policy | None = None, policy_id: str | None = None) -> Decision:
    if policy is not None:
        d = start_decision(policy, intent, Outcome.CANNOT_ANSWER, code, headline)
    else:
        d = Decision(Outcome.CANNOT_ANSWER, intent, code, headline, policy_id=policy_id)
    if missing:
        d.missing.append(missing)
    return d


def _list_policies(d: Decision, policies: list[Policy]) -> Decision:
    for p in policies:
        d.facts.append(Fact(f"פוליסה {p.policy_id}", f"{p.product}, {p.status}, לקוח {p.customer_id} ({p.customer_name})",
                            data_source(p)))
    return d


# --- step 2: policy resolution ------------------------------------------------------------

def _resolve(ex: Extraction, repo: PolicyRepository) -> tuple[Policy | None, Decision | None]:
    intent = ex.primary_intent
    known = [p for p in (repo.get(pid) for pid in [ex.policy_id, *ex.other_policy_ids] if pid) if p]
    customer_ids = [c for c in [ex.customer_id, *ex.other_customer_ids] if c]
    id_customers = {p.customer_id for p in known} | set(customer_ids)
    names = ex.name_matches

    # a. a policy id and a customer id that belong to different customers
    if known and any(c != known[0].customer_id for c in customer_ids):
        d = _cannot(intent, "CONFLICTING_IDENTIFIERS", "מספר הפוליסה ומספר הלקוח אינם שייכים לאותו לקוח",
                    "אישור מהלקוח: איזה מזהה נכון")
        return None, _list_policies(d, known)
    # b. every matched name must belong to one of the ids' customers
    if names and id_customers:
        stray = [n for n, customers in names.items() if not customers & id_customers]
        if stray:
            d = _cannot(intent, "CONFLICTING_IDENTIFIERS",
                        f"השם {', '.join(stray)} אינו תואם את המזהה שנמסר",
                        "אישור מהלקוח: איזה שם ומזהה נכונים")
            return None, _list_policies(d, known)
    # c. more than one customer
    if len(id_customers) > 1:
        d = _cannot(intent, "MULTIPLE_CUSTOMERS", "השאלה מזכירה כמה לקוחות שונים",
                    f"לקוח אחד בכל שאלה; זוהו: {', '.join(sorted(id_customers))}")
        return None, d
    no_ids = not (known or customer_ids or ex.unknown_policy_id)
    if no_ids and len(names) > 1 and len(set().union(*names.values())) > 1:
        d = _cannot(intent, "MULTIPLE_CUSTOMERS", "השאלה מזכירה כמה לקוחות שונים",
                    f"לקוח אחד בכל שאלה; זוהו: {', '.join(names)}")
        return None, d

    # d. resolve to one policy
    if ex.unknown_policy_id:
        # Also next to a known id: never guess which of the two numbers is the policy.
        missing = "מספר פוליסה תקין"
        if known:
            missing = (f"יש לוודא מול הלקוח איזה מספר הוא מספר הפוליסה: {ex.unknown_policy_id} לא נמצא, "
                       f"{known[0].policy_id} קיים")
        return None, _cannot(intent, "POLICY_NOT_FOUND", f"פוליסה {ex.unknown_policy_id} לא נמצאה", missing)
    if len(known) > 1:
        # Never compute amounts for one of two policies: the first may be the one the customer excluded
        # ("C-1072 לא את 4418; הוא רוצה לבטל את 4433").
        d = _cannot(intent, "MULTIPLE_POLICIES", "בשאלה כמה מספרי פוליסה – יש לברר לאיזו פוליסה הלקוח מתכוון",
                    "מספר הפוליסה שהלקוח מתכוון אליה")
        return None, _list_policies(d, known)
    if known:
        return known[0], None
    if customer_ids:
        candidates = repo.by_customer(customer_ids[0])
        if not candidates:
            return None, _cannot(intent, "CUSTOMER_NOT_FOUND", f"לא נמצאו פוליסות ללקוח {customer_ids[0]}",
                                 "מספר לקוח תקין")
    elif names:
        customers = sorted(set().union(*names.values()))
        candidates = [p for c in customers for p in repo.by_customer(c)]
        if len(customers) > 1:
            d = _cannot(intent, "AMBIGUOUS_CUSTOMER", f"השם {', '.join(names)} תואם כמה לקוחות",
                        "מספר פוליסה או מספר לקוח")
            return None, _list_policies(d, candidates)
    else:
        return None, None

    # The same policy twice (duplicate records) is one candidate; step 3 then stops it as CONFLICTING_RECORDS.
    candidates = list({p.policy_id: p for p in reversed(candidates)}.values())[::-1]

    # Resolved from a customer id or a name: EVERY product the rep mentioned must exist for this customer.
    owned = {p.product for p in candidates}
    foreign = [p for p in ex.products_mentioned if p not in owned]
    if foreign:
        products = ", ".join(f"{p.product} ({p.policy_id})" for p in candidates)
        d = _cannot(intent, "PRODUCT_MISMATCH", f"ללקוח אין פוליסת {', '.join(foreign)}; הפוליסות שלו: {products}",
                    f"אישור מהלקוח: על איזו פוליסה מדובר (נאמר: {', '.join(ex.products_mentioned)})")
        return None, _list_policies(d, candidates)
    if ex.product_hint:
        candidates = [p for p in candidates if p.product == ex.product_hint]
    if len(candidates) > 1:
        d = _cannot(intent, "MULTIPLE_POLICIES", "ללקוח כמה פוליסות – יש לבחור אחת", "מספר הפוליסה הרלוונטית")
        return None, _list_policies(d, candidates)
    return candidates[0], None


# --- general questions (the only LLM-written answer) ------------------------------------------

@lru_cache(maxsize=8)
def _docs(directory: str) -> dict[str, str]:
    return load_policy_docs(directory)


def _resolve_dir(policies_dir: str | Path) -> str:
    folder = Path(policies_dir)
    return str(folder if folder.is_absolute() else PROJECT_ROOT / folder)


def general_answer(question: str, llm, policies_dir: str | Path, policy_id: str | None,
                   id_numbers: frozenset[str] = frozenset()) -> Decision:
    docs = _docs(_resolve_dir(policies_dir))
    intent = Intent.GENERAL_POLICY
    corpus = "\n\n".join(f"=== {name} ===\n{text}" for name, text in docs.items())
    try:
        answer_text = llm.complete(GENERAL_SYSTEM_PROMPT + "\n\n" + corpus, question,
                                   max_tokens=GENERAL_MAX_TOKENS, timeout=GENERAL_TIMEOUT)
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)  # never into the decision or the text
        return _cannot(intent, "LLM_UNAVAILABLE", "שירות השפה אינו זמין כרגע",
                       "נסה שוב בעוד רגע, או בדוק ישירות במסמכי המדיניות", policy_id=policy_id)

    answer_text = normalize_unicode(answer_text)
    if NOT_IN_POLICY in answer_text:
        return _cannot(intent, "NOT_IN_POLICY", "התשובה אינה מופיעה במסמכי המדיניות", policy_id=policy_id)

    sections = {name: split_sections(text) for name, text in docs.items()}
    citations = list(dict.fromkeys(_CITATION.findall(answer_text)))
    bad_citations = [f"{f} §{s}" for f, s in citations if s not in sections.get(f, {})]
    without_citations = _CITATION.sub(" ", answer_text)
    # Numbers are grounded in the procedures only – never in the question (a number the rep typed is not
    # policy). The policy / customer ids found in the question are the only exception.
    # A policy / customer id from the question may appear only in identifier context ("פוליסה 4471",
    # "ללקוח C-1088"); anywhere else it is just a number and must be in the procedures.
    in_context = _ID_CONTEXT.sub(lambda m: " " if m.group(1) in id_numbers else m.group(0),
                                 without_thousands(without_citations))
    stray_numbers = ungrounded_numbers(in_context, "\n".join(docs.values()))
    # A currency-marked amount must be in the procedures themselves; the identifier exception never applies.
    doc_numbers = numbers_in("\n".join(docs.values()))
    stray_numbers = sorted(set(stray_numbers) | (currency_amounts(without_citations) - doc_numbers))
    if not citations or bad_citations or stray_numbers:
        d = _cannot(intent, "UNGROUNDED", "תשובת מודל השפה לא עברה בדיקת עיגון במסמכים", policy_id=policy_id)
        if not citations:
            d.warnings.append("התשובה לא ציטטה אף סעיף")
        if bad_citations:
            d.warnings.append(f"ציטוט לסעיף שאינו קיים: {', '.join(bad_citations)}")
        if stray_numbers:
            d.warnings.append(f"מספרים שאינם מופיעים במסמכים: {', '.join(stray_numbers)}")
        return d

    cited = [f"{f} §{s}" for f, s in citations]
    d = Decision(Outcome.ANSWER, intent, "GENERAL_POLICY", "תשובה כללית מתוך מסמכי המדיניות", policy_id=policy_id)
    d.facts.append(Fact("תשובת מודל שפה – יש לאמת מול הטקסט המצוטט", answer_text.strip(), ", ".join(cited)))
    refs = list(citations)
    if any(f == REFUNDS and s.startswith("2") for f, s in refs) and (REFUNDS, "3") not in refs:
        refs.append((REFUNDS, "3"))
    for f, s in refs:
        text = sections.get(f, {}).get(s)
        if text is not None:
            d.facts.append(Fact(f"{f} §{s}", text, f"{f} §{s}"))
    if any(f in (CANCEL, REFUNDS) for f, _ in refs) and docs.get(EXCEPTIONS):
        d.facts.append(Fact("חריגים הגוברים על הנהלים הבסיסיים", docs[EXCEPTIONS].strip(), EXCEPTIONS))
    return d


# --- rendering -----------------------------------------------------------------------------------

def render(d: Decision) -> str:
    head = f"[{LABELS[d.outcome]}] {d.headline}"
    if d.policy_id:
        head += f" (פוליסה {d.policy_id})"
    lines = [head]
    sections = [
        ("עובדות:", [f"- {f.label}: {f.value} [{f.source}]" for f in d.facts]),
        ("אזהרות:", [f"- {w}" for w in d.warnings]),
        ("חסר:", [f"- {m}" for m in d.missing]),
        ("הנחות:", [f"- {a}" for a in d.assumptions]),
        ("מה לומר ללקוח:", [f"- {s}" for s in d.say_to_customer]),
    ]
    for title, items in sections:
        if items:
            lines += ["", title, *items]
    return "\n".join(lines)


# --- the pipeline ---------------------------------------------------------------------------------

def _regulator(intent: Intent, policy_id: str | None) -> Decision:
    d = Decision(Outcome.ESCALATE, intent, "REGULATOR",
                 "הלקוח הזכיר פנייה לרגולטור – העברה מיידית למנהל משמרת", policy_id=policy_id)
    d.facts.append(Fact("פנייה שבה הלקוח מזכיר תלונה לרשות שוק ההון",
                        "מועברת מיידית למנהל המשמרת, ללא מתן תשובה עניינית על ידי הנציג", f"{EXCEPTIONS} §ה"))
    d.warnings.append(f"אין לתת תשובה עניינית ({EXCEPTIONS} §ה)")
    d.say_to_customer.append("הפנייה מועברת למנהל המשמרת, שיחזור אליך.")
    return d


def _id_numbers(ex: Extraction) -> frozenset[str]:
    ids = [ex.policy_id, *ex.other_policy_ids, ex.unknown_policy_id, ex.customer_id, *ex.other_customer_ids]
    return frozenset(i.removeprefix("C-") for i in ids if i)


def _dispatch(intent: Intent, policy: Policy | None, ex: Extraction, question: str, repo, llm,
              as_of: date, policies_dir) -> tuple[Decision, str]:
    pid = policy.policy_id if policy else None
    if intent is Intent.CANCELLATION:
        return evaluate_cancellation(policy, as_of, repo), "rules:cancellation"
    if intent is Intent.CLAIM_PAYMENT_DATE:
        return claim_payment_date(policy), "rules:claims"
    if intent is Intent.CLAIM_STATUS:
        return claim_status(policy), "rules:claims"
    if intent is Intent.POLICY_INFO:
        return policy_info(policy), "rules:policy_info"
    if intent is Intent.CLAIM_DOCUMENTS:
        return claim_documents(policy.product if policy else ex.product_hint, pid), "rules:claims"
    if intent is Intent.REFUND_TIMING:
        return refund_timing(pid), "rules:refunds"
    if intent is Intent.GENERAL_POLICY and llm is not None:
        return general_answer(question, llm, policies_dir, pid, _id_numbers(ex)), "llm:general → grounding"
    if intent is Intent.UNKNOWN and policy is not None:
        return _cannot(intent, "UNCLEAR_REQUEST",
                       f"לא זוהתה הבקשה לגבי פוליסה {pid} – בקש מהלקוח לפרט", policy=policy), "unclear"
    return _cannot(intent, "OUT_OF_SCOPE", "השאלה מחוץ לתחום הכלי", policy_id=pid), "out_of_scope"


def _decide(question: str, ex: Extraction, repo, llm, as_of: date, policies_dir) -> tuple[Decision, list[str]]:
    path: list[str] = []
    if ex.regulator:
        return _regulator(ex.primary_intent, ex.policy_id), ["guardrail:regulator"]

    policy, problem = _resolve(ex, repo)
    if problem is not None:
        return problem, [f"policy:{problem.code}"]
    path.append(f"policy:{policy.policy_id}" if policy else "policy:none")

    if policy is not None and policy.conflicting_records:
        d = _cannot(ex.primary_intent, "CONFLICTING_RECORDS", "לפוליסה כמה רשומות סותרות במקור הנתונים",
                    "רשומה אחת ועדכנית של הפוליסה – יש לבדוק במערכת", policy=policy)
        return d, path

    if policy is not None and has_open_or_unknown_claims(policy) and (
        ex.possible_cancellation
        or ex.keyword_cancel_unlisted
        # Degraded mode: any "בטל"/"ביטול" (even "דמי ביטול") escalates. Deliberate over-escalation:
        # without the LLM we cannot tell context from request, and a missed escalation costs money.
        or (ex.via == "keywords" and ex.cancel_keyword_raw)
        or ex.escalation_only_cancel
    ):
        return claims_escalation(policy, Intent.CANCELLATION), path + ["rules:claims_escalation"]

    intent = ex.primary_intent
    if intent is Intent.CANCELLATION and ex.escalation_only_cancel and policy is not None:
        # "לא רוצה לבטל ... אלא ..." may only escalate; it never produces a cancellation answer or amounts.
        d = _cannot(intent, "UNCLEAR_REQUEST",
                    f"לא ברור מה הלקוח מבקש לגבי פוליסה {policy.policy_id} – בקש מהלקוח לפרט", policy=policy)
        return d, path + ["unclear"]
    if intent is Intent.POLICY_INFO and policy is None and llm is not None:
        intent = Intent.GENERAL_POLICY
    if intent in NEEDS_POLICY and policy is None:
        d = _cannot(intent, "NO_POLICY", "לא זוהתה פוליסה בשאלה", "מספר פוליסה, מספר לקוח או שם הלקוח")
        return d, path

    try:
        d, step = _dispatch(intent, policy, ex, question, repo, llm, as_of, policies_dir)
    except OverflowError:
        d, step = _cannot(intent, "DATE_OUT_OF_RANGE", "תאריכי הפוליסה מחוץ לטווח שניתן לחשב",
                          policy=policy, policy_id=None), "date_guard"
    return d, path + [step]


def _numbers_are_finite(numbers: dict) -> bool:
    return all(not (isinstance(v, float) and not math.isfinite(v)) for v in numbers.values())


def _source_unavailable(question: str) -> tuple[Decision, list[str]]:
    # The regulator rule still wins: it needs no policy data.
    if mentions_regulator(normalize_question(question)):
        return _regulator(Intent.UNKNOWN, None), ["guardrail:regulator"]
    d = Decision(Outcome.CANNOT_ANSWER, Intent.UNKNOWN, "DATA_SOURCE_UNAVAILABLE",
                 "מערכת הפוליסות לא זמינה כרגע – לא ניתן לענות על שאלה על פוליסה")
    d.missing.append("נתוני הפוליסה – יש לנסות שוב בעוד רגע")
    return d, ["data_source:unavailable"]


def _run(question: str, repo, llm, as_of: date, policies_dir) -> tuple[Decision, list[str], Extraction]:
    ex = extract(question, repo, llm)
    path = [f"intent:{ex.primary_intent.value}({ex.via})"]

    d, steps = _decide(question, ex, repo, llm, as_of, policies_dir)
    path += steps

    if not _numbers_are_finite(d.numbers):
        policy = repo.get(d.policy_id) if d.policy_id else None
        d = _cannot(d.intent, "CALCULATION_ERROR", "שגיאת חישוב – הסכומים לא יוצגו",
                    "חישוב תקין – יש לבדוק במערכת", policy=policy, policy_id=d.policy_id)
        path.append("calculation_error")

    if d.code not in ("MULTIPLE_CUSTOMERS", "CONFLICTING_IDENTIFIERS"):
        # With MULTIPLE_POLICIES every mentioned policy is already listed for the rep to choose from.
        other_ids = [] if d.code == "MULTIPLE_POLICIES" else ex.other_policy_ids
        extra = [f"השאלה מזכירה גם את פוליסה {pid} – יש לטפל בה בנפרד" for pid in other_ids]
        extra += [f"זוהתה בקשה נוספת: {INTENT_NAMES[i]} – יש לטפל בה בנפרד"
                  for i in ex.secondary_intents if i is not d.intent]
        d.warnings += extra
        if extra and d.outcome is not Outcome.ESCALATE:
            d.numbers["additional_request"] = True

    if d.outcome is not Outcome.ESCALATE and d.policy_id:
        policy = repo.get(d.policy_id)
        if policy is not None and policy.open_claims is None:
            d.warnings.append(f"לא ידוע אם לפוליסה תביעה פתוחה – אם הלקוח מבקש לבטל, יש להעביר למנהל משמרת ({CANCEL} §6).")
        elif policy is not None and policy.open_claims != 0:
            d.warnings.append(f"לפוליסה תביעה פתוחה – אם הלקוח מבקש לבטל, יש להעביר למנהל משמרת ({CANCEL} §6).")
    return d, path, ex


def answer(question: str, repo: PolicyRepository, llm=None, as_of: date = DEFAULT_AS_OF,
           policies_dir: str | Path = "data/policies") -> Response:
    started = time.perf_counter()
    repo = _OneReadPerCall(ValidatedPolicyRepository(repo))
    try:
        d, path, ex = _run(question, repo, llm, as_of, policies_dir)
    except RepositoryUnavailable as exc:
        # Unavailable is never "not found". The technical detail goes to stderr only.
        print(f"Policy data source unavailable: {exc}", file=sys.stderr)
        d, path = _source_unavailable(question)
        ex = None

    if d.code == "LLM_UNAVAILABLE":
        banner = BANNER_LLM_UNAVAILABLE
    elif ex is not None and ex.llm_failed:
        banner = BANNER_KEYWORDS
    else:
        banner = None
    text = render(d)
    if banner:
        text = f"{banner}\n\n{text}"
    path.append("template")
    latency_ms = round((time.perf_counter() - started) * 1000)
    return Response(decision=d, text=text, latency_ms=latency_ms, path=" → ".join(path), banner=banner)
