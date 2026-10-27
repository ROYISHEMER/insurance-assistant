"""python -m eval.run_eval [--no-llm] [--latency-budget-ms 5000] [--md PATH]

Runs every case in eval/cases.json through the real pipeline and checks it against hand-computed
expectations. Exit 1 only if a core case fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from insurance_assistant.assistant import answer
from insurance_assistant.cli import load_sources
from insurance_assistant.llm import get_llm
from insurance_assistant.models import DEFAULT_AS_OF, StartupError

CASES_PATH = Path(__file__).resolve().parent / "cases.json"
AS_OF = DEFAULT_AS_OF
MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")
NOT_ANSWER_EXPECTED = ("CANNOT_ANSWER", "ANSWER_WITH_GAPS")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_equal(got, expected) -> bool:
    """True is not 1: booleans only equal booleans; 492.0 equals 492."""
    if isinstance(got, bool) or isinstance(expected, bool):
        return isinstance(got, bool) and isinstance(expected, bool) and got == expected
    return got == expected


def check_case(case: dict, decision) -> list[str]:
    reasons = []
    if decision.outcome.value != case["outcome"]:
        reasons.append(f"outcome {decision.outcome.value} ≠ {case['outcome']}")
    if "code" in case and decision.code != case["code"]:
        reasons.append(f"code {decision.code} ≠ {case['code']}")
    if "policy" in case and decision.policy_id != case["policy"]:
        reasons.append(f"policy {decision.policy_id} ≠ {case['policy']}")
    for key, expected in case.get("numbers", {}).items():
        if key not in decision.numbers:
            reasons.append(f"{key} missing (expected {expected!r})")
        elif not strict_equal(decision.numbers[key], expected):
            reasons.append(f"{key} {decision.numbers[key]!r} ≠ {expected!r}")
    for key in case.get("absent", []):
        if key in decision.numbers:
            reasons.append(f"{key} should be absent (got {decision.numbers[key]!r})")
    return reasons


def run_case(case: dict, repo, llm) -> dict:
    started = time.perf_counter()
    result = {"id": case["id"], "set": case["set"], "q": case["q"], "trap": case.get("trap", ""),
              "expected": case["outcome"], "outcome": None, "code": None, "response": None}
    try:
        response = answer(case["q"], repo, llm=llm, as_of=AS_OF)
    except Exception as exc:  # an exception is a failure, never a crash of the run
        result.update(latency_ms=round((time.perf_counter() - started) * 1000),
                      reasons=[f"exception: {type(exc).__name__}: {exc}"], passed=False)
        return result
    reasons = check_case(case, response.decision)
    result.update(latency_ms=response.latency_ms, reasons=reasons, passed=not reasons,
                  outcome=response.decision.outcome.value, code=response.decision.code, response=response)
    return result


def nearest_rank(values: list[int], percent: float) -> int:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percent / 100 * len(ordered))) - 1]


def summarize(results: list[dict], budget_ms: int) -> dict:
    sets = {}
    for name in dict.fromkeys(r["set"] for r in results):
        group = [r for r in results if r["set"] == name]
        escalations = [r for r in group if r["expected"] == "ESCALATE"]
        guarded = [r for r in group if r["expected"] in NOT_ANSWER_EXPECTED]
        sets[name] = {
            "passed": sum(r["passed"] for r in group),
            "total": len(group),
            "escalated": sum(r["outcome"] == "ESCALATE" for r in escalations),
            "escalations_expected": len(escalations),
            # Outcome-level: the case did not come back as a confident ANSWER (an exception counts as a miss).
            "no_invention": sum(r["outcome"] is not None and r["outcome"] != "ANSWER" for r in guarded),
            "no_invention_total": len(guarded),
        }
    latencies = [r["latency_ms"] for r in results]
    return {
        "sets": sets,
        "latency": {
            "p50": nearest_rank(latencies, 50),
            "p95": nearest_rank(latencies, 95),
            "max": max(latencies),
            "over_budget": sum(v > budget_ms for v in latencies),
            "budget": budget_ms,
        },
    }


def _ratio(a: int, b: int) -> str:
    return f"{a}/{b}" if b else "—"


def print_report(results: list[dict], summary: dict, mode: str) -> None:
    width = max(len(r["id"]) for r in results)
    print(f"{'id':<{width}}  {'set':<8}  {'result':<6}  {'ms':>6}  details")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        details = "; ".join(r["reasons"]) if r["reasons"] else r["trap"]
        print(f"{r['id']:<{width}}  {r['set']:<8}  {status:<6}  {r['latency_ms']:>6}  {details}")
    print()
    print(f"Mode: {mode}")
    for name, s in summary["sets"].items():
        print(f"[{name}] passed {_ratio(s['passed'], s['total'])} | escalation recall "
              f"{_ratio(s['escalated'], s['escalations_expected'])} | no invention "
              f"{_ratio(s['no_invention'], s['no_invention_total'])}")
    lat = summary["latency"]
    print(f"Latency: p50 {lat['p50']} ms | p95 {lat['p95']} ms | max {lat['max']} ms | "
          f"over {lat['budget']} ms budget: {lat['over_budget']}")


SET_NAMES = {"core": "ליבה (חישובים, הסלמות, זיהוי)", "language": "ניסוחים חופשיים"}


def markdown_report(results: list[dict], summary: dict, mode: str) -> str:
    lat = summary["latency"]
    lines = [
        "# דוח בדיקת העוזר לנציגים",
        "",
        f"מצב הרצה: **{'עם מודל שפה' if mode == 'model' else 'מילות מפתח בלבד (ללא מודל שפה)'}** · "
        f"תאריך החישוב: {AS_OF.isoformat()}",
        "",
        "## סיכום",
        "",
        "| קבוצה | עברו | הועברו למנהל כשנדרש | לא \"המציא\" תשובה כשחסרו נתונים |",
        "|---|---|---|---|",
    ]
    for name, s in summary["sets"].items():
        lines.append(f"| {SET_NAMES.get(name, name)} | {_ratio(s['passed'], s['total'])} | "
                     f"{_ratio(s['escalated'], s['escalations_expected'])} | "
                     f"{_ratio(s['no_invention'], s['no_invention_total'])} |")
    lines += [
        "",
        f"זמן תגובה: חציון {lat['p50']} מ\"ש, 95% מהשאלות עד {lat['p95']} מ\"ש, המקסימום {lat['max']} מ\"ש; "
        f"{lat['over_budget']} שאלות מעל היעד של {lat['budget']} מ\"ש.",
        "",
        "**איך לקרוא:** כל שאלה נבדקה מול תשובה שחושבה ידנית מתוך הנהלים. \"הועברו למנהל\" – מקרים "
        "שבהם הנוהל מחייב מנהל משמרת (תביעה פתוחה, רגולטור). \"לא המציא\" – מקרים שבהם חסר מידע, "
        "והכלי אכן לא נתן תשובה סופית.",
        "",
        "## כל המקרים",
        "",
        "| # | מקרה | שאלה | מה נבדק (המלכודת) | תוצאה |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        status = "✅ עבר" if r["passed"] else "❌ נכשל: " + "; ".join(r["reasons"])
        question = r["q"].replace("|", "\\|")
        lines.append(f"| {i} | {r['id']} | {question} | {r['trap'] or '—'} | {status} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_eval")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--latency-budget-ms", type=int, default=5000)
    parser.add_argument("--md")
    args = parser.parse_args(argv)

    try:
        repo = load_sources()
    except StartupError as exc:
        print(f"שגיאה: {exc}", file=sys.stderr)
        return 1
    llm = None if args.no_llm else get_llm()
    mode = "model" if llm is not None else "keywords only"

    results = [run_case(case, repo, llm) for case in load_cases()]
    summary = summarize(results, args.latency_budget_ms)
    print_report(results, summary, mode)
    if args.md:
        Path(args.md).write_text(markdown_report(results, summary, mode), encoding="utf-8")
        print(f"Markdown report: {args.md}")
    core_failed = any(not r["passed"] for r in results if r["set"] == "core")
    return 1 if core_failed else 0


if __name__ == "__main__":
    sys.exit(main())
