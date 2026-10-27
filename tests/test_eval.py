from types import SimpleNamespace

from eval.run_eval import check_case, load_cases, nearest_rank, strict_equal, summarize
from eval.stability import classify
from insurance_assistant.models import Decision, Intent, Outcome


def decision(outcome="ANSWER_WITH_GAPS", code="CANCEL_ESTIMATE", policy_id="4471", **numbers):
    return Decision(Outcome(outcome), Intent.CANCELLATION, code, "h", policy_id=policy_id, numbers=numbers)


def test_strict_equal():
    assert strict_equal(492.0, 492) and strict_equal(False, False) and strict_equal("2026-10-31", "2026-10-31")
    assert not strict_equal(True, 1) and not strict_equal(1, True) and not strict_equal(0, False)


def test_check_case():
    case = {"outcome": "ANSWER_WITH_GAPS", "code": "CANCEL_ESTIMATE", "policy": "4471",
            "numbers": {"fee": 150, "arrears": False}, "absent": ["additional_request"]}
    assert check_case(case, decision(fee=150.0, arrears=False)) == []
    reasons = check_case(case, decision(outcome="ANSWER", policy_id="4418", fee=151, arrears=0,
                                        additional_request=True))
    assert len(reasons) == 5
    assert check_case(case, decision(arrears=False)) == ["fee missing (expected 150)"]


def test_nearest_rank():
    values = list(range(1, 21))
    assert nearest_rank(values, 50) == 10 and nearest_rank(values, 95) == 19 and nearest_rank(values, 100) == 20
    assert nearest_rank([7], 95) == 7


def test_summary_counts_no_invention_at_outcome_level():
    results = [
        {"set": "core", "expected": "CANNOT_ANSWER", "outcome": "ANSWER_WITH_GAPS", "passed": False, "latency_ms": 5},
        {"set": "core", "expected": "ANSWER_WITH_GAPS", "outcome": "ANSWER", "passed": False, "latency_ms": 6000},
        {"set": "core", "expected": "ESCALATE", "outcome": "ESCALATE", "passed": True, "latency_ms": 1},
        {"set": "core", "expected": "CANNOT_ANSWER", "outcome": None, "passed": False, "latency_ms": 2},
    ]
    s = summarize(results, 5000)
    assert (s["sets"]["core"]["no_invention"], s["sets"]["core"]["no_invention_total"]) == (1, 3)
    assert (s["sets"]["core"]["escalated"], s["sets"]["core"]["escalations_expected"]) == (1, 1)
    assert s["latency"]["over_budget"] == 1 and s["latency"]["max"] == 6000


def run(passed, d=None, reasons=()):
    return {"passed": passed, "reasons": list(reasons), "response": SimpleNamespace(decision=d, path="p")}


CASE = {"outcome": "ANSWER_WITH_GAPS", "code": "CANCEL_ESTIMATE", "numbers": {"fee": 150},
        "absent": ["conditional_estimate"]}


def test_classify_stable_and_flip():
    good = decision(fee=150)
    assert classify(CASE, [run(True, good)] * 5)[0] == "STABLE-CORRECT"
    other_code = decision(code="TRIP_STARTED", fee=150)
    verdict, _ = classify(CASE, [run(True, good)] * 4 + [run(False, other_code, ["code"])])
    assert verdict == "HARMLESS FLIP (4/5 correct)"


def test_classify_errors():
    good = decision(fee=150)
    assert classify(CASE, [run(True, good)] * 4 + [run(False, decision(fee=50), ["fee"])])[0] == "ERROR"
    invented = decision(fee=150, conditional_estimate=10)
    assert classify(CASE, [run(True, good)] * 4 + [run(False, invented, ["x"])])[0] == "ERROR"
    escalated = decision(outcome="ESCALATE", code="OPEN_CLAIM")
    assert classify(CASE, [run(True, good)] * 4 + [run(False, escalated, ["x"])])[0] == "ERROR"
    answered = decision(outcome="ANSWER", fee=150)
    assert classify(CASE, [run(True, good)] * 4 + [run(False, answered, ["x"])])[0] == "ERROR"
    wrong_code = decision(code="TRIP_STARTED", fee=150)
    assert classify(CASE, [run(True, good)] * 2 + [run(False, wrong_code, ["c"])] * 3)[0] == "ERROR"
    crashed = {"passed": False, "reasons": ["exception: RuntimeError: x"], "response": None}
    assert classify(CASE, [run(True, good)] * 4 + [crashed])[0] == "ERROR"


def test_cases_file_shape():
    cases = load_cases()
    assert len({c["id"] for c in cases}) == len(cases) == 57
    for c in cases:
        if c["id"] not in ("two-policies", "cancel-and-refund"):
            assert "additional_request" in c["absent"], c["id"]
    by_id = {c["id"]: c for c in cases}
    assert "conditional_estimate" in by_id["arrears"]["absent"] and "grant" in by_id["new-health"]["absent"]
