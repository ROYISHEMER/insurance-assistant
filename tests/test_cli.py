import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from insurance_assistant.cli import load_sources, main, parse_as_of
from insurance_assistant.models import StartupError
from insurance_assistant.policy_docs import POLICY_DOC_FILES

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def run(capsys, *argv, **kwargs):
    code = main(list(argv), **kwargs)
    out, err = capsys.readouterr()
    return code, out, err


# --- --as-of ---------------------------------------------------------------------------------

def test_as_of_default_is_export_date():
    assert parse_as_of(None) == date(2026, 10, 25)


@pytest.mark.parametrize("raw,expected", [
    ("2000-01-01", date(2000, 1, 1)),
    ("2100-12-31", date(2100, 12, 31)),
    ("2024-02-29", date(2024, 2, 29)),
])
def test_as_of_valid(raw, expected):
    assert parse_as_of(raw) == expected


@pytest.mark.parametrize("raw", ["1999-12-31", "2101-01-01", "2026-2-5", "2026-02-30", "2025-02-29",
                                 "26-10-25", " 2026-10-25", "2026-10-25 ", "25/10/2026", "", "today",
                                 "2026-10-25T00:00"])
def test_as_of_invalid(raw):
    with pytest.raises(StartupError):
        parse_as_of(raw)


def test_bad_as_of_exits_1(capsys):
    code, out, err = run(capsys, "4471 רוצה לבטל", "--as-of", "2026-13-01", "--no-llm")
    assert code == 1 and err.startswith("שגיאה: ") and "Traceback" not in err and out == ""


# --- startup errors -----------------------------------------------------------------------------

def test_missing_csv(capsys, tmp_path):
    code, out, err = run(capsys, "4471", "--no-llm", csv_path=tmp_path / "none.csv")
    assert code == 1 and err.startswith("שגיאה: ") and "Traceback" not in err


def test_broken_csv(capsys, tmp_path):
    path = tmp_path / "policies.csv"
    path.write_text("policy_id\n", encoding="utf-8")
    code, _, err = run(capsys, "4471", "--no-llm", csv_path=path)
    assert code == 1 and "שגיאה: " in err


def test_missing_docs(capsys, tmp_path):
    folder = tmp_path / "policies"
    folder.mkdir()
    for name in POLICY_DOC_FILES[:3]:
        (folder / name).write_bytes((ROOT / "data" / "policies" / name).read_bytes())
    code, _, err = run(capsys, "4471", "--no-llm", policies_dir=folder)
    assert code == 1 and POLICY_DOC_FILES[3] in err and "Traceback" not in err


def test_load_sources():
    repo = load_sources()
    assert repo.get("4471") is not None
    with pytest.raises(StartupError):
        load_sources(policies_dir=ROOT / "nope")


# --- questions ---------------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_empty_question(capsys, question):
    code, out, err = run(capsys, question, "--no-llm")
    assert code == 1 and "השאלה ריקה" in err


def test_question_prints_text_latency_and_path(capsys):
    code, out, err = run(capsys, "4471 רוצה לבטל", "--no-llm")
    assert code == 0
    assert out.startswith("[תשובה חלקית – חסרים נתונים] הערכת ביטול (פוליסה 4471)")
    assert "זמן: " in out and "מסלול: intent:cancellation(keywords) → policy:4471" in out


def test_debug_prints_decision_json(capsys):
    code, out, _ = run(capsys, "4471 רוצה לבטל", "--no-llm", "--debug")
    assert code == 0 and '"code": "CANCEL_ESTIMATE"' in out and '"conditional_estimate": 342.0' in out


def test_report(capsys):
    code, out, _ = run(capsys, "--report", "--no-llm")
    assert code == 0 and '"duplicate_names"' in out and "מור שפירא" in out


def test_interactive_mode(capsys, monkeypatch):
    answers = iter(["4471 רוצה לבטל", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    code, out, _ = run(capsys, "--no-llm")
    assert code == 0 and "פוליסה 4471" in out


def test_module_entry_point_no_traceback(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    ok = subprocess.run([sys.executable, "-m", "insurance_assistant.cli", "4471 רוצה לבטל", "--no-llm"],
                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert ok.returncode == 0 and "342" in ok.stdout
    bad = subprocess.run([sys.executable, "-m", "insurance_assistant.cli", "x", "--as-of", "2026-1-1"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert bad.returncode == 1 and bad.stderr.startswith("שגיאה: ") and "Traceback" not in bad.stderr
