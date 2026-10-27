import csv
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from insurance_assistant.models import Policy, StartupError
from insurance_assistant.normalize import parse_date, parse_float, parse_int
from insurance_assistant.policy_docs import POLICY_DOC_FILES, check_policy_docs, load_policy_docs
from insurance_assistant.repository import (
    CsvPolicyRepository,
    PolicyRepository,
    ValidatedPolicyRepository,
    validate_policy,
)

ROOT = Path(__file__).resolve().parent.parent
HEADER = ("policy_id,customer_id,customer_name,product,start_date,monthly_premium,"
          "status,last_payment_date,open_claims,notes")
GOOD_ROW = "5001,C-2001,ישראל ישראלי,דירה,2024-01-15,200,פעילה,2026-10-01,0,"


def write_csv(tmp_path, *lines, bom=True, name="policies.csv"):
    path = tmp_path / name
    path.write_bytes(("﻿" if bom else "").encode() + "\n".join(lines).encode("utf-8") + b"\n")
    return path


@pytest.fixture(scope="module")
def repo():
    return CsvPolicyRepository()


# --- normalize ----------------------------------------------------------------

def test_iso_dates_are_never_read_day_first():
    issues = []
    assert parse_date("2026-03-04", "d", issues) == date(2026, 3, 4)
    assert issues == []
    assert parse_date("2026-13-01", "d", issues) is None  # not swapped into 2026-01-13
    assert issues


def test_dd_mm_dates_are_normalized_and_flagged():
    issues = []
    assert parse_date("15/03/2023", "start_date", issues) == date(2023, 3, 15)
    assert len(issues) == 1 and "2023-03-15" in issues[0] and issues[0].startswith("start_date:")


@pytest.mark.parametrize("raw", ["3/4/2023", "2023/03/04", "04-03-2023", "31/02/2024", "abc", "15/03/23"])
def test_other_date_formats_are_not_guessed(raw):
    issues = []
    assert parse_date(raw, "d", issues) is None
    assert len(issues) == 1


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_missing_values_are_noted(raw):
    for parse in (parse_date, parse_int, parse_float):
        issues = []
        assert parse(raw, "f", issues) is None
        assert issues == ["f: ערך חסר"]


@pytest.mark.parametrize("raw", ["2.0", "abc", "1e3", "nan", "inf", "1" + "0" * 400, "1,000"])
def test_parse_int_rejects(raw):
    issues = []
    assert parse_int(raw, "open_claims", issues) is None
    assert len(issues) == 1


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e400", "abc", "1" + "0" * 400, "1,000"])
def test_parse_float_rejects(raw):
    issues = []
    assert parse_float(raw, "monthly_premium", issues) is None
    assert len(issues) == 1


def test_parse_numbers_accept_plain_values():
    issues = []
    assert parse_int(" 2 ", "f", issues) == 2
    assert parse_float("160", "f", issues) == 160.0
    assert parse_float("99.5", "f", issues) == 99.5
    assert issues == []


# --- the real export ------------------------------------------------------------

def test_repo_satisfies_protocol(repo):
    assert isinstance(repo, PolicyRepository)
    assert isinstance(ValidatedPolicyRepository(repo), PolicyRepository)
    assert not hasattr(PolicyRepository, "all_ids")
    assert repo.source_name == "policies.csv"


def test_policy_4421_last_payment_date(repo):
    policy = repo.get("4421")
    assert policy.last_payment_date == date(2026, 10, 1)
    assert policy.start_date == date(2024, 7, 2)
    assert any(i.startswith("last_payment_date:") for i in policy.data_issues)
    assert any(i.startswith("start_date:") for i in policy.data_issues)


def test_policy_4471_loads(repo):
    policy = repo.get("4471")
    assert policy == Policy(
        policy_id="4471", customer_id="C-1088", customer_name="רונן אביטל", product="רכב",
        start_date=date(2023, 2, 10), monthly_premium=410.0, status="פעילה",
        last_payment_date=date(2026, 10, 5), open_claims=0, notes="", data_issues=[],
        source="policies.csv", conflicting_records=False,
    )
    assert repo.get(4471) == policy


def test_unknown_policy_is_none(repo):
    assert repo.get("9999") is None
    assert repo.by_customer("C-0000") == []


def test_dd_mm_rows_flagged_in_report(repo):
    report = repo.data_quality_report()
    for pid in ("4408", "4421", "4449"):
        assert pid in report["issues"]
    assert repo.get("4408").start_date == date(2023, 3, 15)


def test_duplicate_names_reported_not_merged(repo):
    names = repo.data_quality_report()["duplicate_names"]
    assert names["מור שפירא"] == ["C-1059", "C-1063"]
    assert names["הילה אברהם"] == ["C-1067", "C-1093"]
    assert "אורן גבאי" not in names  # C-1072: one customer, two policies
    assert {p.policy_id for p in repo.by_customer("C-1072")} == {"4418", "4433"}


def test_find_names_in(repo):
    found = repo.find_names_in("הלקוחה מור שפירא רוצה לבטל")
    assert {p.customer_id for p in found} == {"C-1059", "C-1063"}
    assert repo.find_names_in("מור") == []


def test_notes_in_report(repo):
    notes = repo.data_quality_report()["notes"]
    assert set(notes) == {"4440", "4455"}


def test_returned_policies_are_copies(repo):
    repo.get("4471").data_issues.append("x")
    assert repo.get("4471").data_issues == []


# --- CSV structure: startup errors -------------------------------------------------

def assert_startup_error(path, *fragments):
    with pytest.raises(StartupError) as info:
        CsvPolicyRepository(path)
    message = str(info.value)
    for fragment in fragments:
        assert fragment in message, message
    assert "Traceback" not in message
    return message


def test_missing_file(tmp_path):
    assert_startup_error(tmp_path / "nope.csv", "לא נמצא")


def test_unreadable_file(tmp_path):
    folder = tmp_path / "dir.csv"
    folder.mkdir()
    assert_startup_error(folder, "לא נמצא")


def test_unreadable_permissions(tmp_path):
    path = write_csv(tmp_path, HEADER, GOOD_ROW)
    path.chmod(0)
    try:
        if path.open("rb"):  # running as root: permissions are not enforced
            pytest.skip("permissions not enforced")
    except PermissionError:
        assert_startup_error(path, "לא ניתן לקרוא")
    finally:
        path.chmod(0o644)


def test_not_utf8(tmp_path):
    path = tmp_path / "policies.csv"
    path.write_bytes((HEADER + "\n" + GOOD_ROW + "\n").encode("cp1255"))
    assert_startup_error(path, "UTF-8")


def test_field_larger_than_limit_reports_data_row(tmp_path):
    big = "x" * (csv.field_size_limit() + 10)
    rows = [GOOD_ROW.replace("5001", "5002"), GOOD_ROW.replace("5001", "5003"),
            f"5004,C-2001,ישראל,דירה,2024-01-15,200,פעילה,2026-10-01,0,{big}"]
    assert_startup_error(write_csv(tmp_path, HEADER, *rows), "שורת נתונים 3")


@pytest.mark.parametrize("content", ["", "   \n\n"])
def test_empty_file(tmp_path, content):
    path = tmp_path / "policies.csv"
    path.write_text(content, encoding="utf-8")
    assert_startup_error(path, "ריק")


def test_bom_only_file_is_empty(tmp_path):
    path = tmp_path / "policies.csv"
    path.write_bytes(b"\xef\xbb\xbf")
    assert_startup_error(path, "ריק")


def test_duplicate_headers(tmp_path):
    assert_startup_error(write_csv(tmp_path, HEADER + ",status", GOOD_ROW + ",x"), "כפולים", "status")


def test_missing_required_headers(tmp_path):
    header = HEADER.replace("open_claims", "claims")
    assert_startup_error(write_csv(tmp_path, header, GOOD_ROW), "חסרות", "open_claims")


def test_notes_column_is_optional(tmp_path):
    header = HEADER.removesuffix(",notes")
    repo = CsvPolicyRepository(write_csv(tmp_path, header, GOOD_ROW.removesuffix(",")))
    assert repo.get("5001").notes == ""


def test_header_only(tmp_path):
    assert_startup_error(write_csv(tmp_path, HEADER), "כותרת בלבד")


@pytest.mark.parametrize("row", [GOOD_ROW + ",extra", GOOD_ROW.removesuffix(",")])
def test_wrong_field_count(tmp_path, row):
    assert_startup_error(write_csv(tmp_path, HEADER, GOOD_ROW.replace("5001", "5000"), row),
                         "שורת נתונים 2")


def test_line_break_inside_field(tmp_path):
    row = GOOD_ROW.removesuffix(",") + ',"שורה\nשנייה"'
    assert_startup_error(write_csv(tmp_path, HEADER, row), "ירידת שורה", "שורת נתונים 1")


def test_every_row_skipped(tmp_path):
    rows = [GOOD_ROW.replace("5001", ""), GOOD_ROW.replace("5001", "  ")]
    assert_startup_error(write_csv(tmp_path, HEADER, *rows), "דולגו")


def test_empty_policy_id_rows_are_skipped(tmp_path):
    rows = [GOOD_ROW, GOOD_ROW.replace("5001", "").replace("C-2001", "C-2999")]
    repo = CsvPolicyRepository(write_csv(tmp_path, HEADER, *rows))
    assert repo.policy_ids() == ["5001"]
    assert repo.get("") is None
    assert repo.by_customer("C-2999") == []
    assert repo.find_names_in("ישראל ישראלי") == [repo.get("5001")]
    assert repo.data_quality_report()["skipped_rows"] == [{"row": 2, "reason": "מספר פוליסה ריק"}]


def test_duplicate_policy_id_flags_both_rows(tmp_path):
    second = GOOD_ROW.replace(",200,", ",300,")
    repo = CsvPolicyRepository(write_csv(tmp_path, HEADER, GOOD_ROW, second))
    first = repo.get("5001")
    assert first.monthly_premium == 200.0
    assert first.conflicting_records
    assert any("2 שורות" in i for i in first.data_issues)
    both = repo.by_customer("C-2001")
    assert len(both) == 2 and all(p.conflicting_records and p.data_issues for p in both)
    assert repo.data_quality_report()["conflicting_policy_ids"] == ["5001"]


def test_no_bom_is_fine(tmp_path):
    repo = CsvPolicyRepository(write_csv(tmp_path, HEADER, GOOD_ROW, bom=False))
    assert repo.get("5001").monthly_premium == 200.0


# --- validation: CSV and any other source get the same handling ------------------------

def make_policy(**overrides):
    base = Policy(
        policy_id="5001", customer_id="C-2001", customer_name="ישראל ישראלי", product="דירה",
        start_date=date(2024, 1, 15), monthly_premium=200.0, status="פעילה",
        last_payment_date=date(2026, 10, 1), open_claims=0, notes="",
    )
    return replace(base, **overrides)


class FakeRepository:
    """In-memory source, standing in for the production API."""

    source_name = "fake-api"

    def __init__(self, *policies):
        self._policies = list(policies)

    def get(self, policy_id):
        return next((p for p in self._policies if p.policy_id == policy_id), None)

    def by_customer(self, customer_id):
        return [p for p in self._policies if p.customer_id == customer_id]

    def find_names_in(self, text):
        return [p for p in self._policies if p.customer_name in text]


def csv_policy(tmp_path, field, raw):
    values = dict(zip(HEADER.split(","), GOOD_ROW.split(",")))
    values[field] = raw
    path = write_csv(tmp_path, HEADER, ",".join(values.values()))
    return ValidatedPolicyRepository(CsvPolicyRepository(path)).get("5001")


def api_policy(field, value):
    return ValidatedPolicyRepository(FakeRepository(make_policy(**{field: value}))).get("5001")


PREMIUM_CASES = [  # (API value, CSV text, expected)
    (-1, "-1", None),
    (math.nan, "nan", None),
    (math.inf, "inf", None),
    (True, "True", None),
    (100_001, "100001", None),
    (10**400, "1" + "0" * 400, None),
    (2.0, "2.0", 2.0),
    (0, "0", 0.0),
    (100_000, "100000", 100_000.0),
]


@pytest.mark.parametrize("api_value,csv_text,expected", PREMIUM_CASES)
def test_premium_validation_same_for_csv_and_api(tmp_path, api_value, csv_text, expected):
    for policy in (api_policy("monthly_premium", api_value), csv_policy(tmp_path, "monthly_premium", csv_text)):
        assert policy.monthly_premium == expected
        flagged = any(i.startswith("monthly_premium:") for i in policy.data_issues)
        assert flagged == (expected is None)


CLAIMS_CASES = [
    (-1, "-1", None),
    (math.nan, "nan", None),
    (math.inf, "inf", None),
    (True, "True", None),
    (2.0, "2.0", None),
    (10**400, "1" + "0" * 400, None),
    (1_001, "1001", None),
    (0, "0", 0),
    (3, "3", 3),
]


@pytest.mark.parametrize("api_value,csv_text,expected", CLAIMS_CASES)
def test_open_claims_validation_same_for_csv_and_api(tmp_path, api_value, csv_text, expected):
    for policy in (api_policy("open_claims", api_value), csv_policy(tmp_path, "open_claims", csv_text)):
        assert policy.open_claims == expected
        assert type(policy.open_claims) in (int, type(None))
        flagged = any(i.startswith("open_claims:") for i in policy.data_issues)
        assert flagged == (expected is None)


def test_missing_values_same_for_csv_and_api(tmp_path):
    for field in ("monthly_premium", "open_claims", "start_date"):
        for policy in (api_policy(field, None), csv_policy(tmp_path, field, "")):
            assert getattr(policy, field) is None
            assert [i for i in policy.data_issues if i.startswith(f"{field}:")] == [f"{field}: ערך חסר"]


def test_validation_does_not_mutate_input():
    original = make_policy(monthly_premium=-5, open_claims=True, data_issues=["קיים"])
    snapshot = replace(original, data_issues=list(original.data_issues))
    checked = validate_policy(original, "fake-api")
    assert original == snapshot
    assert checked is not original and checked.data_issues is not original.data_issues
    assert checked.source == "fake-api" and original.source == ""


def test_validation_never_raises_and_is_idempotent():
    class Exploding:
        def __float__(self):
            raise RuntimeError

    weird = make_policy(monthly_premium=Exploding(), open_claims="2", start_date="2024-01-15",
                        policy_id=None, notes=None, data_issues=None)
    once = validate_policy(weird, "fake-api")
    assert once.monthly_premium is None and once.open_claims is None and once.start_date is None
    assert once.notes == ""
    twice = validate_policy(once, "fake-api")
    assert twice == once


def test_wrapped_repo_validates_every_lookup():
    bad = make_policy(monthly_premium=-5)
    repo = ValidatedPolicyRepository(FakeRepository(bad))
    assert repo.source_name == "fake-api"
    for policy in (repo.get("5001"), *repo.by_customer("C-2001"), *repo.find_names_in("ישראל ישראלי")):
        assert policy.monthly_premium is None and policy.source == "fake-api"
    assert bad.monthly_premium == -5


# --- procedure documents -----------------------------------------------------------

def copy_docs(tmp_path):
    folder = tmp_path / "policies"
    folder.mkdir()
    for name in POLICY_DOC_FILES:
        (folder / name).write_bytes((ROOT / "data" / "policies" / name).read_bytes())
    return folder


def test_real_docs_load():
    docs = load_policy_docs(ROOT / "data" / "policies")
    assert list(docs) == list(POLICY_DOC_FILES)
    assert "## 6. פוליסה עם תביעה פתוחה" in docs["01_תנאי_ביטול.md"]
    check_policy_docs()


def test_docs_folder_missing(tmp_path):
    with pytest.raises(StartupError, match="לא נמצאה"):
        check_policy_docs(tmp_path / "missing")


@pytest.mark.parametrize("name", POLICY_DOC_FILES)
def test_doc_file_missing(tmp_path, name):
    folder = copy_docs(tmp_path)
    (folder / name).unlink()
    with pytest.raises(StartupError, match=name):
        check_policy_docs(folder)


def test_doc_not_utf8(tmp_path):
    folder = copy_docs(tmp_path)
    (folder / POLICY_DOC_FILES[1]).write_bytes("## 1. כללי".encode("cp1255"))
    with pytest.raises(StartupError, match="UTF-8"):
        check_policy_docs(folder)


def test_doc_unreadable(tmp_path):
    folder = copy_docs(tmp_path)
    (folder / POLICY_DOC_FILES[2]).unlink()
    (folder / POLICY_DOC_FILES[2]).mkdir()
    with pytest.raises(StartupError, match=POLICY_DOC_FILES[2]):
        check_policy_docs(folder)


@pytest.mark.parametrize("text", ["# כותרת בלבד\nטקסט", "## כללי\nבלי מספור", "##1. צמוד"])
def test_doc_without_section_heading(tmp_path, text):
    folder = copy_docs(tmp_path)
    (folder / POLICY_DOC_FILES[3]).write_text(text, encoding="utf-8")
    with pytest.raises(StartupError, match="כותרת סעיף"):
        check_policy_docs(folder)
