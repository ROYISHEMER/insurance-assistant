"""The example API adapter against a local fake server that serves the CSV under the assumed contract."""
import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest

from eval.run_eval import load_cases
from insurance_assistant.api_repository import ApiPolicyRepository
from insurance_assistant.assistant import answer
from insurance_assistant.cli import main
from insurance_assistant.models import Outcome
from insurance_assistant.repository import CsvPolicyRepository, RepositoryUnavailable

ROOT = Path(__file__).resolve().parent.parent
MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")
TOKEN = "secret-token-7f3a9c"


def csv_records():
    """The CSV rows as an API would send them: numbers as JSON numbers where they are plain integers."""
    with open(ROOT / "data" / "policies.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in ("monthly_premium", "open_claims"):
            if row[field].isdigit():
                row[field] = int(row[field])
    return {row["policy_id"]: row for row in rows}


class FakePolicyApi(BaseHTTPRequestHandler):
    records: dict = {}
    mode = "ok"  # ok | 500 | 503 | slow | badjson | list | object | 302 | 204 | wrong-id
    auth_headers: list = []

    def log_message(self, *args):  # keep test output clean
        pass

    def _send(self, status, body=b"", content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).auth_headers.append(self.headers.get("Authorization"))
        mode = type(self).mode
        if mode in ("500", "503"):
            return self._send(int(mode), b'{"error": "boom"}')
        if mode == "204":
            return self._send(204)
        if mode == "slow":
            time.sleep(0.6)
        if mode == "badjson":
            return self._send(200, b"{not json")
        parts = [unquote(p) for p in self.path.strip("/").split("/")]
        if len(parts) == 2 and parts[0] == "policies":
            record = type(self).records.get(parts[1])
            if record is None:
                return self._send(404, b'{"error": "not found"}')
            if mode == "list":
                record = [record]
            if mode == "wrong-id":
                record = dict(record, policy_id="4401")
            return self._send(200, json.dumps(record, ensure_ascii=False).encode())
        if len(parts) == 3 and parts[0] == "customers" and parts[2] == "policies":
            found = [r for r in type(self).records.values() if r["customer_id"] == parts[1]]
            payload = {"policies": found} if mode == "object" else found
            return self._send(200, json.dumps(payload, ensure_ascii=False).encode())
        return self._send(404)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakePolicyApi)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    FakePolicyApi.records = csv_records()
    FakePolicyApi.mode = "ok"
    FakePolicyApi.auth_headers = []
    monkeypatch.delenv("POLICY_API_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="module")
def csv_repo():
    return CsvPolicyRepository()


def api(server, **kwargs):
    return ApiPolicyRepository(server, **kwargs)


def no_money(d):
    return not any(k in d.numbers for k in MONEY_KEYS)


# --- same decisions as the CSV -----------------------------------------------------------------------

def core_questions_without_names():
    names_repo = CsvPolicyRepository()
    return [c for c in load_cases() if c["set"] == "core" and not names_repo.find_names_in(c["q"])]


@pytest.mark.parametrize("case", core_questions_without_names(), ids=lambda c: c["id"])
def test_same_decision_as_csv(server, csv_repo, case):
    from_csv = answer(case["q"], csv_repo, llm=None).decision
    from_api = answer(case["q"], api(server), llm=None).decision
    assert (from_api.outcome, from_api.code, from_api.policy_id) == (from_csv.outcome, from_csv.code, from_csv.policy_id)
    assert from_api.numbers == from_csv.numbers
    sources = {f.source.replace("policy-api", "<src>") for f in from_api.facts}
    assert sources == {f.source.replace("policies.csv", "<src>") for f in from_csv.facts}


def test_name_questions_are_excluded_by_design(server):
    assert api(server).find_names_in("רונן אביטל רוצה לבטל") == []
    assert answer("רונן אביטל רוצה לבטל", api(server)).decision.code == "NO_POLICY"


def test_every_policy_parses_identically(server, csv_repo):
    repo = api(server)
    for pid in csv_repo.policy_ids():
        a, c = repo.get(pid), csv_repo.get(pid)
        assert a.source == "policy-api"
        assert a == type(c)(**{**c.__dict__, "source": "policy-api"}), pid


def test_by_customer(server):
    assert {p.policy_id for p in api(server).by_customer("C-1072")} == {"4418", "4433"}
    assert api(server).by_customer("C-9999") == []


# --- not found vs unavailable ------------------------------------------------------------------------

def test_404_is_policy_not_found(server):
    assert api(server).get("4461") is None
    d = answer("4461 רוצה לבטל", api(server)).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "POLICY_NOT_FOUND")


@pytest.mark.parametrize("mode", ["500", "503", "slow", "badjson", "list", "204", "wrong-id"])
def test_unavailable_is_never_not_found(server, mode, capsys):
    FakePolicyApi.mode = mode
    repo = api(server, timeout=0.2)
    with pytest.raises(RepositoryUnavailable):
        repo.get("4471")
    r = answer("4471 רוצה לבטל", repo)
    d = r.decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "DATA_SOURCE_UNAVAILABLE") and no_money(d)
    assert d.headline == "מערכת הפוליסות לא זמינה כרגע – לא ניתן לענות על שאלה על פוליסה"
    assert d.missing and "HTTP" not in r.text and "JSON" not in r.text
    assert "unavailable" in capsys.readouterr().err


def test_customer_list_of_wrong_shape_is_unavailable(server):
    FakePolicyApi.mode = "object"
    assert answer("C-1072 רוצה לבטל", api(server)).decision.code == "DATA_SOURCE_UNAVAILABLE"


def test_unreachable_server_is_unavailable():
    d = answer("4471 רוצה לבטל", ApiPolicyRepository("http://127.0.0.1:9", timeout=0.5)).decision
    assert d.code == "DATA_SOURCE_UNAVAILABLE"


def test_regulator_still_wins_when_unavailable(server):
    FakePolicyApi.mode = "500"
    d = answer("4471 רוצה לבטל ויתלונן לרשות שוק ההון", api(server)).decision
    assert (d.outcome, d.code) == (Outcome.ESCALATE, "REGULATOR")


# --- malformed fields: the same data issues as the CSV -------------------------------------------------

HEADER = ("policy_id,customer_id,customer_name,product,start_date,monthly_premium,status,"
          "last_payment_date,open_claims,notes")


@pytest.mark.parametrize("field,csv_text,json_value", [
    ("open_claims", "-1", -1),
    ("start_date", "15/03/2023", "15/03/2023"),
    ("monthly_premium", "100001", 100001),
    ("monthly_premium", "inf", float("inf")),
    ("open_claims", "2.0", 2.0),
    ("open_claims", "True", True),
    ("monthly_premium", "", None),
])
def test_malformed_field_same_issue_as_csv(server, tmp_path, field, csv_text, json_value):
    base = {"policy_id": "5001", "customer_id": "C-2001", "customer_name": "ישראל ישראלי", "product": "דירה",
            "start_date": "2024-01-15", "monthly_premium": "200", "status": "פעילה",
            "last_payment_date": "2026-10-01", "open_claims": "0", "notes": ""}
    path = tmp_path / "policies.csv"
    path.write_text(HEADER + "\n" + ",".join({**base, field: csv_text}.values()) + "\n", encoding="utf-8")
    from_csv = CsvPolicyRepository(path).get("5001")
    FakePolicyApi.records = {"5001": {**base, field: json_value}}
    from_api = api(server).get("5001")
    assert getattr(from_api, field) == getattr(from_csv, field)
    assert from_api.data_issues == from_csv.data_issues and from_csv.data_issues


def test_missing_key_is_missing_value(server):
    record = dict(csv_records()["4471"])
    del record["open_claims"]
    FakePolicyApi.records = {"4471": record}
    policy = api(server).get("4471")
    assert policy.open_claims is None and "open_claims: ערך חסר" in policy.data_issues
    assert answer("4471 רוצה לבטל", api(server)).decision.code == "UNKNOWN_CLAIMS"


# --- token -----------------------------------------------------------------------------------------------------

def test_token_is_sent_and_never_printed(server, monkeypatch, capsys):
    monkeypatch.setenv("POLICY_API_TOKEN", TOKEN)
    repo = api(server)
    r = answer("4471 רוצה לבטל", repo)
    assert FakePolicyApi.auth_headers and set(FakePolicyApi.auth_headers) == {f"Bearer {TOKEN}"}
    FakePolicyApi.mode = "500"
    failed = answer("4471 רוצה לבטל", repo)
    out, err = capsys.readouterr()
    for text in (r.text, failed.text, r.path, failed.path, out, err, repr(repo)):
        assert TOKEN not in text
    assert main(["4471 רוצה לבטל", "--source", "api", "--api-url", server, "--no-llm", "--debug"]) == 0
    out, err = capsys.readouterr()
    assert TOKEN not in out + err


def test_no_token_no_header(server):
    api(server).get("4471")
    assert FakePolicyApi.auth_headers == [None]


# --- CLI ---------------------------------------------------------------------------------------------------------

def test_cli_api_source_skips_csv_checks(server, capsys, tmp_path):
    code = main(["4471 רוצה לבטל", "--source", "api", "--api-url", server, "--no-llm"],
                csv_path=tmp_path / "missing.csv")
    out, _ = capsys.readouterr()
    assert code == 0 and "policy-api, פוליסה 4471" in out


def test_cli_api_url_from_env(server, capsys, monkeypatch):
    monkeypatch.setenv("POLICY_API_URL", server)
    assert main(["4471 רוצה לבטל", "--source", "api", "--no-llm"]) == 0


@pytest.mark.parametrize("argv,fragment", [
    (["4471", "--source", "api", "--no-llm"], "--api-url"),
    (["4471", "--source", "api", "--api-url", "ftp://x", "--no-llm"], "http"),
    (["--source", "api", "--api-url", "http://x", "--report"], "--report"),
])
def test_cli_api_errors(capsys, monkeypatch, argv, fragment):
    monkeypatch.delenv("POLICY_API_URL", raising=False)
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("שגיאה: ") and fragment in err and "Traceback" not in err


def test_cli_api_still_checks_documents(server, capsys, tmp_path):
    assert main(["4471", "--source", "api", "--api-url", server], policies_dir=tmp_path / "none") == 1
