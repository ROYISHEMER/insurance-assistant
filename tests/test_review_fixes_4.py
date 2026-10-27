"""Fourth review round (phase 7d)."""
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from insurance_assistant import assistant
from insurance_assistant.api_repository import ApiPolicyRepository, token_is_valid
from insurance_assistant.assistant import GENERAL_SYSTEM_PROMPT, answer
from insurance_assistant.cli import main
from insurance_assistant.extract import keyword_signals
from insurance_assistant.models import Intent, Outcome
from insurance_assistant.repository import CsvPolicyRepository, RepositoryUnavailable

MONEY_KEYS = ("fee", "grant", "potential_grant", "conditional_estimate")
RECORD = {"policy_id": "5001", "customer_id": "C-2001", "customer_name": "ישראל ישראלי", "product": "דירה",
          "start_date": "2020-01-15", "monthly_premium": 200, "status": "פעילה",
          "last_payment_date": "2026-10-01", "open_claims": 0, "notes": ""}


def no_money(d):
    return not any(k in d.numbers for k in MONEY_KEYS)


# --- a tiny scriptable HTTP server ----------------------------------------------------------------------

class Scripted(BaseHTTPRequestHandler):
    routes: dict = {}  # path -> callable(handler)
    seen: list = []

    def log_message(self, *args):
        pass

    def send(self, status, body=b"", headers=None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).seen.append((self.server.server_address[1], self.path, self.headers.get("Authorization")))
        route = type(self).routes.get((self.server.server_address[1], self.path))
        if route is None:
            return self.send(404, b"{}")
        route(self)


def json_route(payload, status=200):
    return lambda h: h.send(status, json.dumps(payload, ensure_ascii=False).encode())


@pytest.fixture(scope="module")
def servers():
    started = []
    for _ in range(2):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
    yield [(h.server_address[1], f"http://127.0.0.1:{h.server_address[1]}") for h in started]
    for h in started:
        h.shutdown()


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    Scripted.routes, Scripted.seen = {}, []
    monkeypatch.delenv("POLICY_API_TOKEN", raising=False)
    monkeypatch.delenv("POLICY_API_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# --- 1. redirects are refused ----------------------------------------------------------------------------

@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_is_unavailable_and_token_never_leaves(servers, code):
    (port_a, url_a), (port_b, url_b) = servers
    Scripted.routes[(port_a, "/policies/5001")] = lambda h: h.send(code, b"", {"Location": url_b + "/policies/5001"})
    Scripted.routes[(port_b, "/policies/5001")] = json_route(RECORD)
    repo = ApiPolicyRepository(url_a, token="tok-123")
    with pytest.raises(RepositoryUnavailable):
        repo.get("5001")
    assert all(port == port_a for port, _, _ in Scripted.seen)  # host B never contacted
    d = answer("5001 רוצה לבטל", repo).decision
    assert d.code == "DATA_SOURCE_UNAVAILABLE" and no_money(d)


def test_same_host_redirect_is_also_refused(servers):
    (port_a, url_a), _ = servers
    Scripted.routes[(port_a, "/policies/5001")] = lambda h: h.send(302, b"", {"Location": "/v2/policies/5001"})
    Scripted.routes[(port_a, "/v2/policies/5001")] = json_route(RECORD)
    with pytest.raises(RepositoryUnavailable):
        ApiPolicyRepository(url_a).get("5001")


# --- 2. one read per policy per question ----------------------------------------------------------------------

class CountingRepository:
    source_name = "fake-api"

    def __init__(self, policy, on_get=None, customer_version=None):
        self.policy, self.on_get, self.customer_version = policy, on_get, customer_version
        self.get_calls, self.customer_calls = 0, 0

    def get(self, policy_id):
        self.get_calls += 1
        if policy_id != self.policy.policy_id:
            return None
        return self.on_get(self.get_calls) if self.on_get else replace(self.policy)

    def by_customer(self, customer_id):
        self.customer_calls += 1
        if customer_id != self.policy.customer_id:
            return []
        return [self.customer_version or replace(self.policy)]

    def find_names_in(self, text):
        return []


@pytest.fixture(scope="module")
def base_policy():
    return CsvPolicyRepository().get("4471")


def test_each_policy_read_once(base_policy):
    repo = CountingRepository(base_policy)
    d = answer("4471 רוצה לבטל", repo).decision
    assert d.code == "CANCEL_ESTIMATE"
    assert repo.get_calls == 1 and repo.customer_calls == 1


def test_changing_source_is_read_once_and_consistent(base_policy):
    # Each get() would return a different premium; only the first read is ever used.
    repo = CountingRepository(base_policy, on_get=lambda n: replace(base_policy, monthly_premium=410.0 * n))
    d = answer("4471 רוצה לבטל", repo).decision
    assert repo.get_calls == 1 and d.numbers["grant"] == 492


@pytest.mark.parametrize("change", [dict(monthly_premium=999.0), dict(open_claims=1), dict(status="מבוטלת")])
def test_disagreeing_reads_are_unavailable(base_policy, change):
    repo = CountingRepository(base_policy, customer_version=replace(base_policy, **change))
    d = answer("4471 רוצה לבטל", repo).decision
    assert (d.outcome, d.code) == (Outcome.CANNOT_ANSWER, "DATA_SOURCE_UNAVAILABLE") and no_money(d)


def test_same_object_through_the_pipeline(base_policy, monkeypatch):
    seen = []
    real = assistant.evaluate_cancellation

    def spy(policy, as_of, repo):
        seen.append((policy, repo.get(policy.policy_id), repo.by_customer(policy.customer_id)[0]))
        return real(policy, as_of, repo)

    monkeypatch.setattr(assistant, "evaluate_cancellation", spy)
    answer("4471 רוצה לבטל", CountingRepository(base_policy))
    policy, again, from_customer = seen[0]
    assert policy is again is from_customer


# --- 3. API identity checks in by_customer() --------------------------------------------------------------------

@pytest.mark.parametrize("record", [dict(RECORD, policy_id=""), dict(RECORD, policy_id=None),
                                    dict(RECORD, customer_id="C-9999"), dict(RECORD, customer_id=None)])
def test_customer_records_must_match(servers, record):
    (port, url), _ = servers
    Scripted.routes[(port, "/customers/C-2001/policies")] = json_route([record])
    with pytest.raises(RepositoryUnavailable):
        ApiPolicyRepository(url).by_customer("C-2001")


def test_duplicate_api_records_are_conflicting(servers):
    (port, url), _ = servers
    Scripted.routes[(port, "/customers/C-2001/policies")] = json_route([RECORD, dict(RECORD, monthly_premium=300)])
    policies = ApiPolicyRepository(url).by_customer("C-2001")
    assert len(policies) == 2 and all(p.conflicting_records and p.data_issues for p in policies)
    d = answer("C-2001 רוצה לבטל", ApiPolicyRepository(url)).decision
    assert (d.code, d.policy_id) == ("CONFLICTING_RECORDS", "5001") and no_money(d)


def test_duplicate_seen_after_get_is_unavailable(servers):
    (port, url), _ = servers
    Scripted.routes[(port, "/policies/5001")] = json_route(RECORD)
    Scripted.routes[(port, "/customers/C-2001/policies")] = json_route([RECORD, dict(RECORD, monthly_premium=300)])
    d = answer("5001 רוצה לבטל", ApiPolicyRepository(url)).decision
    assert d.code == "DATA_SOURCE_UNAVAILABLE" and no_money(d)  # the two reads of 5001 disagree


# --- 4. identifier exception only in identifier context ------------------------------------------------------------

class FakeLLM:
    def __init__(self, text):
        self.responses = [json.dumps({"intents": ["general_policy"], "mentions_regulator": False}), text]
        self.calls = 0

    def complete(self, system, user, max_tokens, json_schema=None, timeout=None):
        self.calls += 1
        return self.responses[min(self.calls, 2) - 1]


@pytest.fixture(scope="module")
def csv_repo():
    return CsvPolicyRepository()


@pytest.mark.parametrize("text,code", [
    ("סכום ההחזר (₪): 4471 (02_החזרים.md §1).", "UNGROUNDED"),
    ("ההחזר הוא 4471 (ש״ח) (02_החזרים.md §1).", "UNGROUNDED"),
    ("לגבי פוליסה 4471, דמי הביטול ברכב הם 150 ₪ (04_חריגים.md §א).", "GENERAL_POLICY"),
    ("4471: דמי הביטול ברכב הם 150 ₪ (04_חריגים.md §א).", "UNGROUNDED"),
    ("בפוליסת 4471 דמי הביטול ברכב הם 150 ₪ (04_חריגים.md §א).", "GENERAL_POLICY"),
    ("ההחזר: [4471] (02_החזרים.md §1).", "UNGROUNDED"),
])
def test_identifier_context(csv_repo, text, code):
    d = answer("מה הנוהל להחזר על חיוב כפול בפוליסה 4471?", csv_repo, FakeLLM(text)).decision
    assert d.code == code


@pytest.mark.parametrize("text,code", [
    ("ללקוח C-1088 דמי הביטול ברכב הם 150 ₪ (04_חריגים.md §א).", "GENERAL_POLICY"),
    ("ללקוח 1088 דמי הביטול ברכב הם 150 ₪ (04_חריגים.md §א).", "GENERAL_POLICY"),
    ("ההחזר הוא 1088 (02_החזרים.md §1).", "UNGROUNDED"),
])
def test_customer_identifier_context(csv_repo, text, code):
    assert answer("מה הנוהל להחזר על חיוב כפול ללקוח C-1088?", csv_repo, FakeLLM(text)).decision.code == code


def test_prompt_forbids_repeating_ids():
    assert GENERAL_SYSTEM_PROMPT.endswith("Do not repeat policy or customer numbers from the question.")


# --- 5. any failure while requesting is unavailable --------------------------------------------------------------------

def test_incomplete_read_is_unavailable(servers):
    (port, url), _ = servers

    def truncated(h):
        h.send_response(200)
        h.send_header("Content-Length", "500")
        h.end_headers()
        h.wfile.write(b'{"policy_id": "50')
        h.wfile.flush()
        h.close_connection = True

    Scripted.routes[(port, "/policies/5001")] = truncated
    with pytest.raises(RepositoryUnavailable):
        ApiPolicyRepository(url, timeout=1.0).get("5001")


@pytest.mark.parametrize("base", ["http://bad host", "http://127.0.0.1:99999", "http://[::1", "nonsense"])
def test_bad_base_url_is_unavailable(base):
    with pytest.raises(RepositoryUnavailable):
        ApiPolicyRepository(base, timeout=0.5).get("4471")


@pytest.mark.parametrize("url", ["http://", "http://host:99999", "http://ho st", "not a url", "http://[::1",
                                 "file:///etc/passwd", "ftp://x"])
def test_cli_rejects_bad_urls(capsys, url):
    assert main(["4471", "--source", "api", "--api-url", url, "--no-llm"]) == 1
    err = capsys.readouterr().err
    assert err.strip() == "שגיאה: כתובת API לא תקינה" and "Traceback" not in err


# --- 6. token validation; the token never appears in output ----------------------------------------------------------

@pytest.mark.parametrize("token", ["secret\nInjected: 1", "sec ret", "secret\t", "secret\x00", "secret\r"])
def test_bad_token_is_refused_and_never_printed(servers, capsys, monkeypatch, token):
    _, url = servers[0]
    if "\x00" not in token:  # an environment variable cannot hold NUL; the constructor check below covers it
        check_cli_refuses(capsys, monkeypatch, url, token)
    with pytest.raises(ValueError) as info:
        ApiPolicyRepository(url, token=token)
    assert "secret" not in str(info.value)


def check_cli_refuses(capsys, monkeypatch, url, token):
    monkeypatch.setenv("POLICY_API_TOKEN", token)
    assert main(["4471 רוצה לבטל", "--source", "api", "--api-url", url, "--no-llm"]) == 1
    out, err = capsys.readouterr()
    assert err.strip() == "שגיאה: POLICY_API_TOKEN לא תקין" and out == ""
    assert "secret" not in out + err


def test_token_validity():
    assert token_is_valid("abc.DEF-123_~+/=") and not token_is_valid("") and not token_is_valid("a b")


# --- 7. cancellation verbs as whole words ------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["הלקוח מובטל", "הלקוח מובטלת ושואל", "הפוליסה בוטלה", "וכשהדירה תבוטל"])
def test_not_cancellation_verbs(text):
    s = keyword_signals(text)
    assert not s.cancel and not s.cancel_keyword_raw


@pytest.mark.parametrize("text", ["לבטל", "תבטלו לי", "ביטול", "כשמבטלים", "ולבטל", "ביטלה", "שיבטל"])
def test_cancellation_verbs(text):
    assert keyword_signals(text).cancel


def test_unemployed_customer_gets_no_estimate(csv_repo):
    d = answer("4471 הוא מובטל ורוצה לדעת מה מגיע לו", csv_repo).decision
    assert d.code not in ("CANCEL_ESTIMATE", "COOLING_OFF", "TRAVEL_BEFORE_TRIP") and no_money(d)


def test_unemployed_on_open_claim_is_not_a_cancel_signal(csv_repo):
    d = answer("4428 הוא מובטל, מה מצב התביעה?", csv_repo).decision
    assert d.code == "CLAIM_OPEN_STAGE_UNKNOWN"


def test_explicit_verb_still_cancels(csv_repo):
    assert answer("תבטלו לי את הפוליסה 4471", csv_repo).decision.code == "CANCEL_ESTIMATE"
