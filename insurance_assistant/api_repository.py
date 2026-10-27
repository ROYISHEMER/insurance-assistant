"""Example adapter for the production policy system – for an ASSUMED contract.

The real API format is unknown (an open question for Ronit). This adapter shows how production swaps the
CSV export for an API without touching the rules: it implements PolicyRepository, and every field goes
through the same parsers as the CSV (repository.policy_from_values), so dates, numbers and data issues
behave identically. The pipeline still wraps it in ValidatedPolicyRepository.

Assumed contract:
- GET {base}/policies/{policy_id} -> 200 with one JSON object whose fields have the CSV column names
  (policy_id, customer_id, customer_name, product, start_date "YYYY-MM-DD", monthly_premium, status,
  last_payment_date, open_claims, notes); 404 -> the policy does not exist.
- GET {base}/customers/{customer_id}/policies -> 200 with a JSON list of such objects (an empty list or
  404 -> no policies).
- No free-text name search: find_names_in() returns [], so a customer NAME in a question resolves only
  with the CSV source. Limitation.
- Optional bearer token from env POLICY_API_TOKEN, sent as "Authorization: Bearer ..." (never logged).

Unavailability is never "not found": a timeout, connection error, HTTP 5xx, any status other than
200/404 (redirects are refused, never followed), any exception while building, sending or reading the
request, invalid JSON, a response of the wrong shape, or a record for a different policy/customer raises
RepositoryUnavailable. Two records with the same policy id are both flagged conflicting_records.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import unicodedata

from .models import Policy
from .repository import COLUMNS, RepositoryUnavailable, policy_from_values

DEFAULT_TIMEOUT = 3.0


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the Authorization header must never reach another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError for the 3xx


_OPENER = urllib.request.build_opener(_RefuseRedirects)


def token_is_valid(token: str) -> bool:
    """No whitespace and no control characters (a newline would split the HTTP header)."""
    return bool(token) and not any(ch.isspace() or unicodedata.category(ch).startswith("C") for ch in token)


def _raw(value: object) -> str:
    """A JSON value as the text the CSV would hold, so the same parsers judge it."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        try:
            return str(value)  # True -> "True", 2.0 -> "2.0", inf -> "inf": rejected exactly like the CSV text
        except ValueError:  # an integer too large to print
            return "<too large>"
    return "<invalid>"  # a list or an object where a value is expected


class ApiPolicyRepository:
    source_name = "policy-api"

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._token = token if token is not None else os.environ.get("POLICY_API_TOKEN")
        if self._token is not None and not token_is_valid(self._token):
            raise ValueError("POLICY_API_TOKEN is not valid")  # never include the token itself

    def __repr__(self) -> str:  # never show the token
        return f"ApiPolicyRepository({self.base_url!r})"

    # --- HTTP ------------------------------------------------------------------------------

    def _get_json(self, path: str):
        """(found, data). 404 -> (False, None); anything unexpected -> RepositoryUnavailable.

        Error messages carry only a status code or an exception TYPE, never exception text: that text can
        echo request headers, and so the token."""
        try:
            headers = {"Accept": "application/json"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            request = urllib.request.Request(self.base_url + path, headers=headers, method="GET")
            with _OPENER.open(request, timeout=self.timeout) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as exc:  # includes every refused 3xx redirect
            if exc.code == 404:
                return False, None
            raise RepositoryUnavailable(f"policy API returned HTTP {exc.code}") from None
        except Exception as exc:  # timeout, connection, IncompleteRead, InvalidURL, UnicodeError, ...
            raise RepositoryUnavailable(f"policy API request failed: {type(exc).__name__}") from None
        if status != 200:
            raise RepositoryUnavailable(f"policy API returned HTTP {status}")
        try:
            return True, json.loads(body.decode("utf-8"))
        except Exception:  # UnicodeDecodeError, ValueError, RecursionError on absurd nesting, ...
            raise RepositoryUnavailable("policy API returned invalid JSON") from None

    def _policy(self, record: object) -> Policy:
        if not isinstance(record, dict):
            raise RepositoryUnavailable("policy API returned a record that is not an object")
        for key in ("policy_id", "customer_id"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise RepositoryUnavailable(f"policy API returned a {key} that is not a string or an integer")
        return policy_from_values({column: _raw(record.get(column)) for column in COLUMNS}, self.source_name)

    # --- PolicyRepository --------------------------------------------------------------------

    def get(self, policy_id: str) -> Policy | None:
        key = str(policy_id).strip()
        found, data = self._get_json(f"/policies/{urllib.parse.quote(key, safe='')}")
        if not found:
            return None
        policy = self._policy(data)
        if policy.policy_id != key:
            # Never answer about a different policy than the one asked for.
            raise RepositoryUnavailable("policy API returned a different policy than requested")
        return policy

    def by_customer(self, customer_id: str) -> list[Policy]:
        key = str(customer_id).strip()
        found, data = self._get_json(f"/customers/{urllib.parse.quote(key, safe='')}/policies")
        if not found:
            return []
        if not isinstance(data, list):
            raise RepositoryUnavailable("policy API returned something other than a list of policies")
        policies = [self._policy(record) for record in data]
        if any(not p.policy_id or p.customer_id != key for p in policies):
            raise RepositoryUnavailable("policy API returned a record without a policy id or of another customer")
        # The same policy twice: flag both, exactly like duplicate CSV rows.
        counts = {}
        for p in policies:
            counts[p.policy_id] = counts.get(p.policy_id, 0) + 1
        for p in policies:
            if counts[p.policy_id] > 1:
                p.conflicting_records = True
                p.data_issues.append(f"policy_id: מספר הפוליסה מופיע ב-{counts[p.policy_id]} רשומות – "
                                     "הרשומות סותרות, יש לבדוק במערכת")
        return policies

    def find_names_in(self, text: str) -> list[Policy]:
        return []  # the assumed contract has no name search
