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
200/404, invalid JSON, a response of the wrong shape, or a record for a different policy/customer raises
RepositoryUnavailable.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .models import Policy
from .repository import COLUMNS, RepositoryUnavailable, policy_from_values

DEFAULT_TIMEOUT = 3.0


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

    def __repr__(self) -> str:  # never show the token
        return f"ApiPolicyRepository({self.base_url!r})"

    # --- HTTP ------------------------------------------------------------------------------

    def _get_json(self, path: str):
        """(found, data). 404 -> (False, None); anything unexpected -> RepositoryUnavailable."""
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(self.base_url + path, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False, None
            raise RepositoryUnavailable(f"policy API returned HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RepositoryUnavailable(f"policy API unreachable: {type(exc).__name__}: {reason}") from None
        if status != 200:
            raise RepositoryUnavailable(f"policy API returned HTTP {status}")
        try:
            return True, json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise RepositoryUnavailable("policy API returned invalid JSON") from None

    def _policy(self, record: object) -> Policy:
        if not isinstance(record, dict):
            raise RepositoryUnavailable("policy API returned a record that is not an object")
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
        if any(p.customer_id != key for p in policies):
            raise RepositoryUnavailable("policy API returned policies of a different customer")
        return policies

    def find_names_in(self, text: str) -> list[Policy]:
        return []  # the assumed contract has no name search
