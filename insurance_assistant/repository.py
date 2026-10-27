"""Policy data access. Everything outside this module sees policies only through PolicyRepository."""
from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import Policy, StartupError
from .normalize import parse_date, parse_float, parse_int

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_MONTHLY_PREMIUM = 100_000
MAX_OPEN_CLAIMS = 1_000  # sanity bound so absurd API values match the CSV parser's length limit
COLUMNS = (
    "policy_id", "customer_id", "customer_name", "product", "start_date",
    "monthly_premium", "status", "last_payment_date", "open_claims", "notes",
)
REQUIRED_COLUMNS = tuple(c for c in COLUMNS if c != "notes")
_TEXT_FIELDS = ("policy_id", "customer_id", "customer_name", "product", "status")


@runtime_checkable
class PolicyRepository(Protocol):
    source_name: str

    def get(self, policy_id: str) -> Policy | None: ...

    def by_customer(self, customer_id: str) -> list[Policy]: ...

    def find_names_in(self, text: str) -> list[Policy]:
        """Policies whose full customer_name appears in the text."""
        ...


# --- validation -------------------------------------------------------------

def _shown(value: object) -> str:
    try:
        text = repr(value)
    except Exception:
        return "?"
    return text if len(text) <= 30 else text[:30] + "…"


def validate_policy(policy: Policy, source_name: str) -> Policy:
    """Return a checked copy of policy; never mutates the input and never raises."""
    try:
        issues = [str(i) for i in policy.data_issues or []]
    except Exception:
        issues = ["data_issues: ערך לא תקין"]

    def add(message: str) -> None:
        if message not in issues:
            issues.append(message)

    def missing(field: str) -> None:
        # A parser may already have explained why the value is missing.
        if not any(i.startswith(f"{field}:") for i in issues):
            add(f"{field}: ערך חסר")

    def invalid(field: str, value: object, consequence: str) -> None:
        add(f"{field}: ערך לא תקין ({_shown(value)}) – {consequence}")

    def text(field: str) -> str:
        try:
            value = getattr(policy, field)
            if value is None:
                missing(field)
                return ""
            if isinstance(value, int) and not isinstance(value, bool):
                value = str(value)
            if isinstance(value, str):
                if not value.strip():
                    missing(field)
                return value.strip()
            invalid(field, value, "הערך הושמט")
        except Exception:
            add(f"{field}: ערך לא תקין – הערך הושמט")
        return ""

    def a_date(field: str) -> date | None:
        try:
            value = getattr(policy, field)
            if value is None:
                missing(field)
                return None
            if isinstance(value, date) and not isinstance(value, datetime):
                return value
            invalid(field, value, "התאריך נחשב חסר")
        except Exception:
            add(f"{field}: ערך לא תקין – התאריך נחשב חסר")
        return None

    def open_claims() -> int | None:
        consequence = "מספר התביעות הפתוחות לא ידוע"
        try:
            value = policy.open_claims
            if value is None:
                missing("open_claims")
                return None
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_OPEN_CLAIMS:
                return value
            invalid("open_claims", value, consequence)
        except Exception:
            add(f"open_claims: ערך לא תקין – {consequence}")
        return None

    def monthly_premium() -> float | None:
        consequence = "הפרמיה החודשית נחשבת חסרה"
        try:
            value = policy.monthly_premium
            if value is None:
                missing("monthly_premium")
                return None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)  # OverflowError for e.g. 10**400
                if math.isfinite(number) and 0 <= number <= MAX_MONTHLY_PREMIUM:
                    return number
            invalid("monthly_premium", value, consequence)
        except Exception:
            add(f"monthly_premium: ערך לא תקין – {consequence}")
        return None

    fields = {name: text(name) for name in _TEXT_FIELDS}
    start_date = a_date("start_date")
    last_payment_date = a_date("last_payment_date")
    claims = open_claims()
    premium = monthly_premium()
    try:
        notes = policy.notes.strip() if isinstance(policy.notes, str) else ""
    except Exception:
        notes = ""
    try:
        conflicting = bool(policy.conflicting_records)
    except Exception:
        conflicting = False

    return Policy(
        start_date=start_date,
        monthly_premium=premium,
        last_payment_date=last_payment_date,
        open_claims=claims,
        notes=notes,
        data_issues=issues,
        source=source_name,
        conflicting_records=conflicting,
        **fields,
    )


class RepositoryUnavailable(Exception):
    """The data source could not answer (timeout, error, malformed response). Never means "not found"."""


def policy_from_values(values: dict[str, str], source_name: str) -> Policy:
    """Raw text values by CSV column name -> a validated Policy. Every source goes through these parsers."""
    issues: list[str] = []
    policy = Policy(
        policy_id=values["policy_id"].strip(),
        customer_id=values["customer_id"],
        customer_name=values["customer_name"],
        product=values["product"],
        start_date=parse_date(values["start_date"], "start_date", issues),
        monthly_premium=parse_float(values["monthly_premium"], "monthly_premium", issues),
        status=values["status"],
        last_payment_date=parse_date(values["last_payment_date"], "last_payment_date", issues),
        open_claims=parse_int(values["open_claims"], "open_claims", issues),
        notes=values.get("notes", ""),
        data_issues=issues,
    )
    return validate_policy(policy, source_name)


class ValidatedPolicyRepository:
    """Wraps any repository so that every policy it returns has been validated."""

    def __init__(self, repo: PolicyRepository):
        self._repo = repo
        self.source_name = repo.source_name

    def get(self, policy_id: str) -> Policy | None:
        policy = self._repo.get(policy_id)
        return None if policy is None else validate_policy(policy, self.source_name)

    def by_customer(self, customer_id: str) -> list[Policy]:
        return [validate_policy(p, self.source_name) for p in self._repo.by_customer(customer_id)]

    def find_names_in(self, text: str) -> list[Policy]:
        return [validate_policy(p, self.source_name) for p in self._repo.find_names_in(text)]


# --- CSV pilot export ---------------------------------------------------------

def _copy(policy: Policy) -> Policy:
    return replace(policy, data_issues=list(policy.data_issues))


class CsvPolicyRepository:
    """The only code that reads the CSV export."""

    def __init__(self, path: str | Path = "data/policies.csv"):
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path
        self.path = file_path
        self.source_name = file_path.name
        self._rows: list[tuple[int, Policy]] = []  # (data row number, policy)
        self._skipped: list[dict] = []
        self._load()

    # Lookups
    def get(self, policy_id: str) -> Policy | None:
        key = str(policy_id).strip()
        for _, policy in self._rows:
            if policy.policy_id == key:
                return _copy(policy)
        return None

    def by_customer(self, customer_id: str) -> list[Policy]:
        key = str(customer_id).strip()
        return [_copy(p) for _, p in self._rows if p.customer_id == key]

    def find_names_in(self, text: str) -> list[Policy]:
        if not isinstance(text, str):
            return []
        return [_copy(p) for _, p in self._rows if p.customer_name and p.customer_name in text]

    def policy_ids(self) -> list[str]:
        """For tests only; not part of PolicyRepository."""
        return [p.policy_id for _, p in self._rows]

    def data_quality_report(self) -> dict:
        issues: dict[str, list[str]] = {}
        for _, policy in self._rows:
            if policy.data_issues:
                bucket = issues.setdefault(policy.policy_id, [])
                bucket.extend(i for i in policy.data_issues if i not in bucket)
        ids_by_name: dict[str, set[str]] = defaultdict(set)
        for _, policy in self._rows:
            if policy.customer_name and policy.customer_id:
                ids_by_name[policy.customer_name].add(policy.customer_id)
        return {
            "source": self.source_name,
            "policy_count": len(self._rows),
            "issues": issues,
            "conflicting_policy_ids": sorted({p.policy_id for _, p in self._rows if p.conflicting_records}),
            "skipped_rows": [dict(s) for s in self._skipped],
            # Same name under different customer ids: reported, never merged.
            "duplicate_names": {n: sorted(ids) for n, ids in sorted(ids_by_name.items()) if len(ids) > 1},
            "notes": {p.policy_id: p.notes for _, p in self._rows if p.notes},
        }

    # Loading
    def _fail(self, message: str) -> StartupError:
        return StartupError(f"קובץ הפוליסות {self.path.name}: {message}")

    def _read_text(self) -> str:
        if not self.path.is_file():
            raise StartupError(f"קובץ הפוליסות לא נמצא: {self.path}")
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise self._fail(f"לא ניתן לקרוא את הקובץ ({exc.strerror or exc})") from None
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise self._fail("הקובץ אינו בקידוד UTF-8") from None

    def _load(self) -> None:
        text = self._read_text()
        if not text.strip():
            raise self._fail("הקובץ ריק")
        reader = csv.reader(io.StringIO(text, newline=""))
        try:
            header = [h.strip() for h in next(reader)]
        except csv.Error as exc:
            raise self._fail(f"שורת הכותרת אינה תקינה ({exc})") from None
        self._check_header(header)

        row_number = 0  # data rows, counted here: csv line numbers drift with quoted fields
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                if "field larger than field limit" in str(exc):
                    raise self._fail(
                        f"בשורת נתונים {row_number + 1} יש שדה ארוך מהמותר ({csv.field_size_limit():,} תווים)"
                    ) from None
                raise self._fail(f"שורת נתונים {row_number + 1} אינה תקינה ({exc})") from None
            row_number += 1
            self._add_row(row_number, header, row)

        if row_number == 0:
            raise self._fail("יש בקובץ שורת כותרת בלבד, ללא נתונים")
        if not self._rows:
            raise self._fail(f"כל {row_number} שורות הנתונים דולגו (מספר פוליסה ריק או שורה ריקה)")
        self._mark_conflicts()

    def _check_header(self, header: list[str]) -> None:
        if any("\n" in h or "\r" in h for h in header):
            raise self._fail("שם עמודה בשורת הכותרת מכיל ירידת שורה")
        duplicates = sorted({h for h in header if header.count(h) > 1})
        if duplicates:
            raise self._fail(f"שמות עמודה כפולים בכותרת: {', '.join(duplicates)}")
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise self._fail(f"עמודות חובה חסרות בכותרת: {', '.join(missing)}")

    def _add_row(self, row_number: int, header: list[str], row: list[str]) -> None:
        if not row:
            self._skipped.append({"row": row_number, "reason": "שורה ריקה"})
            return
        if len(row) != len(header):
            raise self._fail(
                f"בשורת נתונים {row_number} יש {len(row)} שדות, בעוד שבכותרת יש {len(header)}"
            )
        if any("\n" in v or "\r" in v for v in row):
            raise self._fail(f"בשורת נתונים {row_number} יש שדה שמכיל ירידת שורה")
        values = dict(zip(header, row))
        policy_id = values["policy_id"].strip()
        if not policy_id:
            self._skipped.append({"row": row_number, "reason": "מספר פוליסה ריק"})
            return
        self._rows.append((row_number, policy_from_values(values, self.source_name)))

    def _mark_conflicts(self) -> None:
        rows_by_id: dict[str, list[int]] = defaultdict(list)
        for row_number, policy in self._rows:
            rows_by_id[policy.policy_id].append(row_number)
        for _, policy in self._rows:
            rows = rows_by_id[policy.policy_id]
            if len(rows) > 1:
                policy.conflicting_records = True
                policy.data_issues.append(
                    f"policy_id: מספר הפוליסה מופיע ב-{len(rows)} שורות "
                    f"({', '.join(map(str, rows))}) – הרשומות סותרות, יש לבדוק במערכת"
                )
