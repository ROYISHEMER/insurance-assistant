"""Parse raw text values. Never raise, never guess: an unusable value becomes None plus a Hebrew note."""
from __future__ import annotations

import math
import re
from datetime import date

_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DMY_DATE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
_INT = re.compile(r"[+-]?\d+")
_FLOAT = re.compile(r"[+-]?\d+(\.\d+)?")
MAX_NUMBER_LENGTH = 20


def _shown(raw: object) -> str:
    try:
        text = str(raw)
    except Exception:
        return "?"
    return text if len(text) <= 30 else text[:30] + "…"


def _clean(raw: object, field: str, issues: list[str]) -> str | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        issues.append(f"{field}: ערך חסר")
        return None
    if not isinstance(raw, str):
        issues.append(f"{field}: ערך לא תקין '{_shown(raw)}'")
        return None
    return raw.strip()


def parse_date(raw: object, field: str, issues: list[str]) -> date | None:
    """ISO YYYY-MM-DD first, then exactly DD/MM/YYYY (Israeli day/month, noted in issues)."""
    try:
        text = _clean(raw, field, issues)
        if text is None:
            return None
        if m := _ISO_DATE.fullmatch(text):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := _DMY_DATE.fullmatch(text):
            value = date(int(m[3]), int(m[2]), int(m[1]))
            issues.append(
                f"{field}: התאריך '{text}' נקרא כיום/חודש (DD/MM/YYYY) ונורמל ל-{value.isoformat()}"
            )
            return value
    except Exception:
        pass
    issues.append(f"{field}: תאריך לא תקין '{_shown(raw)}'")
    return None


def _number_text(raw: object, field: str, issues: list[str], pattern: re.Pattern) -> str | None:
    text = _clean(raw, field, issues)
    if text is None:
        return None
    if len(text) > MAX_NUMBER_LENGTH:
        issues.append(f"{field}: ערך ארוך מדי ({len(text)} תווים)")
        return None
    if not pattern.fullmatch(text):
        issues.append(f"{field}: מספר לא תקין '{_shown(text)}'")
        return None
    return text


def parse_int(raw: object, field: str, issues: list[str]) -> int | None:
    try:
        text = _number_text(raw, field, issues, _INT)
        return None if text is None else int(text)
    except Exception:
        issues.append(f"{field}: מספר לא תקין '{_shown(raw)}'")
        return None


def parse_float(raw: object, field: str, issues: list[str]) -> float | None:
    try:
        text = _number_text(raw, field, issues, _FLOAT)
        if text is None:
            return None
        value = float(text)
        if not math.isfinite(value):
            issues.append(f"{field}: ערך לא סופי '{_shown(text)}'")
            return None
        return value
    except Exception:
        issues.append(f"{field}: מספר לא תקין '{_shown(raw)}'")
        return None
