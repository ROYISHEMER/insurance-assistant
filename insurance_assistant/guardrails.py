"""Checks that run by code before any LLM call."""
from __future__ import annotations

import re

_DASHES = re.compile(r"[-־–—]")  # hyphen, Hebrew maqaf, en dash, em dash
_QUOTES = re.compile(r"['\"׳״]")  # ' " geresh gershayim
_SPACES = re.compile(r"\s+")

# Matched as plain substrings after normalization – no prefix-letter limit, no word start – so any
# inflection ("ושהממונה על הביטוח", "וכשהמפקח", "לרשות שוק ההון") matches. Deliberate over-escalation:
# a needless referral to the shift manager is cheap, a missed regulator complaint is not (04_חריגים.md §ה).
REGULATOR_TERMS = (
    "רשות שוק ההון",
    "שוק ההון",
    "רגולטור",
    "ממונה על הביטוח",
    "מפקח על הביטוח",
    "פיקוח על הביטוח",
    "מפקח",  # bare word on purpose: also matches "מפקח בנייה"
)


def normalize_text(text: str) -> str:
    """Dashes become a space (not removed), quotes are removed, spaces collapsed."""
    text = _DASHES.sub(" ", text)
    text = _QUOTES.sub("", text)
    return _SPACES.sub(" ", text).strip()


def mentions_regulator(text: str) -> bool:
    normalized = normalize_text(text)
    return any(term in normalized for term in REGULATOR_TERMS)
