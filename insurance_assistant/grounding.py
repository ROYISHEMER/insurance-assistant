"""Every number an LLM writes must already appear in the deterministic source text."""
from __future__ import annotations

import re

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


# A number written next to a currency token (optionally one space) is money.
CURRENCY = r"(?:₪|ש\"ח|ש״ח|שח(?![א-ת])|שקל|NIS(?![A-Za-z]))"
_AMOUNT = re.compile(rf"(\d+(?:\.\d+)?) ?{CURRENCY}|{CURRENCY} ?(\d+(?:\.\d+)?)", re.IGNORECASE)


def _canonical(number: str) -> str:
    if "." in number:
        number = number.rstrip("0").rstrip(".")  # "342.00" and "342" are the same number
    return number


def without_thousands(text: str) -> str:
    return _THOUSANDS.sub("", text)


def numbers_in(text: str) -> set[str]:
    return {_canonical(n) for n in _NUMBER.findall(_THOUSANDS.sub("", text))}


def currency_amounts(text: str) -> set[str]:
    return {_canonical(a or b) for a, b in _AMOUNT.findall(_THOUSANDS.sub("", text))}


def ungrounded_numbers(output: str, source: str) -> list[str]:
    return sorted(numbers_in(output) - numbers_in(source))
