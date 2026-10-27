"""Every number an LLM writes must already appear in the deterministic source text."""
from __future__ import annotations

import re

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _canonical(number: str) -> str:
    if "." in number:
        number = number.rstrip("0").rstrip(".")  # "342.00" and "342" are the same number
    return number


def numbers_in(text: str) -> set[str]:
    return {_canonical(n) for n in _NUMBER.findall(_THOUSANDS.sub("", text))}


def ungrounded_numbers(output: str, source: str) -> list[str]:
    return sorted(numbers_in(output) - numbers_in(source))
