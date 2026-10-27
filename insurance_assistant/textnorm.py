"""The one text normalization applied before any matching."""
from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r" {2,}")


def normalize_unicode(text: str) -> str:
    """Every Unicode whitespace -> one ASCII space (collapsed); every dash (category Pd) and the minus
    sign U+2212 -> ASCII "-"."""
    out = []
    for ch in text:
        if ch.isspace() or unicodedata.category(ch) == "Zs":
            out.append(" ")
        elif unicodedata.category(ch) == "Pd" or ch == "−":
            out.append("-")
        else:
            out.append(ch)
    return _SPACES.sub(" ", "".join(out))
