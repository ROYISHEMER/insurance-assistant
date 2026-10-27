"""The four procedure documents: startup checks and loading."""
from __future__ import annotations

import re
from pathlib import Path

from .models import StartupError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "policies"
POLICY_DOC_FILES = (
    "01_תנאי_ביטול.md",
    "02_החזרים.md",
    "03_טיפול_בתביעות.md",
    "04_חריגים.md",
)
# "## 1. כללי" or "## א. דמי ביטול ..."
_SECTION_HEADING = re.compile(r"^##[ \t]+(\d+|[א-ת]{1,2})\.[ \t]+\S", re.MULTILINE)


def _read_doc(path: Path) -> str:
    if not path.is_file():
        raise StartupError(f"מסמך נוהל חסר: {path.name} (בתיקייה {path.parent})")
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except OSError as exc:
        raise StartupError(f"לא ניתן לקרוא את מסמך הנוהל {path.name}: {exc.strerror or exc}") from None
    except UnicodeDecodeError:
        raise StartupError(f"מסמך הנוהל {path.name} אינו בקידוד UTF-8") from None
    if not _SECTION_HEADING.search(text):
        raise StartupError(
            f"במסמך הנוהל {path.name} לא נמצאה אף כותרת סעיף בפורמט '## <סעיף>. ...'"
        )
    return text


def check_policy_docs(directory: str | Path = DEFAULT_DOCS_DIR) -> None:
    load_policy_docs(directory)


def load_policy_docs(directory: str | Path = DEFAULT_DOCS_DIR) -> dict[str, str]:
    folder = Path(directory)
    if not folder.is_dir():
        raise StartupError(f"תיקיית הנהלים לא נמצאה: {folder}")
    return {name: _read_doc(folder / name) for name in POLICY_DOC_FILES}


_SECTION_START = re.compile(r"^##[ \t]+(\d+|[א-ת]{1,2})\.", re.MULTILINE)
_NEXT_HEADING = re.compile(r"^#{1,2}[ \t]", re.MULTILINE)
_SUBSECTION_START = re.compile(r"^\*\*([א-ת])\.", re.MULTILINE)


def split_sections(text: str) -> dict[str, str]:
    """Sections keyed "1" / "א", plus lettered subsections keyed "2ג" (the bold "**ג.**" paragraphs)."""
    sections: dict[str, str] = {}
    for match in _SECTION_START.finditer(text):
        end = _NEXT_HEADING.search(text, match.end())
        body = text[match.start():end.start() if end else len(text)].strip()
        key = match.group(1)
        sections[key] = body
        subs = list(_SUBSECTION_START.finditer(body))
        for i, sub in enumerate(subs):
            stop = subs[i + 1].start() if i + 1 < len(subs) else len(body)
            sections[key + sub.group(1)] = body[sub.start():stop].strip()
    return sections
