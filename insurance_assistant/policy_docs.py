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
