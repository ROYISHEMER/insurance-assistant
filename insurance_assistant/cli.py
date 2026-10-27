"""Command line: python -m insurance_assistant.cli ["question"] [--as-of YYYY-MM-DD] [--no-llm] [--debug]
[--report] [--source csv|api] [--api-url URL]"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import date
from enum import Enum
from pathlib import Path

from .api_repository import ApiPolicyRepository, token_is_valid
from .assistant import answer
from .llm import get_llm
from .models import DEFAULT_AS_OF, StartupError
from .policy_docs import DEFAULT_DOCS_DIR, check_policy_docs
from .repository import CsvPolicyRepository

_AS_OF = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_as_of(raw: str | None) -> date:
    """Strict zero-padded YYYY-MM-DD, a real date, year 2000-2100. Default: the export date, never today."""
    if raw is None:
        return DEFAULT_AS_OF
    error = StartupError(f"תאריך --as-of לא תקין: '{raw}' (נדרש YYYY-MM-DD, שנה 2000–2100)")
    if not _AS_OF.fullmatch(raw):
        raise error
    try:
        value = date.fromisoformat(raw)
    except ValueError:
        raise error from None
    if not 2000 <= value.year <= 2100:
        raise error
    return value


def load_sources(csv_path: str | Path = "data/policies.csv",
                 policies_dir: str | Path = DEFAULT_DOCS_DIR) -> CsvPolicyRepository:
    """Startup checks shared by the CLI and the eval scripts. Raises StartupError."""
    check_policy_docs(policies_dir)
    return CsvPolicyRepository(csv_path)


def load_api_source(api_url: str | None, policies_dir: str | Path = DEFAULT_DOCS_DIR) -> ApiPolicyRepository:
    """Startup for --source api: the document checks still run; the CSV checks are skipped."""
    check_policy_docs(policies_dir)
    if not api_url or not api_url.strip():
        raise StartupError("עם --source api יש לציין --api-url או את משתנה הסביבה POLICY_API_URL")
    url = api_url.strip()
    try:
        parts = urllib.parse.urlsplit(url)
        parts.port  # raises ValueError on a bad port
        if parts.scheme not in ("http", "https") or not parts.hostname or any(ch.isspace() for ch in url):
            raise ValueError
        urllib.request.Request(url + "/policies/0")
    except Exception:
        raise StartupError("כתובת API לא תקינה") from None
    token = os.environ.get("POLICY_API_TOKEN")
    if token is not None and not token_is_valid(token):
        raise StartupError("POLICY_API_TOKEN לא תקין")  # never echo the token
    return ApiPolicyRepository(url, token=token)


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def _run(question: str, repo, llm, as_of: date, policies_dir, debug: bool) -> None:
    response = answer(question, repo, llm=llm, as_of=as_of, policies_dir=policies_dir)
    print(response.text)
    print()
    print(f"זמן: {response.latency_ms} ms | מסלול: {response.path}")
    if debug:
        print(json.dumps(asdict(response.decision), ensure_ascii=False, indent=2, default=_json_default))


def main(argv: list[str] | None = None, csv_path: str | Path = "data/policies.csv",
         policies_dir: str | Path = DEFAULT_DOCS_DIR) -> int:
    parser = argparse.ArgumentParser(prog="insurance_assistant.cli", description="עוזר לנציגי שירות")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--as-of")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--source", choices=("csv", "api"), default="csv")
    parser.add_argument("--api-url", help="policy API base URL (or env POLICY_API_URL)")
    args = parser.parse_args(argv)

    try:
        as_of = parse_as_of(args.as_of)
        if args.source == "api":
            if args.report:
                raise StartupError("--report זמין רק עם --source csv")
            repo = load_api_source(args.api_url or os.environ.get("POLICY_API_URL"), policies_dir)
        else:
            repo = load_sources(csv_path, policies_dir)
    except StartupError as exc:
        print(f"שגיאה: {exc}", file=sys.stderr)
        return 1

    if args.question is not None and not args.question.strip():
        print("השאלה ריקה", file=sys.stderr)
        return 1

    if args.report:
        print(json.dumps(repo.data_quality_report(), ensure_ascii=False, indent=2, default=_json_default))
        if args.question is None:
            return 0

    llm = None if args.no_llm else get_llm()
    if args.question is not None:
        _run(args.question, repo, llm, as_of, policies_dir, args.debug)
        return 0

    print("הקלד שאלה (שורה ריקה או Ctrl-D ליציאה)")
    while True:
        try:
            question = input("שאלה> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question.strip():
            return 0
        _run(question, repo, llm, as_of, policies_dir, args.debug)
        print()


if __name__ == "__main__":
    sys.exit(main())
