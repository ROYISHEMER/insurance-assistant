# Insurance rep assistant – architecture rules
- Python 3.11+, standard library + anthropic + pytest only. No pandas, no vector DB, no frameworks.
- The LLM NEVER computes money, dates or escalation decisions.
- All policy data goes through the PolicyRepository protocol.
- Missing data is reported explicitly, never guessed. Every fact cites its source.
- as_of defaults to 2026-10-25 (the CSV export date), never today's date.
- Never change eval/cases.json expectations to make a test pass – report instead.
- Only read and write files inside this repository, plus the prompt file you are given.
- Run tests with: python -m pytest -q (inside .venv)
