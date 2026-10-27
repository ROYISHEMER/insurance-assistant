"""Thin Anthropic client. Any failure becomes LLMError; callers fall back to code."""
from __future__ import annotations

import os

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_TIMEOUT = 4.0  # seconds: the customer is on the line
_OK_STOP_REASONS = ("end_turn", "stop_sequence")


class LLMError(Exception):
    pass


class AnthropicLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        if client is None:
            import anthropic  # lazy: the app runs without the SDK in keyword mode

            client = anthropic.Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT, max_retries=0)
        self._client = client

    def complete(self, system: str, user: str, max_tokens: int,
                 json_schema: dict | None = None, timeout: float | None = None) -> str:
        # SDK 2.3.0 has no temperature parameter; effort and structured output go in output_config.
        output_config: dict = {"effort": "low"}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            response = self._client.messages.create(**kwargs)
            stop_reason = response.stop_reason
            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        except Exception as exc:
            raise LLMError(f"LLM call failed: {type(exc).__name__}") from exc
        if stop_reason == "refusal":
            raise LLMError("LLM refused")
        if stop_reason == "max_tokens":
            raise LLMError("LLM output truncated (max_tokens)")
        if stop_reason not in _OK_STOP_REASONS:
            raise LLMError(f"unexpected stop reason: {stop_reason!r}")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("empty LLM response")
        return text


def get_llm() -> AnthropicLLM | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicLLM(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception:
        return None
