from __future__ import annotations

import os
import re
import time

from openai import OpenAI


_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class OpenAIText2SQLClient:
    def __init__(
        self,
        model: str,
        reasoning_effort: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = OpenAI(api_key=key)

    def generate_sql(self, system_prompt: str, user_prompt: str) -> tuple[str, str, float]:
        started = time.perf_counter()
        kwargs = dict(
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        # I don't know which model uses temperature, guess I can remove it
        response = self._client.responses.create(**kwargs)
        elapsed = time.perf_counter() - started

        raw_text = response.output_text or ""
        sql = extract_sql(raw_text)
        return sql, raw_text, elapsed


def extract_sql(text: str) -> str:
    candidate = text.strip()

    code_match = _CODE_BLOCK_RE.search(candidate)
    if code_match:
        candidate = code_match.group(1).strip()

    # Keep only the first statement for safer execution.
    if ";" in candidate:
        statements = [part.strip() for part in candidate.split(";") if part.strip()]
        if statements:
            candidate = statements[0]

    return candidate.strip()
