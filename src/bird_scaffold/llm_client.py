from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from bird_scaffold.query_tool import (
    QUERY_TOOL_NAME,
    QueryToolConfig,
    QueryToolExecutor,
    build_query_tool_schema,
    parse_query_tool_arguments,
)
from bird_scaffold.sql_parsing import extract_sql as _extract_sql


@dataclass(frozen=True)
class _FunctionCall:
    call_id: str
    name: str
    arguments: str | dict[str, Any] | None


class OpenAICompatibleText2SQLClient:
    def __init__(
        self,
        model: str,
        *,
        api_base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        query_tool_enabled: bool = True,
        query_tool_max_calls: int = 8,
        query_tool_max_rows: int = 50,
        query_tool_max_output_chars: int = 6000,
        query_tool_max_cell_chars: int = 200,
        query_tool_timeout_seconds: float = 8.0,
    ) -> None:
        resolved_base_url = _normalize_base_url(api_base_url)
        resolved_key = _resolve_api_key(
            explicit_key=api_key,
            has_custom_base_url=resolved_base_url is not None,
        )

        self.model = model
        self.api_base_url = resolved_base_url
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.query_tool_enabled = query_tool_enabled
        self.query_tool_config = QueryToolConfig(
            max_calls=query_tool_max_calls,
            max_rows=query_tool_max_rows,
            max_output_chars=query_tool_max_output_chars,
            max_cell_chars=query_tool_max_cell_chars,
            timeout_seconds=query_tool_timeout_seconds,
        ).validated()

        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)

    def generate_sql(
        self,
        system_prompt: str,
        user_prompt: str,
        db_path: Path | None = None,
    ) -> tuple[str, str, float, int]:
        started = time.perf_counter()
        query_tool_executor = self._build_query_tool_executor(db_path=db_path)
        tools = [build_query_tool_schema()] if query_tool_executor is not None else None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        query_tool_calls = 0
        raw_text = ""
        tool_rounds = 0
        max_tool_rounds = self.query_tool_config.max_calls + 4

        while True:
            try:
                response = self._create_completion(messages=messages, tools=tools)
            except Exception as exc:
                if tools and _is_tool_calling_unsupported_error(exc):
                    tools = None
                    query_tool_executor = None
                    response = self._create_completion(messages=messages, tools=None)
                else:
                    raise

            assistant_message = _extract_assistant_message(response)
            raw_text = _extract_message_text(assistant_message)
            function_calls = _extract_function_calls_from_message(assistant_message)

            if not function_calls or query_tool_executor is None:
                break
            if tool_rounds >= max_tool_rounds:
                break

            tool_rounds += 1
            messages.append(_assistant_message_to_dict(assistant_message))

            for function_call in function_calls:
                if function_call.name != QUERY_TOOL_NAME:
                    output_text = _tool_error_json(f"Unsupported tool '{function_call.name}'.")
                elif query_tool_calls >= self.query_tool_config.max_calls:
                    output_text = _tool_error_json(
                        f"Maximum query tool calls reached ({self.query_tool_config.max_calls}). "
                        "Return final SQL now."
                    )
                else:
                    query_tool_calls += 1
                    sql_argument = parse_query_tool_arguments(function_call.arguments)
                    output_text = query_tool_executor.execute(sql_argument)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": function_call.call_id,
                        "name": function_call.name,
                        "content": output_text,
                    }
                )

        elapsed = time.perf_counter() - started
        sql = _extract_sql(raw_text)
        return sql, raw_text, elapsed, query_tool_calls

    def _build_query_tool_executor(self, db_path: Path | None) -> QueryToolExecutor | None:
        if not self.query_tool_enabled:
            return None
        if db_path is None:
            return None
        return QueryToolExecutor(db_path=db_path, config=self.query_tool_config)

    def _create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature != 0.0:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.reasoning_effort and _looks_like_openai_base_url(self.api_base_url):
            kwargs["reasoning_effort"] = self.reasoning_effort

        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "reasoning_effort" in kwargs and _is_reasoning_parameter_error(exc):
                kwargs.pop("reasoning_effort", None)
                return self._client.chat.completions.create(**kwargs)
            raise


def extract_sql(text: str) -> str:
    return _extract_sql(text)


def _resolve_api_key(*, explicit_key: str | None, has_custom_base_url: bool) -> str:
    if explicit_key:
        return explicit_key

    env_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY")
    if env_key:
        return env_key

    if has_custom_base_url:
        # vLLM often runs with --api-key disabled; OpenAI SDK still needs a non-empty value.
        return "EMPTY"

    raise ValueError(
        "Missing API key. Set OPENAI_API_KEY (or VLLM_API_KEY), or pass --api-key explicitly."
    )


def _normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


def _extract_assistant_message(response: Any) -> Any:
    choices = _get(response, "choices") or []
    if not choices:
        raise RuntimeError("Model returned no choices.")

    first = choices[0]
    message = _get(first, "message")
    if message is None:
        raise RuntimeError("Model response is missing an assistant message.")
    return message


def _extract_message_text(message: Any) -> str:
    content = _get(message, "content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    parts.append(part)
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts)

    return ""


def _extract_function_calls_from_message(message: Any) -> list[_FunctionCall]:
    calls: list[_FunctionCall] = []
    tool_calls = _get(message, "tool_calls") or []
    for item in tool_calls:
        call_id = _get(item, "id")
        if not isinstance(call_id, str):
            continue

        function = _get(item, "function")
        name = _get(function, "name")
        arguments = _get(function, "arguments")
        if isinstance(name, str):
            calls.append(_FunctionCall(call_id=call_id, name=name, arguments=arguments))
    return calls


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    content = _extract_message_text(message)
    if content:
        payload["content"] = content

    function_calls = _extract_function_calls_from_message(message)
    if function_calls:
        payload["tool_calls"] = [
            {
                "id": function_call.call_id,
                "type": "function",
                "function": {
                    "name": function_call.name,
                    "arguments": _normalize_tool_arguments(function_call.arguments),
                },
            }
            for function_call in function_calls
        ]
    return payload


def _normalize_tool_arguments(arguments: str | dict[str, Any] | None) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    return "{}"


def _get(item: Any, field: str) -> Any:
    value = getattr(item, field, None)
    if value is not None:
        return value
    if isinstance(item, dict):
        return item.get(field)
    return None


def _tool_error_json(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _looks_like_openai_base_url(base_url: str | None) -> bool:
    if base_url is None:
        return True
    lowered = base_url.lower()
    return "api.openai.com" in lowered


def _is_tool_calling_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "tool",
        "function",
        "tool_choice",
        "unsupported",
        "not implemented",
        "extra_forbidden",
    ]
    return any(marker in text for marker in markers)


def _is_reasoning_parameter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "reasoning_effort" in text and ("unknown" in text or "unsupported" in text or "extra" in text)
