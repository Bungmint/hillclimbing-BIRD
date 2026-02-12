from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langfuse.openai import openai

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

        self._client = openai.OpenAI(api_key=resolved_key, base_url=resolved_base_url)
        self._trace_context: dict[str, Any] = {}

    def set_trace_context(self, **kwargs: Any) -> None:
        """Set Langfuse trace metadata for subsequent generate_sql calls."""
        self._trace_context = {k: v for k, v in kwargs.items() if v is not None}

    def flush_traces(self) -> None:
        from langfuse import Langfuse
        Langfuse().flush()

    def generate_sql(
        self,
        system_prompt: str,
        user_prompt: str,
        db_path: Path | None = None,
    ) -> tuple[str, str, float, int, dict[str, int], list[dict[str, Any]]]:
        """Returns (sql, raw_text, elapsed, query_tool_calls, token_usage, messages)."""
        # One trace_id groups all tool-calling rounds for this generation.
        self._active_trace_id = uuid.uuid4().hex
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
        total_prompt_tokens = 0
        total_completion_tokens = 0

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

            usage = _extract_usage(response)
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

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
        token_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }
        return sql, raw_text, elapsed, query_tool_calls, token_usage, messages

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
        }
        if _looks_like_openai_base_url(self.api_base_url):
            kwargs["max_completion_tokens"] = self.max_output_tokens
        else:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.temperature != 0.0:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.reasoning_effort and _looks_like_openai_base_url(self.api_base_url):
            kwargs["reasoning_effort"] = self.reasoning_effort

        # Langfuse trace context — the patched openai module intercepts these kwargs.
        kwargs.update(
            _build_langfuse_completion_kwargs(
                default_trace_id=self._active_trace_id,
                trace_context=self._trace_context,
            )
        )

        while True:
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "reasoning_effort" in kwargs and _is_reasoning_parameter_error(exc):
                    kwargs.pop("reasoning_effort", None)
                    continue
                if "max_tokens" in kwargs and _is_max_tokens_unsupported_error(exc):
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if (
                    "max_completion_tokens" in kwargs
                    and _is_max_completion_tokens_unsupported_error(exc)
                ):
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
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


def _extract_usage(response: Any) -> dict[str, int]:
    usage = _get(response, "usage")
    if usage is None:
        return {}
    return {
        "prompt_tokens": int(_get(usage, "prompt_tokens") or 0),
        "completion_tokens": int(_get(usage, "completion_tokens") or 0),
    }


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


def _build_langfuse_completion_kwargs(
    *,
    default_trace_id: str,
    trace_context: dict[str, Any],
) -> dict[str, Any]:
    if not trace_context:
        return {"trace_id": default_trace_id}

    context = {k: v for k, v in trace_context.items() if v is not None}
    trace_id = context.pop("trace_id", None) or default_trace_id

    metadata_raw = context.pop("metadata", None)
    metadata_payload = dict(metadata_raw) if isinstance(metadata_raw, dict) else None

    direct_keys = ("name", "langfuse_public_key", "langfuse_prompt", "parent_observation_id")
    completion_kwargs: dict[str, Any] = {"trace_id": trace_id}
    for key in direct_keys:
        value = context.pop(key, None)
        if value is not None:
            completion_kwargs[key] = value

    if context:
        if metadata_payload is None:
            metadata_payload = {}
        for key, value in context.items():
            metadata_payload.setdefault(key, value)

    if metadata_payload is not None:
        completion_kwargs["metadata"] = metadata_payload
    elif metadata_raw is not None:
        completion_kwargs["metadata"] = metadata_raw

    return completion_kwargs


def _looks_like_openai_base_url(base_url: str | None) -> bool:
    if base_url is None:
        return True
    lowered = base_url.lower()
    return "api.openai.com" in lowered


def _is_tool_calling_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "max_tokens" in text or "max_completion_tokens" in text or "reasoning_effort" in text:
        return False

    mentions_tooling = any(marker in text for marker in ("tool", "tool_choice", "function"))
    mentions_unsupported = any(
        marker in text for marker in ("unsupported", "not implemented", "extra_forbidden")
    )
    return mentions_tooling and mentions_unsupported


def _is_reasoning_parameter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "reasoning_effort" in text and (
        "unknown" in text or "unsupported" in text or "not supported" in text or "extra" in text
    )


def _is_max_tokens_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "max_tokens" in text and ("unsupported" in text or "not supported" in text)


def _is_max_completion_tokens_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "max_completion_tokens" in text and ("unsupported" in text or "not supported" in text)
