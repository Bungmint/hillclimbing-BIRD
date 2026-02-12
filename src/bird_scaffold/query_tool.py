from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bird_scaffold.execution import execute_sql
from bird_scaffold.sql_parsing import extract_sql, is_read_only_sql

QUERY_TOOL_NAME = "query"


@dataclass(frozen=True)
class QueryToolConfig:
    max_calls: int = 8
    max_rows: int = 50
    max_output_chars: int = 6000
    max_cell_chars: int = 200
    timeout_seconds: float = 8.0

    def validated(self) -> QueryToolConfig:
        return QueryToolConfig(
            max_calls=max(0, self.max_calls),
            max_rows=max(1, self.max_rows),
            max_output_chars=max(512, self.max_output_chars),
            max_cell_chars=max(16, self.max_cell_chars),
            timeout_seconds=max(0.1, self.timeout_seconds),
        )


def build_query_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": QUERY_TOOL_NAME,
            "description": "Run a read-only SQL query against the current SQLite database to inspect data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single read-only SQLite SQL statement.",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    }


def parse_query_tool_arguments(arguments: str | dict[str, Any] | None) -> str:
    if isinstance(arguments, dict):
        raw = arguments.get("sql")
        return raw.strip() if isinstance(raw, str) else ""

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
        if isinstance(parsed, dict):
            raw = parsed.get("sql")
            return raw.strip() if isinstance(raw, str) else ""
    return ""


class QueryToolExecutor:
    def __init__(self, db_path: Path, config: QueryToolConfig) -> None:
        self.db_path = db_path
        self.config = config.validated()

    def execute(self, sql_input: str) -> str:
        sql = extract_sql(sql_input)
        if not sql:
            return self._serialize(
                {
                    "ok": False,
                    "error": "Missing `sql` argument. Provide one SQL statement.",
                }
            )

        if not is_read_only_sql(sql):
            return self._serialize(
                {
                    "ok": False,
                    "error": "Only read-only SELECT/WITH/PRAGMA/EXPLAIN SQL is allowed.",
                }
            )

        result = execute_sql(
            db_path=self.db_path,
            sql=sql,
            timeout_seconds=self.config.timeout_seconds,
            max_rows=self.config.max_rows,
        )

        if result.error is not None:
            return self._serialize(
                {
                    "ok": False,
                    "error": result.error,
                    "sql": sql,
                }
            )

        formatted_rows = self._format_rows(result.rows)
        payload = {
            "ok": True,
            "sql": sql,
            "columns": result.columns,
            "rows": formatted_rows,
            "returned_rows": len(formatted_rows),
            "row_limit": self.config.max_rows,
            "rows_truncated": result.truncated,
        }
        return self._serialize(payload)

    def _format_rows(self, rows: list[tuple[Any, ...]]) -> list[list[Any]]:
        formatted: list[list[Any]] = []
        for row in rows:
            formatted_row: list[Any] = []
            for value in row:
                if value is None or isinstance(value, (int, float, bool)):
                    formatted_row.append(value)
                    continue
                if isinstance(value, (bytes, bytearray)):
                    text = value.hex()
                else:
                    text = str(value)
                if len(text) > self.config.max_cell_chars:
                    text = text[: self.config.max_cell_chars] + "..."
                formatted_row.append(text)
            formatted.append(formatted_row)
        return formatted

    def _serialize(self, payload: dict[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= self.config.max_output_chars:
            return text

        compact = dict(payload)
        rows = list(compact.get("rows", []))
        while rows and len(text) > self.config.max_output_chars:
            rows.pop()
            compact["rows"] = rows
            compact["returned_rows"] = len(rows)
            compact["output_truncated"] = True
            text = json.dumps(compact, ensure_ascii=False, default=str)
        if len(text) <= self.config.max_output_chars:
            return text

        compact.pop("rows", None)
        compact["returned_rows"] = 0
        compact["output_truncated"] = True
        compact["message"] = "Output truncated. Query fewer rows/columns for more detail."
        text = json.dumps(compact, ensure_ascii=False, default=str)
        if len(text) <= self.config.max_output_chars:
            return text

        minimal = {
            "ok": bool(compact.get("ok")),
            "output_truncated": True,
            "message": "Output truncated.",
        }
        return json.dumps(minimal, ensure_ascii=False, default=str)
