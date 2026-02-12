from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from bird_scaffold.types import ExecutionResult


def execute_sql(db_path: Path, sql: str, timeout_seconds: float = 20.0) -> ExecutionResult:
    started = time.perf_counter()
    conn: sqlite3.Connection | None = None

    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        deadline = time.perf_counter() + timeout_seconds

        def _progress_handler() -> int:
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_progress_handler, 10_000)
        cur = conn.cursor()
        cur.execute(sql)

        if cur.description:
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description]
        else:
            rows = []
            columns = []

        return ExecutionResult(
            rows=rows,
            columns=columns,
            error=None,
            elapsed_s=time.perf_counter() - started,
        )
    except Exception as exc:
        return ExecutionResult(
            rows=[],
            columns=[],
            error=str(exc),
            elapsed_s=time.perf_counter() - started,
        )
    finally:
        if conn is not None:
            conn.close()


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().lower().rstrip(";").split())


def _normalize_value(value: Any, float_precision: int) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, float_precision)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return value


def canonicalize_rows(
    rows: list[tuple[Any, ...]], ordered: bool = False, float_precision: int = 6
) -> list[tuple[Any, ...]]:
    normalized = [
        tuple(_normalize_value(value, float_precision=float_precision) for value in row)
        for row in rows
    ]
    if ordered:
        return normalized

    return sorted(normalized, key=lambda row: json.dumps(row, sort_keys=False, default=str))


def results_match(
    pred_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    ordered: bool = False,
    float_precision: int = 6,
) -> bool:
    return canonicalize_rows(pred_rows, ordered=ordered, float_precision=float_precision) == canonicalize_rows(
        gold_rows, ordered=ordered, float_precision=float_precision
    )
