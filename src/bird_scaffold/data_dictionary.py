from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_DICTIONARY_MODE_OFF = "off"
DATA_DICTIONARY_MODE_STATS = "stats"
DATA_DICTIONARY_MODE_STATS_AND_SAMPLES = "stats_and_samples"
DATA_DICTIONARY_MODES = {
    DATA_DICTIONARY_MODE_OFF,
    DATA_DICTIONARY_MODE_STATS,
    DATA_DICTIONARY_MODE_STATS_AND_SAMPLES,
}


@dataclass(frozen=True)
class ColumnProfile:
    column_name: str
    inferred_type: str
    null_rate: float
    distinct_count: int
    min_value: str | None
    max_value: str | None
    sample_values: tuple[str, ...]


@dataclass(frozen=True)
class TableProfile:
    table_name: str
    row_count: int
    columns: tuple[ColumnProfile, ...]


@dataclass(frozen=True)
class DataDictionary:
    tables: tuple[TableProfile, ...]


def validate_data_dictionary_mode(mode: str) -> str:
    if mode not in DATA_DICTIONARY_MODES:
        valid = ", ".join(sorted(DATA_DICTIONARY_MODES))
        raise ValueError(f"Invalid data dictionary mode '{mode}'. Expected one of: {valid}")
    return mode


def data_dictionary_enabled(mode: str) -> bool:
    return validate_data_dictionary_mode(mode) != DATA_DICTIONARY_MODE_OFF


def data_dictionary_includes_samples(mode: str) -> bool:
    return validate_data_dictionary_mode(mode) == DATA_DICTIONARY_MODE_STATS_AND_SAMPLES


def build_data_dictionary(
    db_path: Path,
    *,
    max_columns_per_table: int = 80,
    max_sample_values: int = 3,
) -> DataDictionary:
    if max_columns_per_table <= 0:
        raise ValueError("max_columns_per_table must be > 0")
    if max_sample_values <= 0:
        raise ValueError("max_sample_values must be > 0")

    uri = f"file:{db_path}?mode=ro"
    tables: list[TableProfile] = []

    with sqlite3.connect(uri, uri=True) as conn:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        table_names = [row[0] for row in table_rows]

        for table_name in table_names:
            q_table = _quote_ident(table_name)
            col_rows = conn.execute(f"PRAGMA table_info({q_table})").fetchall()
            visible_cols = col_rows[:max_columns_per_table]
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {q_table}").fetchone()[0] or 0)

            column_profiles: list[ColumnProfile] = []
            for _, column_name, declared_type, *_ in visible_cols:
                column_profiles.append(
                    _profile_column(
                        conn=conn,
                        table_name=table_name,
                        column_name=column_name,
                        declared_type=declared_type or "",
                        row_count=row_count,
                        max_sample_values=max_sample_values,
                    )
                )

            tables.append(
                TableProfile(
                    table_name=table_name,
                    row_count=row_count,
                    columns=tuple(column_profiles),
                )
            )

    return DataDictionary(tables=tuple(tables))


def format_data_dictionary(dictionary: DataDictionary, *, include_samples: bool) -> str:
    lines: list[str] = []

    for table in dictionary.tables:
        lines.append(f"TABLE {table.table_name} (rows={table.row_count})")
        if not table.columns:
            lines.append("  - <no columns>")
            continue

        for column in table.columns:
            pieces = [
                f"type={column.inferred_type}",
                f"null_rate={column.null_rate:.3f}",
                f"distinct={column.distinct_count}",
            ]
            if column.min_value is not None and column.max_value is not None:
                pieces.append(f"min={column.min_value}")
                pieces.append(f"max={column.max_value}")
            if include_samples and column.sample_values:
                samples = ", ".join(column.sample_values)
                pieces.append(f"samples=[{samples}]")

            lines.append(f"  - {column.column_name}: " + ", ".join(pieces))

    return "\n".join(lines)


def _profile_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    declared_type: str,
    row_count: int,
    max_sample_values: int,
) -> ColumnProfile:
    q_table = _quote_ident(table_name)
    q_column = _quote_ident(column_name)

    null_count, distinct_count = conn.execute(
        f"""
        SELECT
            SUM(CASE WHEN {q_column} IS NULL THEN 1 ELSE 0 END),
            COUNT(DISTINCT {q_column})
        FROM {q_table}
        """
    ).fetchone()
    null_count = int(null_count or 0)
    distinct_count = int(distinct_count or 0)

    inferred_type = _infer_column_type(
        conn=conn,
        table_name=table_name,
        column_name=column_name,
        declared_type=declared_type,
    )
    min_value: str | None = None
    max_value: str | None = None
    if inferred_type in {"numeric", "date-ish"}:
        min_raw, max_raw = conn.execute(
            f"SELECT MIN({q_column}), MAX({q_column}) FROM {q_table} WHERE {q_column} IS NOT NULL"
        ).fetchone()
        min_value = _render_value(min_raw)
        max_value = _render_value(max_raw)

    samples = tuple(
        _sample_representative_values(
            conn=conn,
            table_name=table_name,
            column_name=column_name,
            max_sample_values=max_sample_values,
        )
    )
    null_rate = (null_count / row_count) if row_count else 0.0

    return ColumnProfile(
        column_name=column_name,
        inferred_type=inferred_type,
        null_rate=null_rate,
        distinct_count=distinct_count,
        min_value=min_value,
        max_value=max_value,
        sample_values=samples,
    )


def _infer_column_type(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    declared_type: str,
) -> str:
    declared = declared_type.upper()
    if _looks_numeric_type(declared):
        return "numeric"
    if _looks_date_type(declared) or _looks_date_name(column_name):
        return "date-ish"

    q_table = _quote_ident(table_name)
    q_column = _quote_ident(column_name)

    non_null_count = int(
        conn.execute(f"SELECT COUNT(*) FROM {q_table} WHERE {q_column} IS NOT NULL").fetchone()[0] or 0
    )
    if non_null_count == 0:
        return "unknown"

    numeric_native_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {q_table}
            WHERE {q_column} IS NOT NULL
              AND typeof({q_column}) IN ('integer', 'real')
            """
        ).fetchone()[0]
        or 0
    )
    numeric_text_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {q_table}
            WHERE {q_column} IS NOT NULL
              AND typeof({q_column}) = 'text'
              AND trim({q_column}) <> ''
              AND trim({q_column}) GLOB '*[0-9]*'
              AND trim({q_column}) NOT GLOB '*[^0-9.-]*'
              AND trim({q_column}) NOT IN ('-', '.', '-.')
            """
        ).fetchone()[0]
        or 0
    )
    numeric_ratio = (numeric_native_count + numeric_text_count) / non_null_count
    if numeric_ratio >= 0.9:
        return "numeric"

    date_like_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {q_table}
            WHERE {q_column} IS NOT NULL
              AND typeof({q_column}) = 'text'
              AND (
                {q_column} LIKE '____-__-__%'
                OR {q_column} LIKE '____/__/__%'
                OR {q_column} LIKE '__/__/____%'
              )
            """
        ).fetchone()[0]
        or 0
    )
    date_ratio = date_like_count / non_null_count
    if date_ratio >= 0.8:
        return "date-ish"

    return "text"


def _sample_representative_values(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    max_sample_values: int,
) -> list[str]:
    q_table = _quote_ident(table_name)
    q_column = _quote_ident(column_name)

    rows = conn.execute(
        f"""
        SELECT {q_column}, COUNT(*) AS freq
        FROM {q_table}
        WHERE {q_column} IS NOT NULL
        GROUP BY {q_column}
        ORDER BY freq DESC, {q_column}
        LIMIT ?
        """,
        (max_sample_values,),
    ).fetchall()

    return [_render_value(row[0]) for row in rows]


def _render_value(value: Any, max_len: int = 40) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.replace("\n", " ").strip()
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        escaped = text.replace("'", "''")
        return "'" + escaped + "'"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _looks_numeric_type(declared_type: str) -> bool:
    return any(token in declared_type for token in ("INT", "REAL", "NUM", "DEC", "DOUB", "FLOA"))


def _looks_date_type(declared_type: str) -> bool:
    return any(token in declared_type for token in ("DATE", "TIME"))


def _looks_date_name(column_name: str) -> bool:
    lowered = column_name.lower()
    return any(token in lowered for token in ("date", "time", "year", "month", "day"))


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
