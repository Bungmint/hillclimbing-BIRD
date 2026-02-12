from __future__ import annotations

import sqlite3
from pathlib import Path

from bird_scaffold.data_dictionary import (
    build_data_dictionary,
    data_dictionary_enabled,
    data_dictionary_includes_samples,
    format_data_dictionary,
)
from bird_scaffold.types import DatabaseContext


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def resolve_db_path(dataset_root: Path, db_id: str) -> Path:
    db_path = dataset_root / "dev_databases" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found for db_id='{db_id}': {db_path}")
    return db_path


def build_schema_text(db_path: Path, sample_rows_per_table: int = 0, max_columns_per_table: int = 80) -> str:
    lines: list[str] = []
    uri = f"file:{db_path}?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        table_names = [row[0] for row in table_rows]

        for table_name in table_names:
            q_table = _quote_ident(table_name)
            col_rows = conn.execute(f"PRAGMA table_info({q_table})").fetchall()
            fk_rows = conn.execute(f"PRAGMA foreign_key_list({q_table})").fetchall()

            lines.append(f"TABLE {table_name}")

            if not col_rows:
                lines.append("  columns: <none>")
            else:
                truncated = len(col_rows) > max_columns_per_table
                visible_cols = col_rows[:max_columns_per_table]
                for col in visible_cols:
                    # pragma table_info columns: cid, name, type, notnull, dflt_value, pk
                    _, col_name, col_type, notnull, _, pk = col
                    tags: list[str] = []
                    if pk:
                        tags.append("pk")
                    if notnull:
                        tags.append("not_null")
                    tag_text = f" [{' '.join(tags)}]" if tags else ""
                    lines.append(f"  - {col_name}: {col_type or 'UNKNOWN'}{tag_text}")
                if truncated:
                    lines.append(f"  - ... {len(col_rows) - max_columns_per_table} more columns")

            if fk_rows:
                lines.append("  foreign_keys:")
                for fk in fk_rows:
                    # pragma foreign_key_list columns:
                    # id, seq, table, from, to, on_update, on_delete, match
                    _, _, target_table, from_col, to_col, *_ = fk
                    lines.append(f"    - {from_col} -> {target_table}.{to_col}")

            if sample_rows_per_table > 0:
                sample_rows = conn.execute(
                    f"SELECT * FROM {q_table} LIMIT ?", (sample_rows_per_table,)
                ).fetchall()
                if sample_rows:
                    lines.append("  sample_rows:")
                    for sample in sample_rows:
                        lines.append(f"    - {repr(sample)}")

    return "\n".join(lines)


def load_database_context(
    dataset_root: Path,
    db_id: str,
    sample_rows_per_table: int = 0,
    max_columns_per_table: int = 80,
    data_dictionary_mode: str = "stats_and_samples",
    data_dictionary_max_values: int = 3,
) -> DatabaseContext:
    db_path = resolve_db_path(dataset_root, db_id)
    schema_text = build_schema_text(
        db_path=db_path,
        sample_rows_per_table=sample_rows_per_table,
        max_columns_per_table=max_columns_per_table,
    )
    data_dictionary_text: str | None = None
    if data_dictionary_enabled(data_dictionary_mode):
        data_dictionary = build_data_dictionary(
            db_path=db_path,
            max_columns_per_table=max_columns_per_table,
            max_sample_values=data_dictionary_max_values,
        )
        data_dictionary_text = format_data_dictionary(
            data_dictionary,
            include_samples=data_dictionary_includes_samples(data_dictionary_mode),
        )

    return DatabaseContext(
        db_id=db_id,
        db_path=db_path,
        schema_text=schema_text,
        data_dictionary_text=data_dictionary_text,
    )
