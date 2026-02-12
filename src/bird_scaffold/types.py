from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BirdExample:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BirdExample:
        return cls(
            question_id=int(raw.get("question_id", -1)),
            db_id=str(raw["db_id"]),
            question=str(raw["question"]),
            evidence=str(raw.get("evidence", "")),
            gold_sql=str(raw["SQL"]),
            difficulty=str(raw.get("difficulty", "unknown")),
        )


@dataclass(frozen=True)
class DatabaseContext:
    db_id: str
    db_path: Path
    schema_text: str
    data_dictionary_text: str | None = None


@dataclass
class ExecutionResult:
    rows: list[tuple[Any, ...]]
    columns: list[str]
    error: str | None
    elapsed_s: float
    truncated: bool = False


@dataclass
class GenerationResult:
    sql: str
    raw_output: str = ""
    error: str | None = None
    latency_s: float | None = None
    query_tool_calls: int = 0


@dataclass
class RunConfig:
    dataset_root: Path
    split_file: str = "dev.json"
    strategy_name: str = "single_shot"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_base_url: str | None = None
    api_key: str | None = None
    reasoning_effort: str | None = "high"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    limit: int | None = None
    offset: int = 0
    db_id: str | None = None
    include_evidence: bool = True
    output_root: Path = Path("outputs")
    ordered_result_compare: bool = False
    float_precision: int = 6
    schema_sample_rows: int = 0
    max_columns_per_table: int = 80
    data_dictionary_mode: str = "off"
    data_dictionary_max_values: int = 3
    query_timeout_seconds: float = 20.0
    query_tool_enabled: bool = True
    query_tool_max_calls: int = 8
    query_tool_max_rows: int = 50
    query_tool_max_output_chars: int = 6000
    query_tool_max_cell_chars: int = 200
    query_tool_timeout_seconds: float = 8.0
    progress_every: int = 10
