from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bird_scaffold.dataset import filter_examples, load_examples
from bird_scaffold.db import load_database_context
from bird_scaffold.execution import execute_sql, normalize_sql, results_match
from bird_scaffold.llm_client import OpenAICompatibleText2SQLClient
from bird_scaffold.strategies import get_strategy
from bird_scaffold.types import DatabaseContext, RunConfig


def _run_dir_name(strategy_name: str, model: str) -> str:
    model_tag = model.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{strategy_name}_{model_tag}"


def _prepare_run_dir(output_root: Path, strategy_name: str, model: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / _run_dir_name(strategy_name=strategy_name, model=model)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base

    suffix = 1
    while True:
        candidate = Path(str(base) + f"_{suffix}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def _config_to_dict(config: RunConfig) -> dict[str, Any]:
    raw = asdict(config)
    raw["dataset_root"] = str(config.dataset_root)
    raw["output_root"] = str(config.output_root)
    return raw


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_experiment(config: RunConfig) -> dict[str, Any]:
    dataset_root = config.dataset_root
    examples = load_examples(dataset_root=dataset_root, split_file=config.split_file)
    examples = filter_examples(
        examples,
        db_id=config.db_id,
        offset=config.offset,
        limit=config.limit,
    )

    if not examples:
        raise ValueError("No examples selected. Check --db-id/--offset/--limit.")

    strategy = get_strategy(config.strategy_name)

    llm_client: OpenAICompatibleText2SQLClient | None = None
    if strategy.requires_llm:
        llm_client = OpenAICompatibleText2SQLClient(
            model=config.model,
            api_base_url=config.api_base_url,
            api_key=config.api_key,
            reasoning_effort=config.reasoning_effort,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            query_tool_enabled=config.query_tool_enabled,
            query_tool_max_calls=config.query_tool_max_calls,
            query_tool_max_rows=config.query_tool_max_rows,
            query_tool_max_output_chars=config.query_tool_max_output_chars,
            query_tool_max_cell_chars=config.query_tool_max_cell_chars,
            query_tool_timeout_seconds=config.query_tool_timeout_seconds,
        )

    run_dir = _prepare_run_dir(
        output_root=config.output_root,
        strategy_name=config.strategy_name,
        model=config.model,
    )

    started_at = datetime.now().isoformat(timespec="seconds")

    db_cache: dict[str, DatabaseContext] = {}
    rows: list[dict[str, Any]] = []

    exec_match_count = 0
    exact_match_count = 0
    executable_count = 0
    generation_error_count = 0
    gold_exec_error_count = 0
    query_tool_calls_total = 0

    total = len(examples)

    for index, example in enumerate(examples, start=1):
        if example.db_id not in db_cache:
            db_cache[example.db_id] = load_database_context(
                dataset_root=dataset_root,
                db_id=example.db_id,
                sample_rows_per_table=config.schema_sample_rows,
                max_columns_per_table=config.max_columns_per_table,
                data_dictionary_mode=config.data_dictionary_mode,
                data_dictionary_max_values=config.data_dictionary_max_values,
            )
        db_context = db_cache[example.db_id]

        generation = strategy.generate(
            example=example,
            db_context=db_context,
            llm_client=llm_client,
            include_evidence=config.include_evidence,
        )

        predicted_sql = generation.sql.strip()
        if not predicted_sql:
            prediction_exec = execute_sql(
                db_path=db_context.db_path,
                sql="SELECT 1 WHERE 0",
                timeout_seconds=config.query_timeout_seconds,
            )
            prediction_exec.error = generation.error or "Empty SQL output"
        else:
            prediction_exec = execute_sql(
                db_path=db_context.db_path,
                sql=predicted_sql,
                timeout_seconds=config.query_timeout_seconds,
            )

        gold_exec = execute_sql(
            db_path=db_context.db_path,
            sql=example.gold_sql,
            timeout_seconds=config.query_timeout_seconds,
        )

        executable = prediction_exec.error is None
        if executable:
            executable_count += 1

        if generation.error:
            generation_error_count += 1
        query_tool_calls_total += generation.query_tool_calls

        if gold_exec.error:
            gold_exec_error_count += 1

        exact_match = bool(predicted_sql) and normalize_sql(predicted_sql) == normalize_sql(example.gold_sql)
        if exact_match:
            exact_match_count += 1

        exec_match = False
        if prediction_exec.error is None and gold_exec.error is None:
            exec_match = results_match(
                pred_rows=prediction_exec.rows,
                gold_rows=gold_exec.rows,
                ordered=config.ordered_result_compare,
                float_precision=config.float_precision,
            )
            if exec_match:
                exec_match_count += 1

        rows.append(
            {
                "index": index - 1,
                "question_id": example.question_id,
                "db_id": example.db_id,
                "difficulty": example.difficulty,
                "question": example.question,
                "evidence": example.evidence,
                "gold_sql": example.gold_sql,
                "predicted_sql": predicted_sql,
                "strategy": config.strategy_name,
                "model": config.model,
                "generation_error": generation.error,
                "generation_latency_s": generation.latency_s,
                "query_tool_calls": generation.query_tool_calls,
                "prediction_exec_error": prediction_exec.error,
                "gold_exec_error": gold_exec.error,
                "prediction_exec_time_s": prediction_exec.elapsed_s,
                "gold_exec_time_s": gold_exec.elapsed_s,
                "prediction_row_count": len(prediction_exec.rows) if prediction_exec.error is None else None,
                "gold_row_count": len(gold_exec.rows) if gold_exec.error is None else None,
                "executable": executable,
                "exact_match": exact_match,
                "exec_match": exec_match,
            }
        )

        if index % config.progress_every == 0 or index == total:
            current_exec_acc = exec_match_count / index
            current_executable_rate = executable_count / index
            print(
                f"[{index}/{total}] exec_acc={current_exec_acc:.3f} executable={current_executable_rate:.3f}"
            )

    finished_at = datetime.now().isoformat(timespec="seconds")

    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "strategy": config.strategy_name,
        "model": config.model,
        "num_examples": total,
        "num_dbs": len({example.db_id for example in examples}),
        "num_executable_predictions": executable_count,
        "executable_rate": executable_count / total,
        "num_exec_matches": exec_match_count,
        "execution_accuracy": exec_match_count / total,
        "num_exact_matches": exact_match_count,
        "exact_match_accuracy": exact_match_count / total,
        "num_generation_errors": generation_error_count,
        "num_gold_exec_errors": gold_exec_error_count,
        "num_query_tool_calls": query_tool_calls_total,
        "avg_query_tool_calls_per_example": query_tool_calls_total / total,
        "invalid_sql_rate": 1.0 - (executable_count / total),
        "ordered_result_compare": config.ordered_result_compare,
        "float_precision": config.float_precision,
        "run_dir": str(run_dir),
        "predictions_path": str(run_dir / "predictions.jsonl"),
        "summary_path": str(run_dir / "summary.json"),
    }

    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "run_config.json", _config_to_dict(config))
    _write_jsonl(run_dir / "predictions.jsonl", rows)

    return summary
