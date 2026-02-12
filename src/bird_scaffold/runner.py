from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bird_scaffold.dataset import filter_examples, load_examples
from bird_scaffold.db import load_database_context
from bird_scaffold.execution import execute_sql, normalize_sql, results_match
from bird_scaffold.llm_client import OpenAICompatibleText2SQLClient
from bird_scaffold.strategies import get_strategy
from bird_scaffold.types import BirdExample, DatabaseContext, RunConfig


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


_thread_local = threading.local()


def _looks_like_openai_endpoint(base_url: str | None) -> bool:
    if base_url is None:
        return True
    return "api.openai.com" in base_url.lower()


def _resolve_generation_temperature(config: RunConfig) -> float | None:
    # GPT-5 chat-completions can reject explicit temperature=0.0;
    # omit the param to let the model use its server default.
    if (
        _looks_like_openai_endpoint(config.api_base_url)
        and config.model.lower().startswith("gpt-5")
        and config.temperature == 0.0
    ):
        return None
    return config.temperature


def _get_thread_llm_client(config: RunConfig) -> OpenAICompatibleText2SQLClient:
    """Return a per-thread LLM client (created lazily, reused within the same thread)."""
    client = getattr(_thread_local, "llm_client", None)
    if client is None:
        client = OpenAICompatibleText2SQLClient(
            model=config.model,
            api_base_url=config.api_base_url,
            api_key=config.api_key,
            reasoning_effort=config.reasoning_effort,
            temperature=_resolve_generation_temperature(config),
            top_p=config.top_p,
            max_output_tokens=config.max_output_tokens,
            query_tool_enabled=config.query_tool_enabled,
            query_tool_max_calls=config.query_tool_max_calls,
            query_tool_max_rows=config.query_tool_max_rows,
            query_tool_max_output_chars=config.query_tool_max_output_chars,
            query_tool_max_cell_chars=config.query_tool_max_cell_chars,
            query_tool_timeout_seconds=config.query_tool_timeout_seconds,
        )
        _thread_local.llm_client = client
    return client


def _process_example(
    index: int,
    example: BirdExample,
    db_context: DatabaseContext,
    config: RunConfig,
    strategy: Any,
    run_dir_name: str,
) -> dict[str, Any]:
    """Process a single example: generate SQL, execute, compare. Thread-safe."""
    llm_client: OpenAICompatibleText2SQLClient | None = None
    if strategy.requires_llm:
        llm_client = _get_thread_llm_client(config)
        llm_client.set_trace_context(
            session_id=run_dir_name,
            name=f"q{example.question_id}_{example.db_id}",
            metadata={
                "question_id": example.question_id,
                "db_id": example.db_id,
                "difficulty": example.difficulty,
                "strategy": config.strategy_name,
                "model": config.model,
            },
            tags=[config.strategy_name, example.db_id, example.difficulty],
        )

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

    exact_match = bool(predicted_sql) and normalize_sql(predicted_sql) == normalize_sql(
        example.gold_sql
    )

    exec_match = False
    if prediction_exec.error is None and gold_exec.error is None:
        exec_match = results_match(
            pred_rows=prediction_exec.rows,
            gold_rows=gold_exec.rows,
            ordered=config.ordered_result_compare,
            float_precision=config.float_precision,
        )

    return {
        "index": index,
        "question_id": example.question_id,
        "db_id": example.db_id,
        "difficulty": example.difficulty,
        "question": example.question,
        "evidence": example.evidence,
        "gold_sql": example.gold_sql,
        "predicted_sql": predicted_sql,
        "strategy": config.strategy_name,
        "model": config.model,
        "raw_output": generation.raw_output,
        "generation_error": generation.error,
        "generation_latency_s": generation.latency_s,
        "query_tool_calls": generation.query_tool_calls,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "total_tokens": generation.total_tokens,
        "messages": generation.messages,
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
    max_workers = max(1, config.max_workers)

    run_dir = _prepare_run_dir(
        output_root=config.output_root,
        strategy_name=config.strategy_name,
        model=config.model,
    )

    started_at = datetime.now().isoformat(timespec="seconds")
    total = len(examples)

    # --- Phase 1: Pre-load all database contexts in parallel ---
    unique_db_ids = list({ex.db_id for ex in examples})
    db_cache: dict[str, DatabaseContext] = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_db_ids))) as pool:
        future_to_db_id = {
            pool.submit(
                load_database_context,
                dataset_root=dataset_root,
                db_id=db_id,
                sample_rows_per_table=config.schema_sample_rows,
                max_columns_per_table=config.max_columns_per_table,
                data_dictionary_mode=config.data_dictionary_mode,
                data_dictionary_max_values=config.data_dictionary_max_values,
            ): db_id
            for db_id in unique_db_ids
        }
        for future in as_completed(future_to_db_id):
            db_id = future_to_db_id[future]
            db_cache[db_id] = future.result()

    print(f"Loaded {len(db_cache)} database context(s). Processing {total} examples with {max_workers} workers...")

    # --- Phase 2: Process examples in parallel ---
    rows: list[dict[str, Any]] = []
    completed_count = 0
    exec_match_count = 0
    executable_count = 0
    progress_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {
            pool.submit(
                _process_example,
                index=i,
                example=example,
                db_context=db_cache[example.db_id],
                config=config,
                strategy=strategy,
                run_dir_name=run_dir.name,
            ): i
            for i, example in enumerate(examples)
        }

        for future in as_completed(future_to_index):
            row = future.result()
            with progress_lock:
                rows.append(row)
                completed_count += 1
                if row["exec_match"]:
                    exec_match_count += 1
                if row["executable"]:
                    executable_count += 1
                if completed_count % config.progress_every == 0 or completed_count == total:
                    current_exec_acc = exec_match_count / completed_count
                    current_executable_rate = executable_count / completed_count
                    print(
                        f"[{completed_count}/{total}] exec_acc={current_exec_acc:.3f} executable={current_executable_rate:.3f}"
                    )

    # Sort rows by original index for deterministic output
    rows.sort(key=lambda r: r["index"])

    finished_at = datetime.now().isoformat(timespec="seconds")

    # --- Aggregate metrics ---
    exec_match_count = sum(1 for r in rows if r["exec_match"])
    exact_match_count = sum(1 for r in rows if r["exact_match"])
    executable_count = sum(1 for r in rows if r["executable"])
    generation_error_count = sum(1 for r in rows if r["generation_error"])
    gold_exec_error_count = sum(1 for r in rows if r["gold_exec_error"])
    query_tool_calls_total = sum(r["query_tool_calls"] for r in rows)
    prompt_tokens_total = sum(r["prompt_tokens"] or 0 for r in rows)
    completion_tokens_total = sum(r["completion_tokens"] or 0 for r in rows)

    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "strategy": config.strategy_name,
        "model": config.model,
        "num_examples": total,
        "num_dbs": len(unique_db_ids),
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
        "total_prompt_tokens": prompt_tokens_total,
        "total_completion_tokens": completion_tokens_total,
        "total_tokens": prompt_tokens_total + completion_tokens_total,
        "avg_prompt_tokens_per_example": prompt_tokens_total / total,
        "avg_completion_tokens_per_example": completion_tokens_total / total,
        "avg_total_tokens_per_example": (prompt_tokens_total + completion_tokens_total) / total,
        "invalid_sql_rate": 1.0 - (executable_count / total),
        "ordered_result_compare": config.ordered_result_compare,
        "float_precision": config.float_precision,
        "max_workers": max_workers,
        "run_dir": str(run_dir),
        "predictions_path": str(run_dir / "predictions.jsonl"),
        "summary_path": str(run_dir / "summary.json"),
    }

    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "run_config.json", _config_to_dict(config))
    _write_jsonl(run_dir / "predictions.jsonl", rows)

    # Flush Langfuse traces from all thread-local clients
    if strategy.requires_llm:
        try:
            from langfuse import Langfuse

            Langfuse().flush()
        except Exception:
            pass

    return summary
