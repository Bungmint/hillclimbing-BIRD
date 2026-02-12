from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from bird_scaffold.data_dictionary import (
    DATA_DICTIONARY_MODE_OFF,
    DATA_DICTIONARY_MODE_STATS,
    DATA_DICTIONARY_MODE_STATS_AND_SAMPLES,
)
from bird_scaffold.dataset import load_examples
from bird_scaffold.db import load_database_context, resolve_db_path
from bird_scaffold.runner import run_experiment
from bird_scaffold.strategies import list_strategies
from bird_scaffold.types import RunConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bird-scaffold", description="BIRD baseline scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="Preview dataset stats and DB wiring")
    preview.add_argument("--dataset-root", default="dev_20240627")
    preview.add_argument("--split-file", default="dev.json")

    show_schema = subparsers.add_parser("show-schema", help="Print schema context for one database")
    show_schema.add_argument("--dataset-root", default="dev_20240627")
    show_schema.add_argument("--db-id", required=True)
    show_schema.add_argument("--sample-rows", type=int, default=0)
    show_schema.add_argument("--max-columns", type=int, default=80)
    show_schema.add_argument(
        "--data-dictionary-mode",
        choices=[DATA_DICTIONARY_MODE_OFF, DATA_DICTIONARY_MODE_STATS, DATA_DICTIONARY_MODE_STATS_AND_SAMPLES],
        default=DATA_DICTIONARY_MODE_OFF,
    )
    show_schema.add_argument("--data-dictionary-max-values", type=int, default=3)

    subparsers.add_parser("strategies", help="List available generation strategies")

    run = subparsers.add_parser("run", help="Run baseline experiment")
    run.add_argument("--dataset-root", default="dev_20240627")
    run.add_argument("--split-file", default="dev.json")
    run.add_argument("--db-id", default=None)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--strategy", default="single_shot")
    run.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    run.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI-compatible base URL (for vLLM use e.g. http://127.0.0.1:8000/v1).",
    )
    run.add_argument(
        "--api-key",
        default=None,
        help="API key override. If omitted, OPENAI_API_KEY or VLLM_API_KEY are used.",
    )
    run.add_argument(
        "--reasoning-effort",
        choices=["off", "minimal", "low", "medium", "high"],
        default="high",
        help="Reasoning effort (OpenAI models only). Use 'off' to disable.",
    )
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-output-tokens", type=int, default=4096)
    run.add_argument("--no-evidence", action="store_true")
    run.add_argument("--ordered", action="store_true")
    run.add_argument("--float-precision", type=int, default=6)
    run.add_argument("--schema-sample-rows", type=int, default=0)
    run.add_argument("--max-columns", type=int, default=80)
    run.add_argument(
        "--data-dictionary-mode",
        choices=[DATA_DICTIONARY_MODE_OFF, DATA_DICTIONARY_MODE_STATS, DATA_DICTIONARY_MODE_STATS_AND_SAMPLES],
        default=DATA_DICTIONARY_MODE_OFF,
    )
    run.add_argument("--data-dictionary-max-values", type=int, default=3)
    run.add_argument("--query-timeout", type=float, default=20.0)
    run.add_argument("--no-query-tool", action="store_true")
    run.add_argument("--query-tool-max-calls", type=int, default=8)
    run.add_argument("--query-tool-max-rows", type=int, default=50)
    run.add_argument("--query-tool-max-output-chars", type=int, default=6000)
    run.add_argument("--query-tool-max-cell-chars", type=int, default=200)
    run.add_argument("--query-tool-timeout", type=float, default=8.0)
    run.add_argument("--progress-every", type=int, default=10)
    run.add_argument("--output-root", default="outputs")

    return parser


def _cmd_preview(dataset_root: Path, split_file: str) -> None:
    examples = load_examples(dataset_root=dataset_root, split_file=split_file)
    counts = Counter(example.db_id for example in examples)

    print(f"dataset_root: {dataset_root}")
    print(f"split_file: {split_file}")
    print(f"num_examples: {len(examples)}")
    print(f"num_databases: {len(counts)}")

    for db_id, count in sorted(counts.items()):
        db_path = resolve_db_path(dataset_root=dataset_root, db_id=db_id)
        print(f"  - {db_id}: {count} examples ({db_path})")


def _cmd_show_schema(
    dataset_root: Path,
    db_id: str,
    sample_rows: int,
    max_columns: int,
    data_dictionary_mode: str,
    data_dictionary_max_values: int,
) -> None:
    context = load_database_context(
        dataset_root=dataset_root,
        db_id=db_id,
        sample_rows_per_table=sample_rows,
        max_columns_per_table=max_columns,
        data_dictionary_mode=data_dictionary_mode,
        data_dictionary_max_values=data_dictionary_max_values,
    )
    print(context.schema_text)
    if context.data_dictionary_text:
        print("\nDB Notes (data dictionary):")
        print(context.data_dictionary_text)


def _cmd_strategies() -> None:
    for strategy_name in list_strategies():
        print(strategy_name)


def _cmd_run(args: argparse.Namespace) -> None:
    api_base_url = args.api_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY")

    config = RunConfig(
        dataset_root=Path(args.dataset_root),
        split_file=args.split_file,
        strategy_name=args.strategy,
        model=args.model,
        api_base_url=api_base_url,
        api_key=api_key,
        reasoning_effort=None if args.reasoning_effort == "off" else args.reasoning_effort,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        limit=args.limit,
        offset=args.offset,
        db_id=args.db_id,
        include_evidence=not args.no_evidence,
        output_root=Path(args.output_root),
        ordered_result_compare=args.ordered,
        float_precision=args.float_precision,
        schema_sample_rows=args.schema_sample_rows,
        max_columns_per_table=args.max_columns,
        data_dictionary_mode=args.data_dictionary_mode,
        data_dictionary_max_values=args.data_dictionary_max_values,
        query_timeout_seconds=args.query_timeout,
        query_tool_enabled=not args.no_query_tool,
        query_tool_max_calls=args.query_tool_max_calls,
        query_tool_max_rows=args.query_tool_max_rows,
        query_tool_max_output_chars=args.query_tool_max_output_chars,
        query_tool_max_cell_chars=args.query_tool_max_cell_chars,
        query_tool_timeout_seconds=args.query_tool_timeout,
        progress_every=args.progress_every,
    )

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


def main() -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "preview":
            _cmd_preview(dataset_root=Path(args.dataset_root), split_file=args.split_file)
            return

        if args.command == "show-schema":
            _cmd_show_schema(
                dataset_root=Path(args.dataset_root),
                db_id=args.db_id,
                sample_rows=args.sample_rows,
                max_columns=args.max_columns,
                data_dictionary_mode=args.data_dictionary_mode,
                data_dictionary_max_values=args.data_dictionary_max_values,
            )
            return

        if args.command == "strategies":
            _cmd_strategies()
            return

        if args.command == "run":
            _cmd_run(args)
            return

        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
