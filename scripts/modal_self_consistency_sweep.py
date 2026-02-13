"""Run Modal inference sweeps over self-consistency k and collect persisted results.

This script launches one
  modal run modal_app.py ... --self-consistency-samples {k}
per k in --ks (default: 1,2,4,8), in parallel.

After runs complete, it downloads persisted artifacts from the Modal output volume
(`summary.json`, `run_config.json`, and optional `predictions.jsonl`) and writes
an aggregated local report to make analysis easy.

Example:
  python scripts/modal_self_consistency_sweep.py \
    -- --model-preset qwen-8b --limit 200 --strategy single_shot --temperature 0.6
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

VOLUME_MOUNT_ROOT = PurePosixPath("/outputs")
DEFAULT_KS = (1, 2, 4, 8)


@dataclass(frozen=True)
class RunResult:
    k: int
    command: list[str]
    stdout: str
    summary_returned: dict[str, Any] | None
    summary_downloaded: dict[str, Any] | None
    local_run_dir: Path | None
    error: str | None


@dataclass(frozen=True)
class CompletedModalRun:
    k: int
    command: list[str]
    returncode: int
    stdout: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep self-consistency k for Modal inference runs and collect results."
    )
    parser.add_argument(
        "--ks",
        default="1,2,4,8",
        help="Comma-separated self-consistency k values. Default: 1,2,4,8",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/self_consistency_sweeps",
        help="Local directory where sweep artifacts are saved.",
    )
    parser.add_argument(
        "--modal-app",
        default="modal_app.py",
        help="Modal app file to run.",
    )
    parser.add_argument(
        "--volume-name",
        default="bird-outputs",
        help="Modal output volume name mounted at /outputs in modal_app.py.",
    )
    parser.add_argument(
        "--modal-env",
        default=None,
        help="Optional Modal environment (-e/--env).",
    )
    parser.add_argument(
        "--download-predictions",
        action="store_true",
        help="Also download predictions.jsonl for each run (can be large).",
    )
    parser.add_argument(
        "--no-quiet-modal",
        action="store_true",
        help="Do not pass -q to `modal run`.",
    )
    parser.add_argument(
        "modal_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to modal_app.py local entrypoint (prefix with --).",
    )
    args = parser.parse_args()

    if args.modal_args and args.modal_args[0] == "--":
        args.modal_args = args.modal_args[1:]

    if any(arg == "--self-consistency-samples" for arg in args.modal_args):
        raise ValueError("Do not pass --self-consistency-samples in passthrough args; use --ks instead.")

    args.k_values = _parse_k_values(args.ks)
    return args


def _parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k < 1:
            raise ValueError(f"Invalid k '{k}'. All k must be >= 1.")
        values.append(k)
    if not values:
        return list(DEFAULT_KS)
    # Deduplicate while preserving order.
    ordered_unique = list(dict.fromkeys(values))
    return ordered_unique


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    last_obj: dict[str, Any] | None = None
    index = 0
    while True:
        start = text.find("{", index)
        if start == -1:
            break
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                last_obj = parsed
        except json.JSONDecodeError:
            pass
        index = start + 1
    return last_obj


def _container_path_to_volume_path(container_path: str) -> str:
    posix = PurePosixPath(container_path)
    try:
        relative = posix.relative_to(VOLUME_MOUNT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Expected path under {VOLUME_MOUNT_ROOT}, got '{container_path}'."
        ) from exc
    return "/" + str(relative)


def _modal_run_command(
    *,
    modal_app: str,
    modal_args: list[str],
    k: int,
    modal_env: str | None,
    quiet_modal: bool,
) -> list[str]:
    cmd = ["modal", "run"]
    if quiet_modal:
        cmd.append("-q")
    if modal_env:
        cmd.extend(["-e", modal_env])
    cmd.append(modal_app)
    cmd.extend(modal_args)
    cmd.extend(["--self-consistency-samples", str(k)])
    return cmd


def _run_command(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    output = completed.stdout or ""
    return completed.returncode, output


def _run_modal_commands_parallel(
    *,
    k_values: list[int],
    modal_app: str,
    modal_args: list[str],
    modal_env: str | None,
    quiet_modal: bool,
) -> list[CompletedModalRun]:
    commands = {
        k: _modal_run_command(
            modal_app=modal_app,
            modal_args=modal_args,
            k=k,
            modal_env=modal_env,
            quiet_modal=quiet_modal,
        )
        for k in k_values
    }

    print("\n=== Launching parallel Modal runs ===")
    for k in k_values:
        print(f"k={k}: {' '.join(commands[k])}")

    completed_by_k: dict[int, CompletedModalRun] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(k_values))) as executor:
        future_to_context = {
            executor.submit(_run_command, command): (k, command) for k, command in commands.items()
        }
        for future in as_completed(future_to_context):
            k, command = future_to_context[future]
            returncode, output = future.result()
            completed_by_k[k] = CompletedModalRun(
                k=k,
                command=command,
                returncode=returncode,
                stdout=output,
            )
            print(f"\n=== Completed k={k} (exit={returncode}) ===")
            print(output, end="" if output.endswith("\n") else "\n")

    return [completed_by_k[k] for k in k_values]


def _modal_volume_get(
    *,
    volume_name: str,
    remote_path: str,
    local_destination: Path,
    modal_env: str | None,
) -> None:
    local_destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", "--force"]
    if modal_env:
        cmd.extend(["-e", modal_env])
    cmd.extend([volume_name, remote_path, str(local_destination)])
    returncode, output = _run_command(cmd)
    if returncode != 0:
        raise RuntimeError(
            f"Failed to download '{remote_path}' from volume '{volume_name}'.\nCommand: {' '.join(cmd)}\n{output}"
        )


def _iso_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _duration_seconds(summary: dict[str, Any]) -> float | None:
    started = summary.get("started_at")
    finished = summary.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        started_at = datetime.fromisoformat(started)
        finished_at = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return (finished_at - started_at).total_seconds()


def _build_metric_row(
    *,
    k: int,
    summary: dict[str, Any],
    local_summary_path: Path,
) -> dict[str, Any]:
    return {
        "k": k,
        "execution_accuracy": summary.get("execution_accuracy"),
        "exact_match_accuracy": summary.get("exact_match_accuracy"),
        "executable_rate": summary.get("executable_rate"),
        "avg_total_tokens_per_example": summary.get("avg_total_tokens_per_example"),
        "avg_query_tool_calls_per_example": summary.get("avg_query_tool_calls_per_example"),
        "num_examples": summary.get("num_examples"),
        "duration_s": _duration_seconds(summary),
        "run_dir": summary.get("run_dir"),
        "local_summary_path": str(local_summary_path),
        "self_consistency_vote_fraction": summary.get("avg_self_consistency_vote_fraction"),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _print_metrics_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No successful runs to report.")
        return

    header = (
        f"{'k':>3}  {'exec_acc':>8}  {'exact_acc':>9}  {'exec_rate':>9}  {'avg_tokens':>10}  {'duration_s':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        exec_acc = row.get("execution_accuracy")
        exact_acc = row.get("exact_match_accuracy")
        exec_rate = row.get("executable_rate")
        avg_tokens = row.get("avg_total_tokens_per_example")
        duration_s = row.get("duration_s")
        print(
            f"{row['k']:>3}  "
            f"{_fmt_float(exec_acc):>8}  "
            f"{_fmt_float(exact_acc):>9}  "
            f"{_fmt_float(exec_rate):>9}  "
            f"{_fmt_float(avg_tokens):>10}  "
            f"{_fmt_float(duration_s):>10}"
        )


def _fmt_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.4f}"


def _error_run_result(
    *,
    k: int,
    command: list[str],
    stdout: str,
    error: str,
    summary_returned: dict[str, Any] | None = None,
) -> RunResult:
    return RunResult(
        k=k,
        command=command,
        stdout=stdout,
        summary_returned=summary_returned,
        summary_downloaded=None,
        local_run_dir=None,
        error=error,
    )


def _download_run_artifacts(
    *,
    k: int,
    summary_returned: dict[str, Any],
    sweep_dir: Path,
    volume_name: str,
    modal_env: str | None,
    download_predictions: bool,
) -> tuple[Path, dict[str, Any], str]:
    run_dir_value = summary_returned.get("run_dir")
    summary_path_value = summary_returned.get("summary_path")
    if not isinstance(run_dir_value, str) or not run_dir_value:
        raise ValueError(f"Summary for k={k} is missing string field 'run_dir'.")
    if not isinstance(summary_path_value, str) or not summary_path_value:
        raise ValueError(f"Summary for k={k} is missing string field 'summary_path'.")

    run_config_path_container = str(PurePosixPath(run_dir_value) / "run_config.json")
    run_dir_volume = _container_path_to_volume_path(run_dir_value)
    summary_path_volume = _container_path_to_volume_path(summary_path_value)
    run_config_path_volume = _container_path_to_volume_path(run_config_path_container)

    local_run_dir = sweep_dir / f"k_{k}"
    local_run_dir.mkdir(parents=True, exist_ok=True)

    _modal_volume_get(
        volume_name=volume_name,
        remote_path=summary_path_volume,
        local_destination=local_run_dir / "summary.json",
        modal_env=modal_env,
    )
    _modal_volume_get(
        volume_name=volume_name,
        remote_path=run_config_path_volume,
        local_destination=local_run_dir / "run_config.json",
        modal_env=modal_env,
    )

    if download_predictions:
        predictions_container = str(PurePosixPath(run_dir_value) / "predictions.jsonl")
        predictions_volume = _container_path_to_volume_path(predictions_container)
        _modal_volume_get(
            volume_name=volume_name,
            remote_path=predictions_volume,
            local_destination=local_run_dir / "predictions.jsonl",
            modal_env=modal_env,
        )

    summary_downloaded = json.loads((local_run_dir / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary_downloaded, dict):
        raise ValueError(f"Downloaded summary for k={k} is not a JSON object.")
    return local_run_dir, summary_downloaded, run_dir_volume


def main() -> int:
    args = _parse_args()
    sweep_id = _iso_utc_timestamp()
    sweep_dir = Path(args.output_root) / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "sweep_id": sweep_id,
        "k_values": args.k_values,
        "modal_app": args.modal_app,
        "volume_name": args.volume_name,
        "modal_env": args.modal_env,
        "modal_args": args.modal_args,
        "download_predictions": args.download_predictions,
        "quiet_modal": not args.no_quiet_modal,
    }
    _write_json(sweep_dir / "sweep_config.json", config_snapshot)

    results: list[RunResult] = []
    completed_runs = _run_modal_commands_parallel(
        k_values=args.k_values,
        modal_app=args.modal_app,
        modal_args=args.modal_args,
        modal_env=args.modal_env,
        quiet_modal=not args.no_quiet_modal,
    )

    for run in completed_runs:
        if run.returncode != 0:
            results.append(
                _error_run_result(
                    k=run.k,
                    command=run.command,
                    stdout=run.stdout,
                    error=f"modal run failed for k={run.k} (exit={run.returncode})",
                )
            )
            continue

        summary_returned = _extract_last_json_object(run.stdout)
        if summary_returned is None:
            results.append(
                _error_run_result(
                    k=run.k,
                    command=run.command,
                    stdout=run.stdout,
                    error=f"Could not parse JSON summary from modal output for k={run.k}.",
                )
            )
            continue

        try:
            local_run_dir, summary_downloaded, run_dir_volume = _download_run_artifacts(
                k=run.k,
                summary_returned=summary_returned,
                sweep_dir=sweep_dir,
                volume_name=args.volume_name,
                modal_env=args.modal_env,
                download_predictions=args.download_predictions,
            )
        except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
            results.append(
                _error_run_result(
                    k=run.k,
                    command=run.command,
                    stdout=run.stdout,
                    summary_returned=summary_returned,
                    error=f"Artifact download failed for k={run.k}: {exc}",
                )
            )
            continue

        results.append(
            RunResult(
                k=run.k,
                command=run.command,
                stdout=run.stdout,
                summary_returned=summary_returned,
                summary_downloaded=summary_downloaded,
                local_run_dir=local_run_dir,
                error=None,
            )
        )
        print(f"Downloaded artifacts for k={run.k} to {local_run_dir}")
        print(f"Volume run dir: {run_dir_volume}")

    serializable_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for run in results:
        summary_for_metrics = run.summary_downloaded or run.summary_returned
        local_summary_path = run.local_run_dir / "summary.json" if run.local_run_dir else None
        if summary_for_metrics is not None and local_summary_path is not None:
            metric_rows.append(
                _build_metric_row(
                    k=run.k,
                    summary=summary_for_metrics,
                    local_summary_path=local_summary_path,
                )
            )
        serializable_results.append(
            {
                "k": run.k,
                "command": run.command,
                "error": run.error,
                "local_run_dir": str(run.local_run_dir) if run.local_run_dir else None,
                "summary_returned": run.summary_returned,
                "summary_downloaded": run.summary_downloaded,
            }
        )

    metric_rows.sort(
        key=lambda row: (
            -(row["execution_accuracy"] if isinstance(row.get("execution_accuracy"), (int, float)) else -1.0),
            row["k"],
        )
    )

    _write_json(sweep_dir / "results.json", serializable_results)
    _write_jsonl(sweep_dir / "results.jsonl", serializable_results)
    _write_json(sweep_dir / "metrics.json", metric_rows)

    csv_columns = [
        "k",
        "execution_accuracy",
        "exact_match_accuracy",
        "executable_rate",
        "avg_total_tokens_per_example",
        "avg_query_tool_calls_per_example",
        "self_consistency_vote_fraction",
        "num_examples",
        "duration_s",
        "run_dir",
        "local_summary_path",
    ]
    _write_csv(sweep_dir / "metrics.csv", metric_rows, csv_columns)

    print("\n=== Sweep complete ===")
    print(f"Artifacts saved to: {sweep_dir}")
    _print_metrics_table(metric_rows)

    failures = [run for run in results if run.error is not None]
    if failures:
        print("\nFailures:")
        for run in failures:
            print(f"  k={run.k}: {run.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
