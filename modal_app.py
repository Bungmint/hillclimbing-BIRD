"""Modal entrypoint for running BIRD experiments on top of vLLM (Qwen3 8B/30B-A3B).

Examples:
  modal run modal_app.py --model-preset qwen-8b --limit 200 --strategy single_shot
  modal run modal_app.py --model-preset qwen-30b --limit 100 --query-tool-max-calls 6
  modal run modal_app.py --model-preset qwen-8b --lora-adapter-path /adapters/your_grpo_adapter --limit 200
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import modal

APP_NAME = "bird-scaffold-vllm"
DATASET_VOLUME_NAME = "bird-dev-20240627"
OUTPUT_VOLUME_NAME = "bird-outputs"
MODEL_CACHE_VOLUME_NAME = "hf-model-cache"
ADAPTER_VOLUME_NAME = "bird-rl-adapters"

REMOTE_DATASET_ROOT = Path("/datasets/dev_20240627")
REMOTE_OUTPUT_ROOT = Path("/outputs")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_API_KEY = "bird-vllm"


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_id: str
    gpu: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int = 8192


MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen-8b": ModelPreset(
        name="qwen-8b",
        model_id="Qwen/Qwen3-8B",
        gpu="B200",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
    ),
    "qwen-30b": ModelPreset(
        name="qwen-30b",
        model_id="Qwen/Qwen3-30B-A3B",
        gpu="B200",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
    ),
}


def _build_vllm_command(
    preset: ModelPreset,
    *,
    model_id: str,
    lora_adapter_path: str | None,
    lora_adapter_name: str,
) -> list[str]:
    return [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--model",
        model_id,
        "--served-model-name",
        model_id,
        "--tensor-parallel-size",
        str(preset.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(preset.gpu_memory_utilization),
        "--max-model-len",
        str(preset.max_model_len),
        "--api-key",
        VLLM_API_KEY,
        "--trust-remote-code",
    ] + (
        ["--enable-lora", "--lora-modules", f"{lora_adapter_name}={lora_adapter_path}"]
        if lora_adapter_path
        else []
    )


def _extract_served_model_ids(payload: dict) -> list[str]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if isinstance(model_id, str) and model_id:
            ids.append(model_id)
    return ids


def _wait_for_server_ready(
    base_url: str,
    process: subprocess.Popen[str],
    *,
    requested_model_id: str,
    timeout_seconds: float = 900.0,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    models_url = f"{base_url}/models"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM server exited before becoming ready.")
        try:
            with urlopen(models_url, timeout=3.0) as response:  # noqa: S310
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    served_model_ids = _extract_served_model_ids(payload)
                    if requested_model_id in served_model_ids:
                        return requested_model_id
                    if served_model_ids:
                        return served_model_ids[0]
                    return requested_model_id
        except Exception:
            time.sleep(2.0)
            continue

    raise TimeoutError(f"Timed out waiting for vLLM readiness at {models_url}.")


def _start_vllm_server(
    preset: ModelPreset,
    *,
    model_id: str,
    lora_adapter_path: str | None,
    lora_adapter_name: str,
) -> tuple[subprocess.Popen[str], str]:
    command = _build_vllm_command(
        preset,
        model_id=model_id,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_name=lora_adapter_name,
    )
    process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
        text=False,
    )
    try:
        served_model_id = _wait_for_server_ready(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            process=process,
            requested_model_id=model_id,
        )
        return process, served_model_id
    except Exception:
        _stop_vllm_server(process)
        raise


def _stop_vllm_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _run_experiment_with_vllm(
    *,
    preset: ModelPreset,
    limit: int | None,
    strategy: str,
    split_file: str,
    db_id: str | None,
    offset: int,
    include_evidence: bool,
    data_dictionary_mode: str,
    data_dictionary_max_values: int,
    schema_sample_rows: int,
    max_columns_per_table: int,
    ordered_result_compare: bool,
    query_timeout_seconds: float,
    query_tool_enabled: bool,
    query_tool_max_calls: int,
    query_tool_max_rows: int,
    query_tool_max_output_chars: int,
    query_tool_max_cell_chars: int,
    query_tool_timeout_seconds: float,
    model_id_override: str | None,
    lora_adapter_path: str | None,
    lora_adapter_name: str,
) -> dict:
    from bird_scaffold.runner import RunConfig, run_experiment

    resolved_model_id = model_id_override or preset.model_id
    process, served_model_id = _start_vllm_server(
        preset,
        model_id=resolved_model_id,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_name=lora_adapter_name,
    )
    try:
        config = RunConfig(
            dataset_root=REMOTE_DATASET_ROOT,
            split_file=split_file,
            strategy_name=strategy,
            model=served_model_id,
            api_base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key=VLLM_API_KEY,
            reasoning_effort=None,
            limit=limit,
            offset=offset,
            db_id=db_id,
            include_evidence=include_evidence,
            output_root=REMOTE_OUTPUT_ROOT,
            ordered_result_compare=ordered_result_compare,
            schema_sample_rows=schema_sample_rows,
            max_columns_per_table=max_columns_per_table,
            data_dictionary_mode=data_dictionary_mode,
            data_dictionary_max_values=data_dictionary_max_values,
            query_timeout_seconds=query_timeout_seconds,
            query_tool_enabled=query_tool_enabled,
            query_tool_max_calls=query_tool_max_calls,
            query_tool_max_rows=query_tool_max_rows,
            query_tool_max_output_chars=query_tool_max_output_chars,
            query_tool_max_cell_chars=query_tool_max_cell_chars,
            query_tool_timeout_seconds=query_tool_timeout_seconds,
        )
        summary = run_experiment(config)
        summary["model_preset"] = preset.name
        summary["requested_model_id"] = resolved_model_id
        summary["served_model_id"] = served_model_id
        summary["lora_adapter_path"] = lora_adapter_path
        summary["lora_adapter_name"] = lora_adapter_name if lora_adapter_path else None
        summary["vllm_base_url"] = config.api_base_url
        return summary
    finally:
        _stop_vllm_server(process)


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir("src", remote_path="/root/project/src", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml", copy=True)
    .add_local_file("README.md", remote_path="/root/project/README.md", copy=True)
    .pip_install(
        "/root/project",
        "vllm>=0.8.0",
    )
)

dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
model_cache_volume = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)
adapter_volume = modal.Volume.from_name(ADAPTER_VOLUME_NAME, create_if_missing=True)

COMMON_FUNCTION_ARGS: dict[str, object] = {
    "image": image,
    "timeout": 60 * 60 * 6,
    "volumes": {
        "/datasets": dataset_volume,
        "/outputs": output_volume,
        "/adapters": adapter_volume,
        "/root/.cache/huggingface": model_cache_volume,
    },
}


def _remote_run(
    preset_name: str,
    *,
    limit: int | None = 200,
    strategy: str = "single_shot",
    split_file: str = "dev.json",
    db_id: str | None = None,
    offset: int = 0,
    no_evidence: bool = False,
    data_dictionary_mode: str = "off",
    data_dictionary_max_values: int = 3,
    schema_sample_rows: int = 0,
    max_columns_per_table: int = 80,
    ordered_result_compare: bool = False,
    query_timeout_seconds: float = 20.0,
    no_query_tool: bool = False,
    query_tool_max_calls: int = 8,
    query_tool_max_rows: int = 50,
    query_tool_max_output_chars: int = 6000,
    query_tool_max_cell_chars: int = 200,
    query_tool_timeout_seconds: float = 8.0,
    model_id_override: str | None = None,
    lora_adapter_path: str | None = None,
    lora_adapter_name: str = "sqlrl",
) -> dict:
    preset = MODEL_PRESETS[preset_name]
    return _run_experiment_with_vllm(
        preset=preset,
        limit=limit,
        strategy=strategy,
        split_file=split_file,
        db_id=db_id,
        offset=offset,
        include_evidence=not no_evidence,
        data_dictionary_mode=data_dictionary_mode,
        data_dictionary_max_values=data_dictionary_max_values,
        schema_sample_rows=schema_sample_rows,
        max_columns_per_table=max_columns_per_table,
        ordered_result_compare=ordered_result_compare,
        query_timeout_seconds=query_timeout_seconds,
        query_tool_enabled=not no_query_tool,
        query_tool_max_calls=query_tool_max_calls,
        query_tool_max_rows=query_tool_max_rows,
        query_tool_max_output_chars=query_tool_max_output_chars,
        query_tool_max_cell_chars=query_tool_max_cell_chars,
        query_tool_timeout_seconds=query_tool_timeout_seconds,
        model_id_override=model_id_override,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_name=lora_adapter_name,
    )


@app.function(gpu=MODEL_PRESETS["qwen-8b"].gpu, **COMMON_FUNCTION_ARGS)
def run_qwen_8b_remote(
    limit: int | None = 200,
    strategy: str = "single_shot",
    split_file: str = "dev.json",
    db_id: str | None = None,
    offset: int = 0,
    no_evidence: bool = False,
    data_dictionary_mode: str = "off",
    data_dictionary_max_values: int = 3,
    schema_sample_rows: int = 0,
    max_columns_per_table: int = 80,
    ordered_result_compare: bool = False,
    query_timeout_seconds: float = 20.0,
    no_query_tool: bool = False,
    query_tool_max_calls: int = 8,
    query_tool_max_rows: int = 50,
    query_tool_max_output_chars: int = 6000,
    query_tool_max_cell_chars: int = 200,
    query_tool_timeout_seconds: float = 8.0,
    model_id_override: str | None = None,
    lora_adapter_path: str | None = None,
    lora_adapter_name: str = "sqlrl",
) -> dict:
    return _remote_run(
        "qwen-8b",
        limit=limit,
        strategy=strategy,
        split_file=split_file,
        db_id=db_id,
        offset=offset,
        no_evidence=no_evidence,
        data_dictionary_mode=data_dictionary_mode,
        data_dictionary_max_values=data_dictionary_max_values,
        schema_sample_rows=schema_sample_rows,
        max_columns_per_table=max_columns_per_table,
        ordered_result_compare=ordered_result_compare,
        query_timeout_seconds=query_timeout_seconds,
        no_query_tool=no_query_tool,
        query_tool_max_calls=query_tool_max_calls,
        query_tool_max_rows=query_tool_max_rows,
        query_tool_max_output_chars=query_tool_max_output_chars,
        query_tool_max_cell_chars=query_tool_max_cell_chars,
        query_tool_timeout_seconds=query_tool_timeout_seconds,
        model_id_override=model_id_override,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_name=lora_adapter_name,
    )


@app.function(gpu=MODEL_PRESETS["qwen-30b"].gpu, **COMMON_FUNCTION_ARGS)
def run_qwen_30b_remote(
    limit: int | None = 200,
    strategy: str = "single_shot",
    split_file: str = "dev.json",
    db_id: str | None = None,
    offset: int = 0,
    no_evidence: bool = False,
    data_dictionary_mode: str = "off",
    data_dictionary_max_values: int = 3,
    schema_sample_rows: int = 0,
    max_columns_per_table: int = 80,
    ordered_result_compare: bool = False,
    query_timeout_seconds: float = 20.0,
    no_query_tool: bool = False,
    query_tool_max_calls: int = 8,
    query_tool_max_rows: int = 50,
    query_tool_max_output_chars: int = 6000,
    query_tool_max_cell_chars: int = 200,
    query_tool_timeout_seconds: float = 8.0,
    model_id_override: str | None = None,
    lora_adapter_path: str | None = None,
    lora_adapter_name: str = "sqlrl",
) -> dict:
    return _remote_run(
        "qwen-30b",
        limit=limit,
        strategy=strategy,
        split_file=split_file,
        db_id=db_id,
        offset=offset,
        no_evidence=no_evidence,
        data_dictionary_mode=data_dictionary_mode,
        data_dictionary_max_values=data_dictionary_max_values,
        schema_sample_rows=schema_sample_rows,
        max_columns_per_table=max_columns_per_table,
        ordered_result_compare=ordered_result_compare,
        query_timeout_seconds=query_timeout_seconds,
        no_query_tool=no_query_tool,
        query_tool_max_calls=query_tool_max_calls,
        query_tool_max_rows=query_tool_max_rows,
        query_tool_max_output_chars=query_tool_max_output_chars,
        query_tool_max_cell_chars=query_tool_max_cell_chars,
        query_tool_timeout_seconds=query_tool_timeout_seconds,
        model_id_override=model_id_override,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_name=lora_adapter_name,
    )


@app.local_entrypoint()
def main(
    model_preset: str = "qwen-8b",
    limit: int | None = 200,
    strategy: str = "single_shot",
    split_file: str = "dev.json",
    db_id: str | None = None,
    offset: int = 0,
    no_evidence: bool = False,
    data_dictionary_mode: str = "off",
    data_dictionary_max_values: int = 3,
    schema_sample_rows: int = 0,
    max_columns_per_table: int = 80,
    ordered_result_compare: bool = False,
    query_timeout_seconds: float = 20.0,
    no_query_tool: bool = False,
    query_tool_max_calls: int = 8,
    query_tool_max_rows: int = 50,
    query_tool_max_output_chars: int = 6000,
    query_tool_max_cell_chars: int = 200,
    query_tool_timeout_seconds: float = 8.0,
    model_id_override: str | None = None,
    lora_adapter_path: str | None = None,
    lora_adapter_name: str = "sqlrl",
) -> None:
    if model_preset not in MODEL_PRESETS:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unsupported --model-preset '{model_preset}'. Supported presets: {supported}")

    kwargs = {
        "limit": limit,
        "strategy": strategy,
        "split_file": split_file,
        "db_id": db_id,
        "offset": offset,
        "no_evidence": no_evidence,
        "data_dictionary_mode": data_dictionary_mode,
        "data_dictionary_max_values": data_dictionary_max_values,
        "schema_sample_rows": schema_sample_rows,
        "max_columns_per_table": max_columns_per_table,
        "ordered_result_compare": ordered_result_compare,
        "query_timeout_seconds": query_timeout_seconds,
        "no_query_tool": no_query_tool,
        "query_tool_max_calls": query_tool_max_calls,
        "query_tool_max_rows": query_tool_max_rows,
        "query_tool_max_output_chars": query_tool_max_output_chars,
        "query_tool_max_cell_chars": query_tool_max_cell_chars,
        "query_tool_timeout_seconds": query_tool_timeout_seconds,
        "model_id_override": model_id_override,
        "lora_adapter_path": lora_adapter_path,
        "lora_adapter_name": lora_adapter_name,
    }

    if model_preset == "qwen-8b":
        summary = run_qwen_8b_remote.remote(**kwargs)
    else:
        summary = run_qwen_30b_remote.remote(**kwargs)

    print(json.dumps(summary, indent=2))
