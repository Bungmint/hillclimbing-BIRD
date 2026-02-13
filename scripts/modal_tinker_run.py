"""Modal entrypoint for a single Tinker GRPO training run on BIRD.

The training orchestrator is CPU-only — Tinker's cloud handles all GPU
compute (forward/backward passes, optimizer steps).  Running on Modal:
  - Cloud-to-cloud networking to Tinker servers (lower latency vs home)
  - Frees up your local machine
  - Outputs persist on a shared Modal volume

Prerequisites:
  1. Upload dataset (one-time):
       modal volume put bird-dev-20240627 dev_20240627/ /dev_20240627
  2. Create secrets (one-time):
       modal secret create tinker-credentials TINKER_API_KEY=<key> WANDB_API_KEY=<key>

Examples:
  modal run scripts/modal_tinker_run.py
  modal run scripts/modal_tinker_run.py --model Qwen/Qwen3-8B --train-limit 512
  modal run scripts/modal_tinker_run.py --learning-rate 3e-5 --group-size 16
  modal run scripts/modal_tinker_run.py --loss-fn ppo --kl-penalty-coef 0.01
  modal run scripts/modal_tinker_run.py --load-checkpoint-path "tinker://UUID:train:0/weights/000010"
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "bird-tinker-grpo"
DATASET_VOLUME_NAME = "bird-dev-20240627"
OUTPUT_VOLUME_NAME = "bird-outputs"

REMOTE_DATASET_ROOT = Path("/datasets/dev_20240627")
REMOTE_OUTPUT_ROOT = Path("/outputs/tinker_grpo")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir("src", remote_path="/root/project/src", copy=True)
    .add_local_dir("tinker-cookbook", remote_path="/root/project/tinker-cookbook", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml", copy=True)
    .add_local_file("README.md", remote_path="/root/project/README.md", copy=True)
    .pip_install("/root/project/tinker-cookbook", "/root/project[rl]")
)

dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

COMMON_FUNCTION_KWARGS: dict[str, object] = {
    "image": image,
    "cpu": 8,
    "memory": 8192,
    "timeout": 60 * 60 * 24,  # 24h — training runs can be long
    "secrets": [modal.Secret.from_name("tinker-credentials")],
    "volumes": {
        "/datasets": dataset_volume,
        "/outputs": output_volume,
    },
}


@app.function(**COMMON_FUNCTION_KWARGS)
def run_grpo_remote(
    model: str = "Qwen/Qwen3-8B",
    split_file: str = "dev.json",
    db_id: str | None = None,
    train_offset: int = 0,
    train_limit: int = 256,
    eval_offset: int | None = None,
    eval_limit: int = 64,
    no_shuffle: bool = False,
    seed: int = 0,
    group_size: int = 8,
    groups_per_batch: int = 16,
    learning_rate: float = 1e-5,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    lora_rank: int = 32,
    loss_fn: str = "importance_sampling",
    kl_penalty_coef: float = 0.0,
    kl_discount_factor: float = 0.0,
    compute_post_kl: bool = False,
    num_substeps: int = 1,
    stream_minibatch: bool = False,
    num_minibatches: int = 4,
    max_steps_off_policy: int | None = None,
    reward_exec_match: float = 1.0,
    reward_executable: float = 0.0,
    reward_exact_sql: float = 0.0,
    eval_every: int = 10,
    save_every: int = 10,
    no_query_tool: bool = False,
    query_tool_max_calls: int = 8,
    max_turns: int | None = None,
    max_trajectory_tokens: int = 32768,
    failed_parse_reward: float = -0.1,
    continue_on_parse_error: bool = False,
    load_checkpoint_path: str | None = None,
    base_url: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    # Internal: override log_path (used by sweep script).
    _log_path: str | None = None,
) -> dict:
    from bird_scaffold.tinker_grpo import TinkerGRPOConfig, run_tinker_grpo_training

    if _log_path is not None:
        log_path = Path(_log_path)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_tag = model.replace("/", "-")
        log_path = REMOTE_OUTPUT_ROOT / f"{timestamp}_{model_tag}"

    config = TinkerGRPOConfig(
        dataset_root=REMOTE_DATASET_ROOT,
        split_file=split_file,
        model=model,
        db_id=db_id,
        train_offset=train_offset,
        train_limit=train_limit,
        eval_offset=eval_offset,
        eval_limit=eval_limit,
        shuffle=not no_shuffle,
        seed=seed,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        temperature=temperature,
        lora_rank=lora_rank,
        loss_fn=loss_fn,
        kl_penalty_coef=kl_penalty_coef,
        kl_discount_factor=kl_discount_factor,
        compute_post_kl=compute_post_kl,
        num_substeps=num_substeps,
        stream_minibatch=stream_minibatch,
        num_minibatches=num_minibatches,
        max_steps_off_policy=max_steps_off_policy,
        reward_exec_match=reward_exec_match,
        reward_executable=reward_executable,
        reward_exact_sql=reward_exact_sql,
        eval_every=eval_every,
        save_every=save_every,
        query_tool_enabled=not no_query_tool,
        query_tool_max_calls=query_tool_max_calls,
        max_turns=max_turns,
        max_trajectory_tokens=max_trajectory_tokens,
        failed_parse_reward=failed_parse_reward,
        terminate_on_parse_error=not continue_on_parse_error,
        log_path=log_path,
        load_checkpoint_path=load_checkpoint_path,
        base_url=base_url,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    summary = run_tinker_grpo_training(config)
    output_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-8B",
    split_file: str = "dev.json",
    db_id: str | None = None,
    train_offset: int = 0,
    train_limit: int = 256,
    eval_offset: int | None = None,
    eval_limit: int = 64,
    no_shuffle: bool = False,
    seed: int = 0,
    group_size: int = 8,
    groups_per_batch: int = 16,
    learning_rate: float = 1e-5,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    lora_rank: int = 32,
    loss_fn: str = "importance_sampling",
    kl_penalty_coef: float = 0.0,
    kl_discount_factor: float = 0.0,
    compute_post_kl: bool = False,
    num_substeps: int = 1,
    stream_minibatch: bool = False,
    num_minibatches: int = 4,
    max_steps_off_policy: int | None = None,
    reward_exec_match: float = 1.0,
    reward_executable: float = 0.0,
    reward_exact_sql: float = 0.0,
    eval_every: int = 10,
    save_every: int = 10,
    no_query_tool: bool = False,
    query_tool_max_calls: int = 8,
    max_turns: int | None = None,
    max_trajectory_tokens: int = 32768,
    failed_parse_reward: float = -0.1,
    continue_on_parse_error: bool = False,
    load_checkpoint_path: str | None = None,
    base_url: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> None:
    summary = run_grpo_remote.remote(
        model=model,
        split_file=split_file,
        db_id=db_id,
        train_offset=train_offset,
        train_limit=train_limit,
        eval_offset=eval_offset,
        eval_limit=eval_limit,
        no_shuffle=no_shuffle,
        seed=seed,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        temperature=temperature,
        lora_rank=lora_rank,
        loss_fn=loss_fn,
        kl_penalty_coef=kl_penalty_coef,
        kl_discount_factor=kl_discount_factor,
        compute_post_kl=compute_post_kl,
        num_substeps=num_substeps,
        stream_minibatch=stream_minibatch,
        num_minibatches=num_minibatches,
        max_steps_off_policy=max_steps_off_policy,
        reward_exec_match=reward_exec_match,
        reward_executable=reward_executable,
        reward_exact_sql=reward_exact_sql,
        eval_every=eval_every,
        save_every=save_every,
        no_query_tool=no_query_tool,
        query_tool_max_calls=query_tool_max_calls,
        max_turns=max_turns,
        max_trajectory_tokens=max_trajectory_tokens,
        failed_parse_reward=failed_parse_reward,
        continue_on_parse_error=continue_on_parse_error,
        load_checkpoint_path=load_checkpoint_path,
        base_url=base_url,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    print(json.dumps(summary, indent=2))
