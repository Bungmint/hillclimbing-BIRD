"""Modal entrypoint for parallel Tinker GRPO hyperparameter sweeps.

Runs all combinations from SWEEP_GRID in parallel on Modal containers.
Each run gets a unique log directory on the shared output volume.

Prerequisites (same as modal_tinker_run.py):
  1. Upload dataset:   modal volume put bird-dev-20240627 dev_20240627/ /dev_20240627
  2. Create secrets:   modal secret create tinker-credentials TINKER_API_KEY=<key> WANDB_API_KEY=<key>

Usage:
  # Run the default sweep grid
  modal run scripts/modal_tinker_sweep.py

  # Override base config values
  modal run scripts/modal_tinker_sweep.py --model Qwen/Qwen3-32B --train-limit 512

Edit SWEEP_GRID below to change which hyperparameters are swept.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Sweep configuration — edit these to customise your sweep
# ---------------------------------------------------------------------------

SWEEP_GRID: dict[str, list] = {
    "learning_rate": [1e-5, 3e-5],
    "group_size": [8, 16],
    "temperature": [0.8, 1.0],
}

BASE_CONFIG: dict[str, object] = {
    "model": "Qwen/Qwen3-8B",
    "train_limit": 256,
    "eval_limit": 64,
    "groups_per_batch": 16,
    "lora_rank": 32,
    "loss_fn": "importance_sampling",
    "reward_exec_match": 1.0,
    "eval_every": 10,
    "save_every": 10,
}

# ---------------------------------------------------------------------------
# Modal setup — reuses the same image / volumes as the single-run script
# ---------------------------------------------------------------------------

APP_NAME = "bird-tinker-sweep"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_sweep_configs(
    base: dict[str, object],
    grid: dict[str, list],
) -> list[dict[str, object]]:
    """Cartesian product of grid values merged onto the base config."""
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    configs: list[dict[str, object]] = []
    for combo in itertools.product(*value_lists):
        override = dict(zip(keys, combo))
        merged = {**base, **override}
        configs.append(merged)
    return configs


def _sweep_tag(config: dict[str, object], grid_keys: list[str]) -> str:
    """Short human-readable tag from the swept values, e.g. 'lr1e-05_gs8_t0.8'."""
    abbrevs = {
        "learning_rate": "lr",
        "group_size": "gs",
        "groups_per_batch": "gpb",
        "lora_rank": "lora",
        "temperature": "t",
        "loss_fn": "loss",
        "kl_penalty_coef": "kl",
        "reward_exec_match": "rexec",
        "reward_executable": "rexe",
        "num_substeps": "sub",
    }
    parts: list[str] = []
    for key in grid_keys:
        short = abbrevs.get(key, key[:4])
        parts.append(f"{short}{config[key]}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    cpu=8,
    memory=8192,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("tinker-credentials")],
    volumes={
        "/datasets": dataset_volume,
        "/outputs": output_volume,
    },
)
def run_sweep_arm(config: dict[str, object], log_path: str) -> dict:
    """Run a single arm of the sweep."""
    from bird_scaffold.tinker_grpo import TinkerGRPOConfig, run_tinker_grpo_training

    grpo_config = TinkerGRPOConfig(
        dataset_root=REMOTE_DATASET_ROOT,
        split_file=str(config.get("split_file", "dev.json")),
        model=str(config.get("model", "Qwen/Qwen3-8B")),
        db_id=config.get("db_id"),  # type: ignore[arg-type]
        train_offset=int(config.get("train_offset", 0)),
        train_limit=int(config.get("train_limit", 256)),
        eval_offset=config.get("eval_offset"),  # type: ignore[arg-type]
        eval_limit=int(config.get("eval_limit", 64)),
        shuffle=bool(config.get("shuffle", True)),
        seed=int(config.get("seed", 0)),
        group_size=int(config.get("group_size", 8)),
        groups_per_batch=int(config.get("groups_per_batch", 16)),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        max_tokens=int(config.get("max_tokens", 4096)),
        temperature=float(config.get("temperature", 1.0)),
        lora_rank=int(config.get("lora_rank", 32)),
        loss_fn=str(config.get("loss_fn", "importance_sampling")),
        kl_penalty_coef=float(config.get("kl_penalty_coef", 0.0)),
        kl_discount_factor=float(config.get("kl_discount_factor", 0.0)),
        compute_post_kl=bool(config.get("compute_post_kl", False)),
        num_substeps=int(config.get("num_substeps", 1)),
        stream_minibatch=bool(config.get("stream_minibatch", False)),
        num_minibatches=int(config.get("num_minibatches", 4)),
        max_steps_off_policy=config.get("max_steps_off_policy"),  # type: ignore[arg-type]
        reward_exec_match=float(config.get("reward_exec_match", 1.0)),
        reward_executable=float(config.get("reward_executable", 0.0)),
        reward_exact_sql=float(config.get("reward_exact_sql", 0.0)),
        eval_every=int(config.get("eval_every", 10)),
        save_every=int(config.get("save_every", 10)),
        query_tool_enabled=bool(config.get("query_tool_enabled", True)),
        query_tool_max_calls=int(config.get("query_tool_max_calls", 8)),
        max_turns=config.get("max_turns"),  # type: ignore[arg-type]
        max_trajectory_tokens=int(config.get("max_trajectory_tokens", 32768)),
        failed_parse_reward=float(config.get("failed_parse_reward", -0.1)),
        terminate_on_parse_error=bool(config.get("terminate_on_parse_error", True)),
        log_path=Path(log_path),
        load_checkpoint_path=config.get("load_checkpoint_path"),  # type: ignore[arg-type]
        base_url=config.get("base_url"),  # type: ignore[arg-type]
        wandb_project=str(config.get("wandb_project", "bird-grpo-sweep")),
        wandb_name=config.get("wandb_name"),  # type: ignore[arg-type]
    )

    summary = run_tinker_grpo_training(grpo_config)
    summary["sweep_config"] = {k: config[k] for k in SWEEP_GRID}
    output_volume.commit()
    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-8B",
    train_limit: int = 256,
    eval_limit: int = 64,
    dry_run: bool = False,
) -> None:
    base = {**BASE_CONFIG, "model": model, "train_limit": train_limit, "eval_limit": eval_limit}
    configs = _generate_sweep_configs(base, SWEEP_GRID)
    grid_keys = list(SWEEP_GRID.keys())

    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = model.replace("/", "-")

    print(f"Sweep {sweep_id}: {len(configs)} configurations")
    for i, cfg in enumerate(configs):
        tag = _sweep_tag(cfg, grid_keys)
        print(f"  [{i}] {tag}")

    if dry_run:
        print("\n--dry-run: printing configs and exiting.")
        print(json.dumps(configs, indent=2, default=str))
        return

    # Build (config, log_path) pairs for starmap.
    args: list[tuple[dict, str]] = []
    for cfg in configs:
        tag = _sweep_tag(cfg, grid_keys)
        log_path = str(REMOTE_OUTPUT_ROOT / f"sweep_{sweep_id}_{model_tag}" / tag)
        cfg["wandb_name"] = f"sweep-{sweep_id}-{tag}"
        args.append((cfg, log_path))

    # Launch all arms in parallel.
    results: list[dict] = []
    for summary in run_sweep_arm.starmap(args):
        results.append(summary)

    # Print results table.
    print(f"\n{'=' * 80}")
    print(f"SWEEP RESULTS — {len(results)} / {len(configs)} completed")
    print(f"{'=' * 80}")

    header_keys = grid_keys + ["train/exec_match", "eval/exec_match", "train/loss", "log_path"]
    print("  ".join(f"{k:>20s}" for k in header_keys))
    print("-" * (22 * len(header_keys)))

    for summary in results:
        sweep_cfg = summary.get("sweep_config", {})
        row: list[str] = []
        for k in grid_keys:
            row.append(f"{sweep_cfg.get(k, '?'):>20}")
        for k in ["train/exec_match", "eval/exec_match", "train/loss"]:
            val = summary.get(k)
            row.append(f"{val:>20.4f}" if isinstance(val, (int, float)) else f"{'n/a':>20}")
        row.append(f"{summary.get('log_path', '?'):>20}")
        print("  ".join(row))

    # Dump full results JSON.
    sweep_results_path = f"sweep_{sweep_id}_results.json"
    with open(sweep_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {sweep_results_path}")
