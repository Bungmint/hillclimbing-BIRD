"""Modal entrypoint for scaling BIRD runs.

Usage (after setting up Modal secrets/volume):
  modal run modal_app.py --limit 200 --model gpt-4.1-mini --strategy single_shot
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "bird-scaffold"
DATASET_VOLUME_NAME = "bird-dev-20240627"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(".")

# Assumes you copied dev_20240627 into this volume at /datasets/dev_20240627
volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("openai-api-key")],
    volumes={"/datasets": volume},
)
def run_remote(limit: int = 200, model: str = "gpt-4.1-mini", strategy: str = "single_shot") -> dict:
    from bird_scaffold.runner import RunConfig, run_experiment

    config = RunConfig(
        dataset_root=Path("/datasets/dev_20240627"),
        split_file="dev.json",
        strategy_name=strategy,
        model=model,
        limit=limit,
        output_root=Path("/tmp/bird_outputs"),
    )
    return run_experiment(config)


@app.local_entrypoint()
def main(limit: int = 200, model: str = "gpt-4.1-mini", strategy: str = "single_shot") -> None:
    summary = run_remote.remote(limit=limit, model=model, strategy=strategy)
    print(summary)
