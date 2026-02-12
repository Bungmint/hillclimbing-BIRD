# BIRD Scaffold (Local + Modal-Ready)

Local-first scaffold for BIRD-style text-to-SQL experiments on `dev_20240627`.

## What this gives you

- Baseline runner: `question -> OpenAI-compatible model (+ query tool calls) -> SQL -> execute -> evaluate`
- Execution-accuracy style metrics against BIRD gold SQL
- Strategy interface so you can add new ideas without changing the core loop
- Tinker GRPO training entrypoint for trajectory-outcome rewards
- Modal entrypoint reusing the same runner for larger experiments later

## Quick Start (MacBook local)

1. Create env and install:

```bash
uv sync
```

2. Point to a model endpoint (vLLM local example):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

3. Preview dataset wiring:

```bash
bird-scaffold preview --dataset-root dev_20240627
```

4. Run a quick baseline sample:

```bash
bird-scaffold run \
  --dataset-root dev_20240627 \
  --model Qwen/Qwen3-8B \
  --api-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --strategy single_shot \
  --data-dictionary-mode off \
  --limit 20
```

Run Idea 1A (schema + data dictionary) with the same baseline strategy:

```bash
bird-scaffold run \
  --dataset-root dev_20240627 \
  --model Qwen/Qwen3-8B \
  --api-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --strategy single_shot \
  --data-dictionary-mode stats_and_samples \
  --limit 20
```

5. Smoke test the pipeline offline (no API call):

```bash
bird-scaffold run \
  --dataset-root dev_20240627 \
  --strategy gold_oracle \
  --limit 20
```

Outputs are written to `outputs/<timestamp>_<strategy>_<model>/` with:

- `summary.json`
- `predictions.jsonl`
- `run_config.json`

## Useful Commands

```bash
bird-scaffold strategies
bird-scaffold show-schema --dataset-root dev_20240627 --db-id financial
bird-scaffold rl-train --help
```

## Extend with new project ideas

Add a strategy in `src/bird_scaffold/strategies/`:

1. Subclass `SQLGenerationStrategy`
2. Register it in `src/bird_scaffold/strategies/__init__.py`
3. Run with `--strategy <your_name>`

This keeps data loading, execution, and evaluation unchanged while you iterate on prompting, repair, reranking, or self-consistency.

## Query Tool (default on)

`single_shot` now enables a function tool named `query(sql)` by default.

- Tool queries run read-only against the current example's SQLite database.
- Tool results are truncated by row count and output size for prompt safety.
- Final answer is still a single SQL query returned by the model.

Useful flags for production control:

- `--no-query-tool`
- `--query-tool-max-calls`
- `--query-tool-max-rows`
- `--query-tool-max-output-chars`
- `--query-tool-max-cell-chars`
- `--query-tool-timeout`

## Data Dictionary Modes (Idea 1A)

- `--data-dictionary-mode off`: baseline schema-only prompt (default)
- `--data-dictionary-mode stats`: schema + per-column stats (no sample values)
- `--data-dictionary-mode stats_and_samples`: schema + stats + representative values

This is designed for direct ablations without changing strategy code.

## Modal vLLM

`modal_app.py` now starts vLLM in-container and runs the same experiment loop against it.

Supported presets:

- `qwen-8b` -> `Qwen/Qwen3-8B`
- `qwen-30b` -> `Qwen/Qwen3-30B-A3B`

Example:

```bash
modal run modal_app.py --model-preset qwen-8b --limit 200 --strategy single_shot
```

You can also serve a fine-tuned checkpoint/adapter:

```bash
modal run modal_app.py \
  --model-preset qwen-8b \
  --model-id-override Qwen/Qwen3-8B \
  --lora-adapter-path /adapters/your_grpo_adapter \
  --lora-adapter-name bird-grpo \
  --limit 200 \
  --strategy single_shot
```

## Tinker GRPO (Trajectory Outcome Reward)

Install RL dependencies first:

```bash
uv pip install tinker tinker_cookbook chz
```

Run vanilla GRPO with execution-outcome reward (default model is `Qwen/Qwen3-8B`):

```bash
bird-scaffold rl-train \
  --dataset-root dev_20240627 \
  --split-file dev.json \
  --model Qwen/Qwen3-8B \
  --group-size 8 \
  --groups-per-batch 16 \
  --train-limit 256 \
  --eval-limit 64 \
  --reward-exec-match 1.0 \
  --reward-executable 0.0 \
  --reward-exact-sql 0.0
```

Training logs/checkpoint metadata are written under `outputs/tinker_grpo/...`.
Use the saved adapter/checkpoint from that run for Modal serving and eval.
