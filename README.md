# BIRD Scaffold (Local + Modal-Ready)

Local-first scaffold for BIRD-style text-to-SQL experiments on `dev_20240627`.

## What this gives you

- Baseline runner: `question -> OpenAI model -> SQL -> execute -> evaluate`
- Execution-accuracy style metrics against BIRD gold SQL
- Strategy interface so you can add new ideas without changing the core loop
- Modal entrypoint reusing the same runner for larger experiments later

## Quick Start (MacBook local)

1. Create env and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

2. Set API key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

3. Preview dataset wiring:

```bash
bird-scaffold preview --dataset-root dev_20240627
```

4. Run a quick baseline sample:

```bash
bird-scaffold run \
  --dataset-root dev_20240627 \
  --model gpt-4.1-mini \
  --strategy single_shot \
  --data-dictionary-mode off \
  --limit 20
```

Run Idea 1A (schema + data dictionary) with the same baseline strategy:

```bash
bird-scaffold run \
  --dataset-root dev_20240627 \
  --model gpt-4.1-mini \
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
```

## Extend with new project ideas

Add a strategy in `src/bird_scaffold/strategies/`:

1. Subclass `SQLGenerationStrategy`
2. Register it in `src/bird_scaffold/strategies/__init__.py`
3. Run with `--strategy <your_name>`

This keeps data loading, execution, and evaluation unchanged while you iterate on prompting, repair, reranking, or self-consistency.

## Data Dictionary Modes (Idea 1A)

- `--data-dictionary-mode off`: baseline schema-only prompt (default)
- `--data-dictionary-mode stats`: schema + per-column stats (no sample values)
- `--data-dictionary-mode stats_and_samples`: schema + stats + representative values

This is designed for direct ablations without changing strategy code.

## Modal later

`modal_app.py` provides a starter for remote execution with the same runner.

- Keep local development with `bird-scaffold run`
- Move heavier sweeps to Modal by copying dataset to a Modal Volume and setting a Modal Secret for `OPENAI_API_KEY`
