from __future__ import annotations

import json
from pathlib import Path

from bird_scaffold.types import BirdExample


def load_examples(dataset_root: Path, split_file: str = "dev.json") -> list[BirdExample]:
    split_path = dataset_root / split_file
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    raw_items = json.loads(split_path.read_text(encoding="utf-8"))
    return [BirdExample.from_dict(item) for item in raw_items]


def filter_examples(
    examples: list[BirdExample],
    db_id: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[BirdExample]:
    filtered = examples
    if db_id:
        filtered = [item for item in filtered if item.db_id == db_id]

    if offset > 0:
        filtered = filtered[offset:]

    if limit is not None:
        filtered = filtered[:limit]

    return filtered
