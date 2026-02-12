from __future__ import annotations

from bird_scaffold.strategies.base import SQLGenerationStrategy
from bird_scaffold.strategies.gold_oracle import GoldOracleStrategy
from bird_scaffold.strategies.single_shot import SingleShotStrategy


_STRATEGY_REGISTRY: dict[str, type[SQLGenerationStrategy]] = {
    SingleShotStrategy.name: SingleShotStrategy,
    GoldOracleStrategy.name: GoldOracleStrategy,
}


def get_strategy(name: str) -> SQLGenerationStrategy:
    strategy_cls = _STRATEGY_REGISTRY.get(name)
    if strategy_cls is None:
        available = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return strategy_cls()


def list_strategies() -> list[str]:
    return sorted(_STRATEGY_REGISTRY)
