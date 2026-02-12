from __future__ import annotations

from bird_scaffold.strategies.base import SQLGenerationStrategy
from bird_scaffold.types import GenerationResult


class GoldOracleStrategy(SQLGenerationStrategy):
    name = "gold_oracle"
    requires_llm = False

    def generate(self, example, db_context, llm_client, include_evidence: bool) -> GenerationResult:
        del db_context, llm_client, include_evidence
        return GenerationResult(sql=example.gold_sql, raw_output=example.gold_sql, latency_s=0.0)
