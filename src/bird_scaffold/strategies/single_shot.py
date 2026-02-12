from __future__ import annotations

from bird_scaffold.prompting import SYSTEM_PROMPT, build_user_prompt
from bird_scaffold.strategies.base import SQLGenerationStrategy
from bird_scaffold.types import GenerationResult


class SingleShotStrategy(SQLGenerationStrategy):
    name = "single_shot"
    requires_openai = True

    def generate(self, example, db_context, openai_client, include_evidence: bool) -> GenerationResult:
        if openai_client is None:
            return GenerationResult(sql="", error="OpenAI client is required for single_shot strategy")

        user_prompt = build_user_prompt(
            example=example,
            schema_text=db_context.schema_text,
            include_evidence=include_evidence,
            data_dictionary_text=db_context.data_dictionary_text,
        )

        try:
            sql, raw_output, latency_s = openai_client.generate_sql(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            return GenerationResult(sql=sql, raw_output=raw_output, latency_s=latency_s)
        except Exception as exc:
            return GenerationResult(sql="", error=str(exc))
