from __future__ import annotations

from bird_scaffold.prompting import build_system_prompt, build_user_prompt
from bird_scaffold.strategies.base import SQLGenerationStrategy
from bird_scaffold.types import GenerationResult


class SingleShotStrategy(SQLGenerationStrategy):
    name = "single_shot"
    requires_llm = True

    def generate(self, example, db_context, llm_client, include_evidence: bool) -> GenerationResult:
        if llm_client is None:
            return GenerationResult(sql="", error="LLM client is required for single_shot strategy")

        user_prompt = build_user_prompt(
            example=example,
            schema_text=db_context.schema_text,
            include_evidence=include_evidence,
            data_dictionary_text=db_context.data_dictionary_text,
            enable_query_tool=llm_client.query_tool_enabled,
        )

        try:
            sql, raw_output, latency_s, query_tool_calls = llm_client.generate_sql(
                system_prompt=build_system_prompt(enable_query_tool=llm_client.query_tool_enabled),
                user_prompt=user_prompt,
                db_path=db_context.db_path,
            )
            return GenerationResult(
                sql=sql,
                raw_output=raw_output,
                latency_s=latency_s,
                query_tool_calls=query_tool_calls,
            )
        except Exception as exc:
            return GenerationResult(sql="", error=str(exc))
