from __future__ import annotations

from bird_scaffold.types import BirdExample


_BASE_SYSTEM_PROMPT = """You are an expert SQLite text-to-SQL system for the BIRD benchmark.
Write exactly one executable SQLite SQL query.
Rules:
- Output your final SQL inside a ```sql code block.
- Use only tables and columns from the provided schema.
- Respect SQLite syntax.
- If evidence is provided, use it as a hint.
"""


_QUERY_TOOL_INSTRUCTIONS = """
You may call the `query` tool to explore the current database before finalizing SQL.
Tool usage rules:
- Use the tool for lightweight exploration only (for example: row samples, cardinality checks, DISTINCT values).
- Keep exploration queries read-only and selective (prefer LIMIT and focused projections).
- After exploration, return the final SQL answer in a ```sql code block.
"""


def build_system_prompt(enable_query_tool: bool) -> str:
    if not enable_query_tool:
        return _BASE_SYSTEM_PROMPT
    return _BASE_SYSTEM_PROMPT + _QUERY_TOOL_INSTRUCTIONS


def build_user_prompt(
    example: BirdExample,
    schema_text: str,
    include_evidence: bool = True,
    data_dictionary_text: str | None = None,
    enable_query_tool: bool = False,
) -> str:
    sections: list[str] = []
    sections.append(f"Database ID: {example.db_id}")
    sections.append("Schema:\n" + schema_text)
    if data_dictionary_text:
        sections.append("DB Notes (data dictionary):\n" + data_dictionary_text)
    sections.append("Question:\n" + example.question)

    if include_evidence and example.evidence.strip():
        sections.append("Evidence:\n" + example.evidence.strip())

    if enable_query_tool:
        sections.append(
            "Use `query` tool calls for exploration when needed, then return only the final SQL."
        )
    else:
        sections.append("Return only SQL.")
    return "\n\n".join(sections)
