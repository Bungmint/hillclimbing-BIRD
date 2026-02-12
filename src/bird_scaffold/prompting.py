from __future__ import annotations

from bird_scaffold.types import BirdExample


SYSTEM_PROMPT = """You are an expert SQLite text-to-SQL system for the BIRD benchmark.
Write exactly one executable SQLite SQL query.
Rules:
- Output SQL only, no markdown and no explanation.
- Use only tables and columns from the provided schema.
- Respect SQLite syntax.
- If evidence is provided, use it as a hint.
"""


def build_user_prompt(
    example: BirdExample,
    schema_text: str,
    include_evidence: bool = True,
    data_dictionary_text: str | None = None,
) -> str:
    sections: list[str] = []
    sections.append(f"Database ID: {example.db_id}")
    sections.append("Schema:\n" + schema_text)
    if data_dictionary_text:
        sections.append("DB Notes (data dictionary):\n" + data_dictionary_text)
    sections.append("Question:\n" + example.question)

    if include_evidence and example.evidence.strip():
        sections.append("Evidence:\n" + example.evidence.strip())

    sections.append("Return only SQL.")
    return "\n\n".join(sections)
