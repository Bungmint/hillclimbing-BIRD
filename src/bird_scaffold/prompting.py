from __future__ import annotations

from bird_scaffold.types import BirdExample


_BASE_INPUT_PROMPT_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```
"""


_EVIDENCE_SECTION = """
Evidence:
{evidence}
"""


_QUERY_TOOL_INSTRUCTIONS = """
You may call the `query` tool to explore the current database before finalizing SQL.
Tool usage rules:
- Use the tool for lightweight exploration only (for example: row samples, cardinality checks, DISTINCT values).
- Keep exploration queries read-only and selective (prefer LIMIT and focused projections).
- After exploration, return the final SQL answer in a ```sql code block.
"""


_BASE_SYSTEM_PROMPT = "You are a data science expert specializing in SQL query generation."


def build_system_prompt(enable_query_tool: bool) -> str:
    if not enable_query_tool:
        return _BASE_SYSTEM_PROMPT
    return _BASE_SYSTEM_PROMPT + "\n" + _QUERY_TOOL_INSTRUCTIONS


def build_user_prompt(
    example: BirdExample,
    schema_text: str,
    include_evidence: bool = False,
    data_dictionary_text: str | None = None,
    enable_query_tool: bool = False,
) -> str:
    prompt = _BASE_INPUT_PROMPT_TEMPLATE.format(
        db_details=schema_text,
        question=example.question,
    )

    if include_evidence and example.evidence.strip():
        prompt += _EVIDENCE_SECTION.format(evidence=example.evidence.strip())

    if data_dictionary_text:
        prompt += f"\nDB Notes (data dictionary):\n{data_dictionary_text}\n"

    if enable_query_tool:
        prompt += "\nUse `query` tool calls for exploration when needed, then return only the final SQL.\n"

    return prompt
