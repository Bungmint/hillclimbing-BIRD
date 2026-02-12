from __future__ import annotations

import re


_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_READ_ONLY_PREFIXES = ("select", "with", "pragma", "explain")
_MUTATING_KEYWORDS_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|reindex|analyze|begin|commit|rollback|savepoint|release)\b",
    re.IGNORECASE,
)
_LITERAL_RE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")


def extract_sql(text: str) -> str:
    candidate = text.strip()

    code_match = _CODE_BLOCK_RE.search(candidate)
    if code_match:
        candidate = code_match.group(1).strip()

    statements = split_sql_statements(candidate)
    if statements:
        return statements[0].strip()

    return candidate.strip()


def split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    in_backtick = False
    i = 0

    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if not (in_single or in_double or in_backtick):
            if ch == "-" and nxt == "-":
                i += 2
                while i < len(sql) and sql[i] != "\n":
                    i += 1
                continue
            if ch == "/" and nxt == "*":
                i += 2
                while i + 1 < len(sql) and not (sql[i] == "*" and sql[i + 1] == "/"):
                    i += 1
                i = min(i + 2, len(sql))
                continue
            if ch == ";":
                statement = "".join(current).strip()
                if statement:
                    statements.append(statement)
                current = []
                i += 1
                continue
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "`":
                in_backtick = True
        else:
            if in_single and ch == "'":
                if nxt == "'":
                    current.append(ch)
                    current.append(nxt)
                    i += 2
                    continue
                in_single = False
            elif in_double and ch == '"':
                if nxt == '"':
                    current.append(ch)
                    current.append(nxt)
                    i += 2
                    continue
                in_double = False
            elif in_backtick and ch == "`":
                in_backtick = False

        current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def is_read_only_sql(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped:
        return False
    lowered = stripped.lower().lstrip("(")
    if not lowered.startswith(_READ_ONLY_PREFIXES):
        return False

    # Ignore quoted literals before scanning for mutating keywords.
    literal_free = _LITERAL_RE.sub("''", stripped)
    return _MUTATING_KEYWORDS_RE.search(literal_free) is None
