from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import chz
import tinker
from tinker.types import LossFnType
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
from tinker_cookbook.rl.train import AsyncConfig, Config as RLTrainConfig, StreamMinibatchConfig
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv
from tinker_cookbook.tool_use.tools import handle_tool_call, simple_tool_result
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult
from tinker_cookbook.utils import ml_log

from bird_scaffold.data_dictionary import DATA_DICTIONARY_MODE_STATS_AND_SAMPLES
from bird_scaffold.dataset import filter_examples, load_examples
from bird_scaffold.db import load_database_context
from bird_scaffold.execution import execute_sql, normalize_sql, results_match
from bird_scaffold.prompting import build_system_prompt, build_user_prompt
from bird_scaffold.query_tool import (
    QUERY_TOOL_NAME,
    QueryToolConfig,
    QueryToolExecutor,
    build_query_tool_schema,
    parse_query_tool_arguments,
)
from bird_scaffold.sql_parsing import extract_sql
from bird_scaffold.types import BirdExample, DatabaseContext

logger = logging.getLogger(__name__)

_TOOL_SUPPORT_HINT = (
    "Use tool calls with this exact format: "
    '<tool_call>{"name":"query","args":{"sql":"SELECT ..."}}</tool_call>. '
    "After tool responses, return the final SQL in a ```sql``` block."
)


@dataclass(frozen=True)
class RewardConfig:
    exec_match: float = 1.0
    executable: float = 0.0
    exact_sql: float = 0.0


@dataclass(frozen=True)
class BirdSQLBundle:
    example: BirdExample
    db_context: DatabaseContext
    gold_rows: list[tuple[Any, ...]]
    gold_error: str | None


def _renderer_supports_tool_calls(renderer_name: str) -> bool:
    lowered = renderer_name.lower()
    return any(token in lowered for token in ("qwen", "gpt_oss", "gpt-oss", "deepseek", "kimi"))


def _tool_error_json(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _safe_json_dict(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)
                elif item.get("type") == "thinking":
                    thinking_value = item.get("thinking")
                    if isinstance(thinking_value, str):
                        chunks.append(thinking_value)
        return "\n".join(chunk for chunk in chunks if chunk)
    if content is None:
        return ""
    return str(content)


def _extract_latest_assistant_text(history: list[Message]) -> str:
    for message in reversed(history):
        if message.get("role") != "assistant":
            continue
        return _content_to_text(message.get("content"))
    return ""


def _tool_message_has_error(content: Any) -> bool:
    if not isinstance(content, str):
        return True
    payload = _safe_json_dict(content)
    if payload is None:
        return False
    if "ok" in payload:
        return not bool(payload.get("ok"))
    return "error" in payload


def _count_query_tool_calls(history: list[Message]) -> int:
    count = 0
    for message in history:
        if message.get("role") != "tool":
            continue
        if message.get("name") == QUERY_TOOL_NAME:
            count += 1
    return count


def _count_tool_rounds(history: list[Message]) -> int:
    count = 0
    for message in history:
        if message.get("role") != "assistant":
            continue
        tool_calls = list(message.get("tool_calls") or [])
        unparsed_calls = list(message.get("unparsed_tool_calls") or [])
        if tool_calls or unparsed_calls:
            count += 1
    return count


def _count_tool_errors(history: list[Message]) -> int:
    errors = 0
    for message in history:
        if message.get("role") != "tool":
            continue
        if _tool_message_has_error(message.get("content")):
            errors += 1
    return errors


def _accumulate_numeric_metrics(target: dict[str, float], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if not isinstance(value, (int, float)):
            continue
        target[key] = target.get(key, 0.0) + float(value)


class BirdQueryTool:
    def __init__(self, db_path: Path, config: QueryToolConfig) -> None:
        self._config = config.validated()
        self._executor = QueryToolExecutor(db_path=db_path, config=self._config)
        self._call_count = 0
        function_schema = build_query_tool_schema().get("function", {})
        self._description = str(function_schema.get("description", ""))
        parameters = function_schema.get("parameters", {})
        self._parameters_schema = parameters if isinstance(parameters, dict) else {}

    @property
    def name(self) -> str:
        return QUERY_TOOL_NAME

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._parameters_schema

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        sql = parse_query_tool_arguments(input.arguments)
        if self._call_count >= self._config.max_calls:
            return simple_tool_result(
                _tool_error_json(
                    f"Maximum query tool calls reached ({self._config.max_calls}). Return final SQL now."
                ),
                call_id=input.call_id or "",
                name=self.name,
                metrics={
                    "query_tool_calls_this_step": 0.0,
                    "tool_call_errors_in_turn": 1.0,
                },
            )

        self._call_count += 1
        output_text = self._executor.execute(sql)
        payload = _safe_json_dict(output_text)
        tool_error = payload is None or not bool(payload.get("ok"))

        metrics = {
            "query_tool_calls_this_step": 1.0,
            "tool_call_errors_in_turn": float(tool_error),
        }
        if payload is not None and bool(payload.get("rows_truncated")):
            metrics["query_tool_rows_truncated"] = 1.0

        return simple_tool_result(
            output_text,
            call_id=input.call_id or "",
            name=self.name,
            metrics=metrics,
        )


@dataclass(frozen=True)
class BirdSQLReward:
    bundle: BirdSQLBundle
    reward_config: RewardConfig
    query_timeout_seconds: float
    ordered_result_compare: bool
    float_precision: int
    query_tool_enabled: bool

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        predicted_sql = extract_sql(_extract_latest_assistant_text(history)).strip()

        if predicted_sql:
            prediction_exec = execute_sql(
                db_path=self.bundle.db_context.db_path,
                sql=predicted_sql,
                timeout_seconds=self.query_timeout_seconds,
            )
        else:
            prediction_exec = execute_sql(
                db_path=self.bundle.db_context.db_path,
                sql="SELECT 1 WHERE 0",
                timeout_seconds=self.query_timeout_seconds,
            )
            prediction_exec.error = "Empty SQL output"

        executable = prediction_exec.error is None
        exact_match = bool(predicted_sql) and normalize_sql(predicted_sql) == normalize_sql(
            self.bundle.example.gold_sql
        )

        exec_match = False
        if prediction_exec.error is None and self.bundle.gold_error is None:
            exec_match = results_match(
                pred_rows=prediction_exec.rows,
                gold_rows=self.bundle.gold_rows,
                ordered=self.ordered_result_compare,
                float_precision=self.float_precision,
            )

        reward = (
            self.reward_config.exec_match * float(exec_match)
            + self.reward_config.executable * float(executable)
            + self.reward_config.exact_sql * float(exact_match)
        )

        metrics: dict[str, float] = {
            "exec_match": float(exec_match),
            "executable": float(executable),
            "exact_match": float(exact_match),
            "parse_success": 1.0,
            "prediction_has_sql": float(bool(predicted_sql)),
            "prediction_exec_error": float(prediction_exec.error is not None),
            "gold_exec_error": float(self.bundle.gold_error is not None),
            "query_tool_calls": float(_count_query_tool_calls(history)),
            "tool_rounds": float(_count_tool_rounds(history)),
            "tool_call_errors_total": float(_count_tool_errors(history)),
            "query_tool_enabled": float(self.query_tool_enabled),
        }
        return reward, metrics


@dataclass
class BirdSQLMessageEnv(AgentToolMessageEnv):
    _query_tool_calls: int = 0
    _tool_rounds: int = 0
    _last_tool_error_count: int = field(default=0, init=False)
    _last_tool_metrics: dict[str, float] = field(default_factory=dict, init=False)

    async def initial_observation(self) -> list[Message]:
        self.history = list(self.initial_messages)
        self._turn_count = 0
        self._should_stop = False
        self._query_tool_calls = 0
        self._tool_rounds = 0
        self._last_tool_error_count = 0
        self._last_tool_metrics = {}
        return self.history

    def _append_unparsed_tool_feedback(self, unparsed_tool_calls: list[Any]) -> int:
        if not unparsed_tool_calls:
            return 0

        for unparsed in unparsed_tool_calls:
            detail = getattr(unparsed, "error", "Malformed tool call")
            self.history.append(
                {
                    "role": "tool",
                    "name": QUERY_TOOL_NAME,
                    "content": _tool_error_json(
                        "Malformed tool call. Use JSON with name/arguments fields. "
                        f"Details: {detail}"
                    ),
                }
            )
        return len(unparsed_tool_calls)

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        self._last_tool_error_count = 0
        self._last_tool_metrics = {}
        if not tool_calls:
            return []

        tool_results = await asyncio.gather(
            *[handle_tool_call(self._tool_dict, tool_call) for tool_call in tool_calls]
        )
        all_messages: list[Message] = []

        for tool_result in tool_results:
            for msg in tool_result.messages:
                self.history.append(msg)
                all_messages.append(msg)
            if tool_result.should_stop:
                self._should_stop = True

            _accumulate_numeric_metrics(self._last_tool_metrics, tool_result.metrics)

            has_tool_error = bool(tool_result.metadata.get("error"))
            if not has_tool_error:
                for msg in tool_result.messages:
                    if _tool_message_has_error(msg.get("content")):
                        has_tool_error = True
                        break
            if has_tool_error:
                self._last_tool_error_count += 1

        self._query_tool_calls += len(tool_calls)
        return all_messages

    async def step(self, message: Message) -> MessageStepResult:
        self._turn_count += 1
        self.history.append(message)

        tool_calls = list(message.get("tool_calls") or [])
        unparsed_tool_calls = list(message.get("unparsed_tool_calls") or [])

        if tool_calls or unparsed_tool_calls:
            self._tool_rounds += 1

        step_metrics: dict[str, float] = {
            "parse_success": 1.0,
            "tool_call_turn": float(bool(tool_calls or unparsed_tool_calls)),
            "tool_call_count_in_turn": float(len(tool_calls)),
            "unparsed_tool_calls_in_turn": float(len(unparsed_tool_calls)),
        }

        tool_error_count = self._append_unparsed_tool_feedback(unparsed_tool_calls)
        await self._handle_tool_calls(tool_calls)
        tool_error_count += self._last_tool_error_count
        _accumulate_numeric_metrics(step_metrics, self._last_tool_metrics)

        if tool_error_count > 0:
            step_metrics["tool_call_errors_in_turn"] = float(tool_error_count)

        no_tool_activity = len(tool_calls) == 0 and len(unparsed_tool_calls) == 0
        max_turns_reached = self._turn_count >= self.max_turns
        done = no_tool_activity or max_turns_reached or self._should_stop

        if max_turns_reached and not no_tool_activity:
            step_metrics["max_turns_reached"] = 1.0
        if self._should_stop:
            step_metrics["tool_requested_stop"] = 1.0

        if not done:
            return MessageStepResult(
                reward=0.0,
                episode_done=False,
                next_messages=self.history,
                metrics=step_metrics,
            )

        reward, reward_metrics = await self.reward_fn(self.history)
        merged_metrics = dict(step_metrics)
        merged_metrics.update(reward_metrics)
        merged_metrics.setdefault("query_tool_calls", float(self._query_tool_calls))
        merged_metrics.setdefault("tool_rounds", float(self._tool_rounds))
        merged_metrics.setdefault("turns_taken", float(self._turn_count))

        return MessageStepResult(
            reward=reward,
            episode_done=True,
            next_messages=self.history,
            metrics=merged_metrics,
        )


def _build_initial_messages(
    *,
    bundle: BirdSQLBundle,
    include_evidence: bool,
    query_tool_obj: BirdQueryTool | None,
) -> list[Message]:
    query_tool_enabled = query_tool_obj is not None
    system_prompt = build_system_prompt(enable_query_tool=query_tool_enabled)
    if query_tool_enabled:
        system_prompt = f"{system_prompt}\n\n{_TOOL_SUPPORT_HINT}"

    if query_tool_enabled:
        tool_schema = build_query_tool_schema().get("function", {})
        tool_schema_text = json.dumps(tool_schema, ensure_ascii=False, indent=2)
        system_prompt = (
            f"{system_prompt}\n\n"
            "Tool schema:\n"
            f"```json\n{tool_schema_text}\n```\n\n"
            "When calling the tool, emit exactly:\n"
            f'<tool_call>{{"name":"{QUERY_TOOL_NAME}","args":{{"sql":"SELECT ..."}}}}</tool_call>'
        )
    prefix = [{"role": "system", "content": system_prompt}]

    user_prompt = build_user_prompt(
        example=bundle.example,
        schema_text=bundle.db_context.schema_text,
        include_evidence=include_evidence,
        data_dictionary_text=bundle.db_context.data_dictionary_text,
        enable_query_tool=query_tool_enabled,
    )

    return prefix + [{"role": "user", "content": user_prompt}]


@dataclass(frozen=True)
class BirdSQLEnvGroupBuilder(EnvGroupBuilder):
    bundle: BirdSQLBundle
    renderer: renderers.Renderer
    include_evidence: bool
    reward_config: RewardConfig
    query_timeout_seconds: float
    ordered_result_compare: bool
    float_precision: int
    query_tool_enabled: bool
    query_tool_config: QueryToolConfig
    max_trajectory_tokens: int
    max_turns: int
    failed_parse_reward: float
    terminate_on_parse_error: bool
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        envs: list[Env] = []
        for _ in range(self.group_size):
            query_tool_obj = (
                BirdQueryTool(db_path=self.bundle.db_context.db_path, config=self.query_tool_config)
                if self.query_tool_enabled
                else None
            )
            tools: list[Tool] = [query_tool_obj] if query_tool_obj is not None else []

            reward_fn = BirdSQLReward(
                bundle=self.bundle,
                reward_config=self.reward_config,
                query_timeout_seconds=self.query_timeout_seconds,
                ordered_result_compare=self.ordered_result_compare,
                float_precision=self.float_precision,
                query_tool_enabled=self.query_tool_enabled,
            )
            message_env = BirdSQLMessageEnv(
                tools=tools,
                initial_messages=_build_initial_messages(
                    bundle=self.bundle,
                    include_evidence=self.include_evidence,
                    query_tool_obj=query_tool_obj,
                ),
                max_turns=self.max_turns,
                reward_fn=reward_fn,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=self.renderer,
                    message_env=message_env,
                    failed_parse_reward=self.failed_parse_reward,
                    terminate_on_parse_error=self.terminate_on_parse_error,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
            )

        return envs

    def logging_tags(self) -> list[str]:
        return ["bird", self.bundle.example.db_id, self.bundle.example.difficulty]


class BirdSQLRLDataset(RLDataset):
    def __init__(
        self,
        *,
        bundles: list[BirdSQLBundle],
        renderer: renderers.Renderer,
        include_evidence: bool,
        reward_config: RewardConfig,
        query_timeout_seconds: float,
        ordered_result_compare: bool,
        float_precision: int,
        query_tool_enabled: bool,
        query_tool_config: QueryToolConfig,
        max_trajectory_tokens: int,
        max_turns: int,
        failed_parse_reward: float,
        terminate_on_parse_error: bool,
        group_size: int,
        groups_per_batch: int,
    ):
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if groups_per_batch <= 0:
            raise ValueError("groups_per_batch must be positive")

        self._bundles = bundles
        self._renderer = renderer
        self._include_evidence = include_evidence
        self._reward_config = reward_config
        self._query_timeout_seconds = query_timeout_seconds
        self._ordered_result_compare = ordered_result_compare
        self._float_precision = float_precision
        self._query_tool_enabled = query_tool_enabled
        self._query_tool_config = query_tool_config.validated()
        self._max_trajectory_tokens = max_trajectory_tokens
        self._max_turns = max_turns
        self._failed_parse_reward = failed_parse_reward
        self._terminate_on_parse_error = terminate_on_parse_error
        self._group_size = group_size
        self._groups_per_batch = groups_per_batch

    def __len__(self) -> int:
        return math.ceil(len(self._bundles) / self._groups_per_batch)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if index < 0 or index >= len(self):
            raise IndexError(f"batch index out of range: {index}")

        start = index * self._groups_per_batch
        end = min(len(self._bundles), start + self._groups_per_batch)

        return [
            BirdSQLEnvGroupBuilder(
                bundle=bundle,
                renderer=self._renderer,
                include_evidence=self._include_evidence,
                reward_config=self._reward_config,
                query_timeout_seconds=self._query_timeout_seconds,
                ordered_result_compare=self._ordered_result_compare,
                float_precision=self._float_precision,
                query_tool_enabled=self._query_tool_enabled,
                query_tool_config=self._query_tool_config,
                max_trajectory_tokens=self._max_trajectory_tokens,
                max_turns=self._max_turns,
                failed_parse_reward=self._failed_parse_reward,
                terminate_on_parse_error=self._terminate_on_parse_error,
                group_size=self._group_size,
            )
            for bundle in self._bundles[start:end]
        ]


@chz.chz
class BirdSQLDatasetBuilder(RLDatasetBuilder):
    dataset_root: str
    split_file: str = "dev.json"
    db_id: str | None = None

    model_name_for_tokenizer: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3"

    train_offset: int = 0
    train_limit: int = 256
    eval_offset: int | None = None
    eval_limit: int = 64
    shuffle: bool = True
    seed: int = 0

    include_evidence: bool = False
    data_dictionary_mode: str = DATA_DICTIONARY_MODE_STATS_AND_SAMPLES
    data_dictionary_max_values: int = 3
    schema_sample_rows: int = 0
    max_columns_per_table: int = 80
    query_timeout_seconds: float = 20.0
    ordered_result_compare: bool = False
    float_precision: int = 6
    query_tool_enabled: bool = True
    query_tool_max_calls: int = 8
    query_tool_max_rows: int = 50
    query_tool_max_output_chars: int = 6000
    query_tool_max_cell_chars: int = 200
    query_tool_timeout_seconds: float = 8.0

    max_turns: int | None = None
    max_trajectory_tokens: int = 32 * 1024
    failed_parse_reward: float = -0.1
    terminate_on_parse_error: bool = True

    reward_exec_match: float = 1.0
    reward_executable: float = 0.0
    reward_exact_sql: float = 0.0

    group_size: int = 8
    groups_per_batch: int = 16

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        dataset_root = Path(self.dataset_root)
        examples = load_examples(dataset_root=dataset_root, split_file=self.split_file)
        examples = filter_examples(examples, db_id=self.db_id)

        if self.shuffle:
            random.Random(self.seed).shuffle(examples)

        train_examples = _slice_examples(
            examples=examples,
            offset=self.train_offset,
            limit=self.train_limit,
        )
        if not train_examples:
            raise ValueError("No training examples selected for GRPO. Check train offset/limit.")

        resolved_eval_offset = self.eval_offset
        if resolved_eval_offset is None:
            resolved_eval_offset = self.train_offset + len(train_examples)

        eval_examples = _slice_examples(
            examples=examples,
            offset=resolved_eval_offset,
            limit=self.eval_limit,
        )

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        query_tool_enabled = self.query_tool_enabled
        if query_tool_enabled and not _renderer_supports_tool_calls(self.renderer_name):
            logger.warning(
                "Renderer '%s' does not support cookbook tool-call parsing; disabling query tool for RL.",
                self.renderer_name,
            )
            query_tool_enabled = False

        query_tool_config = QueryToolConfig(
            max_calls=self.query_tool_max_calls,
            max_rows=self.query_tool_max_rows,
            max_output_chars=self.query_tool_max_output_chars,
            max_cell_chars=self.query_tool_max_cell_chars,
            timeout_seconds=self.query_tool_timeout_seconds,
        ).validated()

        reward_config = RewardConfig(
            exec_match=self.reward_exec_match,
            executable=self.reward_executable,
            exact_sql=self.reward_exact_sql,
        )

        resolved_max_turns = self.max_turns
        if resolved_max_turns is None:
            resolved_max_turns = query_tool_config.max_calls + 2 if query_tool_enabled else 1
        resolved_max_turns = max(1, resolved_max_turns)

        train_bundles = _build_bundles(
            examples=train_examples,
            dataset_root=dataset_root,
            sample_rows=self.schema_sample_rows,
            max_columns_per_table=self.max_columns_per_table,
            data_dictionary_mode=self.data_dictionary_mode,
            data_dictionary_max_values=self.data_dictionary_max_values,
            query_timeout_seconds=self.query_timeout_seconds,
        )

        train_dataset = BirdSQLRLDataset(
            bundles=train_bundles,
            renderer=renderer,
            include_evidence=self.include_evidence,
            reward_config=reward_config,
            query_timeout_seconds=self.query_timeout_seconds,
            ordered_result_compare=self.ordered_result_compare,
            float_precision=self.float_precision,
            query_tool_enabled=query_tool_enabled,
            query_tool_config=query_tool_config,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_turns=resolved_max_turns,
            failed_parse_reward=self.failed_parse_reward,
            terminate_on_parse_error=self.terminate_on_parse_error,
            group_size=self.group_size,
            groups_per_batch=self.groups_per_batch,
        )

        eval_dataset: BirdSQLRLDataset | None = None
        if eval_examples:
            eval_bundles = _build_bundles(
                examples=eval_examples,
                dataset_root=dataset_root,
                sample_rows=self.schema_sample_rows,
                max_columns_per_table=self.max_columns_per_table,
                data_dictionary_mode=self.data_dictionary_mode,
                data_dictionary_max_values=self.data_dictionary_max_values,
                query_timeout_seconds=self.query_timeout_seconds,
            )
            eval_dataset = BirdSQLRLDataset(
                bundles=eval_bundles,
                renderer=renderer,
                include_evidence=self.include_evidence,
                reward_config=reward_config,
                query_timeout_seconds=self.query_timeout_seconds,
                ordered_result_compare=self.ordered_result_compare,
                float_precision=self.float_precision,
                query_tool_enabled=query_tool_enabled,
                query_tool_config=query_tool_config,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_turns=resolved_max_turns,
                failed_parse_reward=self.failed_parse_reward,
                terminate_on_parse_error=self.terminate_on_parse_error,
                group_size=1,
                groups_per_batch=min(self.groups_per_batch, len(eval_bundles)),
            )

        return train_dataset, eval_dataset


@dataclass(frozen=True)
class TinkerGRPOConfig:
    dataset_root: Path
    split_file: str = "dev.json"
    model: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = None
    db_id: str | None = None

    train_offset: int = 0
    train_limit: int = 256
    eval_offset: int | None = None
    eval_limit: int = 64
    shuffle: bool = True
    seed: int = 0

    include_evidence: bool = False
    data_dictionary_mode: str = DATA_DICTIONARY_MODE_STATS_AND_SAMPLES
    data_dictionary_max_values: int = 3
    schema_sample_rows: int = 0
    max_columns_per_table: int = 80
    query_timeout_seconds: float = 20.0
    ordered_result_compare: bool = False
    float_precision: int = 6
    query_tool_enabled: bool = True
    query_tool_max_calls: int = 8
    query_tool_max_rows: int = 50
    query_tool_max_output_chars: int = 6000
    query_tool_max_cell_chars: int = 200
    query_tool_timeout_seconds: float = 8.0
    max_trajectory_tokens: int = 32 * 1024

    group_size: int = 8
    groups_per_batch: int = 16
    learning_rate: float = 1e-5
    max_tokens: int = 4096
    temperature: float = 1.0
    lora_rank: int = 32
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    max_steps_off_policy: int | None = None
    stream_minibatch: bool = False
    num_minibatches: int = 4

    reward_exec_match: float = 1.0
    reward_executable: float = 0.0
    reward_exact_sql: float = 0.0

    eval_every: int = 10
    save_every: int = 10
    log_path: Path | None = None
    load_checkpoint_path: str | None = None
    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    num_groups_to_log: int = 4

    max_turns: int | None = None
    failed_parse_reward: float = -0.1
    terminate_on_parse_error: bool = True


class _BirdAliasLogger(ml_log.Logger):
    def __init__(self, inner: ml_log.Logger):
        self._inner = inner

    def log_hparams(self, config: Any) -> None:
        self._inner.log_hparams(config)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        merged = dict(metrics)
        derived = _derive_tracking_metrics(merged)
        for key, value in derived.items():
            merged.setdefault(key, value)
        self._inner.log_metrics(merged, step=step)

    def log_long_text(self, key: str, text: str) -> None:
        self._inner.log_long_text(key, text)

    def close(self) -> None:
        self._inner.close()

    def sync(self) -> None:
        self._inner.sync()

    def get_logger_url(self) -> str | None:
        return self._inner.get_logger_url()


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _copy_metric(
    source: dict[str, Any],
    destination: dict[str, float],
    source_key: str,
    destination_key: str,
) -> None:
    value = _as_float(source.get(source_key))
    if value is not None:
        destination[destination_key] = value


def _first_numeric_loss_metric(metrics: dict[str, Any]) -> float | None:
    preferred = (
        "train/loss",
        "optim/loss",
        "loss",
        "optim/fwd_loss",
        "optim/fwd_train_loss",
    )

    for key in preferred:
        value = _as_float(metrics.get(key))
        if value is not None:
            return value

    for key, value in metrics.items():
        if "loss" not in key.lower():
            continue
        float_value = _as_float(value)
        if float_value is not None:
            return float_value

    return None


def _derive_tracking_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    derived: dict[str, float] = {}

    _copy_metric(metrics, derived, "env/all/reward/total", "train/reward_total")
    _copy_metric(metrics, derived, "env/all/exec_match", "train/exec_match")
    _copy_metric(metrics, derived, "env/all/executable", "train/executable")
    _copy_metric(metrics, derived, "env/all/exact_match", "train/exact_match")
    _copy_metric(metrics, derived, "env/all/parse_success", "train/parse_success")
    _copy_metric(metrics, derived, "env/all/query_tool_calls", "train/query_tool_calls")

    _copy_metric(metrics, derived, "test/env/all/reward/total", "eval/reward_total")
    _copy_metric(metrics, derived, "test/env/all/exec_match", "eval/exec_match")
    _copy_metric(metrics, derived, "test/env/all/executable", "eval/executable")
    _copy_metric(metrics, derived, "test/env/all/exact_match", "eval/exact_match")
    _copy_metric(metrics, derived, "test/env/all/parse_success", "eval/parse_success")
    _copy_metric(metrics, derived, "test/env/all/query_tool_calls", "eval/query_tool_calls")

    _copy_metric(metrics, derived, "optim/kl_sample_train_v1", "train/kl_sample_train_v1")
    _copy_metric(metrics, derived, "optim/kl_sample_train_v2", "train/kl_sample_train_v2")
    _copy_metric(metrics, derived, "optim/entropy", "train/entropy")

    maybe_loss = _first_numeric_loss_metric(metrics)
    if maybe_loss is not None:
        derived["train/loss"] = maybe_loss

    return derived


def _average_numeric_dicts(dicts: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for metric_dict in dicts:
        for key, value in metric_dict.items():
            float_value = _as_float(value)
            if float_value is None:
                continue
            sums[key] = sums.get(key, 0.0) + float_value
            counts[key] = counts.get(key, 0) + 1

    return {key: sums[key] / counts[key] for key in sums if counts.get(key, 0) > 0}


_METRIC_PATCH_INSTALLED = False


def _install_training_metric_patches() -> None:
    global _METRIC_PATCH_INSTALLED

    if _METRIC_PATCH_INSTALLED:
        return

    original_setup_logging = rl_train.ml_log.setup_logging

    def _patched_setup_logging(*args: Any, **kwargs: Any) -> ml_log.Logger:
        base_logger = original_setup_logging(*args, **kwargs)
        return _BirdAliasLogger(base_logger)

    async def _patched_train_step(
        data_D: list[tinker.Datum],
        training_client: tinker.TrainingClient,
        learning_rate: float,
        num_substeps: int,
        loss_fn: LossFnType,
        loss_fn_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> list[Any]:
        batches = rl_train.split_list(data_D, min(num_substeps, len(data_D)))
        if not batches:
            return []

        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        training_logprobs_D: list[Any] = []
        optim_result: tinker.OptimStepResponse | None = None
        fwd_metric_snapshots: list[dict[str, Any]] = []

        fwd_bwd_future = await training_client.forward_backward_async(
            [rl_train._remove_mask(datum) for datum in batches[0]],
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        optim_future = await training_client.optim_step_async(adam_params)

        for i in range(len(batches)):
            if i + 1 < len(batches):
                next_fwd_bwd_future = await training_client.forward_backward_async(
                    [rl_train._remove_mask(datum) for datum in batches[i + 1]],
                    loss_fn=loss_fn,
                    loss_fn_config=loss_fn_config,
                )
                next_optim_future = await training_client.optim_step_async(adam_params)
            else:
                next_fwd_bwd_future = None
                next_optim_future = None

            fwd_bwd_result = await fwd_bwd_future.result_async()
            training_logprobs_D.extend(rl_train._training_logprobs_from_fwd_bwd(fwd_bwd_result))
            if fwd_bwd_result.metrics:
                fwd_metric_snapshots.append(dict(fwd_bwd_result.metrics))

            optim_result = await optim_future.result_async()

            if next_fwd_bwd_future is not None and next_optim_future is not None:
                fwd_bwd_future = next_fwd_bwd_future
                optim_future = next_optim_future

        if metrics is not None:
            if optim_result is not None and optim_result.metrics:
                metrics.update(optim_result.metrics)

            if fwd_metric_snapshots:
                averaged_fwd_metrics = _average_numeric_dicts(fwd_metric_snapshots)
                metrics.update({f"optim/fwd_{key}": value for key, value in averaged_fwd_metrics.items()})

                maybe_loss = _first_numeric_loss_metric(averaged_fwd_metrics)
                if maybe_loss is not None:
                    metrics.setdefault("train/loss", maybe_loss)

            maybe_loss = _first_numeric_loss_metric(metrics)
            if maybe_loss is not None:
                metrics.setdefault("train/loss", maybe_loss)

        return training_logprobs_D

    rl_train.ml_log.setup_logging = _patched_setup_logging
    rl_train.train_step = _patched_train_step

    _METRIC_PATCH_INSTALLED = True


def run_tinker_grpo_training(config: TinkerGRPOConfig) -> dict[str, Any]:
    if config.group_size < 2:
        raise ValueError("group_size must be >= 2 for GRPO-style grouped rollouts")

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model)
    query_tool_active = config.query_tool_enabled and _renderer_supports_tool_calls(renderer_name)
    if config.query_tool_enabled and not query_tool_active:
        logger.warning(
            "Renderer '%s' does not support cookbook tool-call parsing; disabling query tool for RL.",
            renderer_name,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = config.model.replace("/", "-")
    default_log_path = Path("outputs") / "tinker_grpo" / f"{timestamp}_{model_tag}"
    resolved_log_path = config.log_path or default_log_path
    default_wandb_name = f"bird-grpo-{model_tag}-{config.loss_fn}-{timestamp}"

    resolved_wandb_project, resolved_wandb_name = _resolve_wandb_settings(
        config=config,
        default_wandb_name=default_wandb_name,
        log_path=resolved_log_path,
    )

    cli_utils.check_log_dir(
        str(resolved_log_path),
        behavior_if_exists="ask",
    )

    _install_training_metric_patches()

    dataset_builder = BirdSQLDatasetBuilder(
        dataset_root=str(config.dataset_root),
        split_file=config.split_file,
        db_id=config.db_id,
        model_name_for_tokenizer=config.model,
        renderer_name=renderer_name,
        train_offset=config.train_offset,
        train_limit=config.train_limit,
        eval_offset=config.eval_offset,
        eval_limit=config.eval_limit,
        shuffle=config.shuffle,
        seed=config.seed,
        include_evidence=config.include_evidence,
        data_dictionary_mode=config.data_dictionary_mode,
        data_dictionary_max_values=config.data_dictionary_max_values,
        schema_sample_rows=config.schema_sample_rows,
        max_columns_per_table=config.max_columns_per_table,
        query_timeout_seconds=config.query_timeout_seconds,
        ordered_result_compare=config.ordered_result_compare,
        float_precision=config.float_precision,
        query_tool_enabled=query_tool_active,
        query_tool_max_calls=config.query_tool_max_calls,
        query_tool_max_rows=config.query_tool_max_rows,
        query_tool_max_output_chars=config.query_tool_max_output_chars,
        query_tool_max_cell_chars=config.query_tool_max_cell_chars,
        query_tool_timeout_seconds=config.query_tool_timeout_seconds,
        max_turns=config.max_turns,
        max_trajectory_tokens=config.max_trajectory_tokens,
        failed_parse_reward=config.failed_parse_reward,
        terminate_on_parse_error=config.terminate_on_parse_error,
        reward_exec_match=config.reward_exec_match,
        reward_executable=config.reward_executable,
        reward_exact_sql=config.reward_exact_sql,
        group_size=config.group_size,
        groups_per_batch=config.groups_per_batch,
    )

    async_config = None
    if config.max_steps_off_policy is not None:
        async_config = AsyncConfig(
            max_steps_off_policy=config.max_steps_off_policy,
            groups_per_batch=config.groups_per_batch,
        )

    if async_config is not None and config.stream_minibatch:
        raise ValueError("stream_minibatch cannot be enabled together with max_steps_off_policy.")

    stream_minibatch_config = None
    if config.stream_minibatch:
        if config.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive when stream_minibatch is enabled.")
        stream_minibatch_config = StreamMinibatchConfig(
            groups_per_batch=config.groups_per_batch,
            num_minibatches=config.num_minibatches,
        )

    train_config = RLTrainConfig(
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        compute_post_kl=config.compute_post_kl,
        lora_rank=config.lora_rank,
        kl_penalty_coef=config.kl_penalty_coef,
        kl_discount_factor=config.kl_discount_factor,
        num_substeps=config.num_substeps,
        loss_fn=config.loss_fn,
        loss_fn_config=config.loss_fn_config,
        log_path=str(resolved_log_path),
        load_checkpoint_path=config.load_checkpoint_path,
        base_url=config.base_url,
        wandb_project=resolved_wandb_project,
        wandb_name=resolved_wandb_name,
        eval_every=config.eval_every,
        save_every=config.save_every,
        async_config=async_config,
        stream_minibatch_config=stream_minibatch_config,
        num_groups_to_log=max(0, config.num_groups_to_log),
    )

    asyncio.run(rl_train.main(train_config))

    summary: dict[str, Any] = {
        "log_path": str(resolved_log_path),
        "model": config.model,
        "renderer": renderer_name,
        "loss_fn": config.loss_fn,
        "loss_fn_config": config.loss_fn_config,
        "group_size": config.group_size,
        "groups_per_batch": config.groups_per_batch,
        "stream_minibatch": config.stream_minibatch,
        "num_minibatches": config.num_minibatches if config.stream_minibatch else None,
        "compute_post_kl": config.compute_post_kl,
        "kl_discount_factor": config.kl_discount_factor,
        "num_groups_to_log": max(0, config.num_groups_to_log),
        "query_tool_requested": config.query_tool_enabled,
        "query_tool_active": query_tool_active,
        "wandb_project": resolved_wandb_project,
        "wandb_name": resolved_wandb_name,
    }

    checkpoints_path = resolved_log_path / "checkpoints.jsonl"
    last_checkpoint = _read_last_jsonl_record(checkpoints_path)
    if last_checkpoint is not None:
        summary["last_checkpoint"] = last_checkpoint

    metrics_path = resolved_log_path / "metrics.jsonl"
    last_metrics = _read_last_jsonl_record(metrics_path)
    if last_metrics is not None:
        for key in (
            "train/loss",
            "train/reward_total",
            "train/exec_match",
            "eval/reward_total",
            "eval/exec_match",
        ):
            if key in last_metrics:
                summary[key] = last_metrics[key]

    return summary


def _resolve_wandb_settings(
    *,
    config: TinkerGRPOConfig,
    default_wandb_name: str,
    log_path: Path,
) -> tuple[str | None, str | None]:
    wandb_enabled = bool(config.wandb_project)
    if not wandb_enabled:
        return None, None

    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError(
            "wandb is not installed. Install it first (`uv pip install wandb` "
            "or `uv sync --extra rl`)."
        )

    os.environ.setdefault("WANDB_DIR", str(log_path))

    wandb_mode = os.environ.get("WANDB_MODE", "online")
    has_api_key = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_mode == "offline" and not has_api_key:
        os.environ["WANDB_API_KEY"] = "offline-local-key"

    resolved_project = config.wandb_project or "bird-grpo"
    resolved_name = config.wandb_name or default_wandb_name
    return resolved_project, resolved_name


def _slice_examples(examples: list[BirdExample], offset: int, limit: int) -> list[BirdExample]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    sliced = examples[offset:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def _read_last_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    record = json.loads(lines[-1])
    if not isinstance(record, dict):
        raise ValueError(f"Expected JSON object in {path}, got: {type(record).__name__}")
    return record


def _build_bundles(
    *,
    examples: list[BirdExample],
    dataset_root: Path,
    sample_rows: int,
    max_columns_per_table: int,
    data_dictionary_mode: str,
    data_dictionary_max_values: int,
    query_timeout_seconds: float,
) -> list[BirdSQLBundle]:
    db_cache: dict[str, DatabaseContext] = {}
    bundles: list[BirdSQLBundle] = []

    for example in examples:
        if example.db_id not in db_cache:
            db_cache[example.db_id] = load_database_context(
                dataset_root=dataset_root,
                db_id=example.db_id,
                sample_rows_per_table=sample_rows,
                max_columns_per_table=max_columns_per_table,
                data_dictionary_mode=data_dictionary_mode,
                data_dictionary_max_values=data_dictionary_max_values,
            )

        db_context = db_cache[example.db_id]
        gold_exec = execute_sql(
            db_path=db_context.db_path,
            sql=example.gold_sql,
            timeout_seconds=query_timeout_seconds,
        )

        bundles.append(
            BirdSQLBundle(
                example=example,
                db_context=db_context,
                gold_rows=gold_exec.rows,
                gold_error=gold_exec.error,
            )
        )

    return bundles
