from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import chz
from tinker.types import LossFnType
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.train import (
    AsyncConfig,
    Config as RLTrainConfig,
    KLReferenceConfig,
    StreamMinibatchConfig,
)
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import AgentToolMessageEnv, Tool, ToolInput, ToolResult, simple_tool_result

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



def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                chunks.append(item["text"])
                continue
            if item.get("type") == "thinking" and isinstance(item.get("thinking"), str):
                chunks.append(item["thinking"])
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



def _count_query_tool_calls(history: list[Message]) -> int:
    return sum(
        1
        for message in history
        if message.get("role") == "tool" and message.get("name") == QUERY_TOOL_NAME
    )


class BirdQueryTool:
    """Lightweight wrapper that adapts QueryToolExecutor to cookbook Tool protocol."""

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
            )

        self._call_count += 1
        output_text = self._executor.execute(sql)
        return simple_tool_result(output_text, call_id=input.call_id or "", name=self.name)


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
            executable = prediction_exec.error is None
        else:
            prediction_exec = None
            executable = False

        exact_match = bool(predicted_sql) and normalize_sql(predicted_sql) == normalize_sql(
            self.bundle.example.gold_sql
        )

        exec_match = False
        if prediction_exec is not None and prediction_exec.error is None and self.bundle.gold_error is None:
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

        metrics = {
            "exec_match": float(exec_match),
            "executable": float(executable),
            "exact_match": float(exact_match),
            "prediction_has_sql": float(bool(predicted_sql)),
            "query_tool_calls": float(_count_query_tool_calls(history)),
            "query_tool_enabled": float(self.query_tool_enabled),
        }
        return reward, metrics



def _build_initial_messages(
    *,
    bundle: BirdSQLBundle,
    include_evidence: bool,
    query_tool_enabled: bool,
) -> list[Message]:
    system_prompt = build_system_prompt(enable_query_tool=query_tool_enabled)

    if query_tool_enabled:
        tool_schema = build_query_tool_schema().get("function", {})
        compact_schema = json.dumps(tool_schema, ensure_ascii=False, separators=(",", ":"))
        system_prompt = (
            f"{system_prompt}\n\n{_TOOL_SUPPORT_HINT}\n"
            f"Tool schema: {compact_schema}"
        )

    user_prompt = build_user_prompt(
        example=bundle.example,
        schema_text=bundle.db_context.schema_text,
        include_evidence=include_evidence,
        data_dictionary_text=bundle.db_context.data_dictionary_text,
        enable_query_tool=query_tool_enabled,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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
        initial_messages = _build_initial_messages(
            bundle=self.bundle,
            include_evidence=self.include_evidence,
            query_tool_enabled=self.query_tool_enabled,
        )
        reward_fn = BirdSQLReward(
            bundle=self.bundle,
            reward_config=self.reward_config,
            query_timeout_seconds=self.query_timeout_seconds,
            ordered_result_compare=self.ordered_result_compare,
            float_precision=self.float_precision,
            query_tool_enabled=self.query_tool_enabled,
        )

        envs: list[Env] = []
        for _ in range(self.group_size):
            query_tool_obj = (
                BirdQueryTool(db_path=self.bundle.db_context.db_path, config=self.query_tool_config)
                if self.query_tool_enabled
                else None
            )
            tools: list[Tool] = [query_tool_obj] if query_tool_obj is not None else []
            msg_env = AgentToolMessageEnv(
                tools=tools,
                initial_messages=initial_messages,
                max_turns=self.max_turns,
                reward_fn=reward_fn,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=self.renderer,
                    message_env=msg_env,
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
    ) -> None:
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

        train_examples = _slice_examples(examples, offset=self.train_offset, limit=self.train_limit)
        if not train_examples:
            raise ValueError("No training examples selected for GRPO. Check train offset/limit.")

        eval_offset = self.eval_offset
        if eval_offset is None:
            eval_offset = self.train_offset + len(train_examples)
        eval_examples = _slice_examples(examples, offset=eval_offset, limit=self.eval_limit)

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

        max_turns = self.max_turns
        if max_turns is None:
            max_turns = query_tool_config.max_calls + 2 if query_tool_enabled else 1
        max_turns = max(1, max_turns)

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
            max_turns=max_turns,
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
                max_turns=max_turns,
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
    kl_reference_base_model: str | None = None
    kl_reference_checkpoint_path: str | None = None
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



def _resolve_wandb_settings(
    *,
    config: TinkerGRPOConfig,
    default_wandb_name: str,
    log_path: Path,
) -> tuple[str | None, str | None]:
    os.environ.setdefault("WANDB_DIR", str(log_path))

    wandb_mode = os.environ.get("WANDB_MODE", "online")
    has_api_key = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_mode == "offline" and not has_api_key:
        os.environ["WANDB_API_KEY"] = "offline-local-key"

    resolved_project = config.wandb_project or "bird-grpo"
    resolved_name = config.wandb_name or default_wandb_name
    return resolved_project, resolved_name



def _resolve_kl_reference_config(config: TinkerGRPOConfig) -> KLReferenceConfig | None:
    if config.kl_penalty_coef <= 0:
        return None

    base_model = config.kl_reference_base_model or config.model
    if not base_model:
        raise ValueError("kl_reference_base_model (or model) must be set when kl_penalty_coef > 0")

    return KLReferenceConfig(
        base_model=base_model,
        load_checkpoint_path=config.kl_reference_checkpoint_path,
    )



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

    default_wandb_name = f"bird-grpo-{model_tag}-{timestamp}"
    wandb_project, wandb_name = _resolve_wandb_settings(
        config=config,
        default_wandb_name=default_wandb_name,
        log_path=resolved_log_path,
    )

    cli_utils.check_log_dir(str(resolved_log_path), behavior_if_exists="ask")

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
        raise ValueError("stream_minibatch cannot be enabled together with max_steps_off_policy")

    stream_minibatch_config = None
    if config.stream_minibatch:
        if config.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive when stream_minibatch is enabled")
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
        kl_reference_config=_resolve_kl_reference_config(config),
        num_substeps=config.num_substeps,
        loss_fn=config.loss_fn,
        loss_fn_config=config.loss_fn_config,
        log_path=str(resolved_log_path),
        load_checkpoint_path=config.load_checkpoint_path,
        base_url=config.base_url,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
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
        "group_size": config.group_size,
        "groups_per_batch": config.groups_per_batch,
        "query_tool_requested": config.query_tool_enabled,
        "query_tool_active": query_tool_active,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
    }

    if config.kl_penalty_coef > 0:
        summary["kl_reference_base_model"] = config.kl_reference_base_model or config.model
        summary["kl_reference_checkpoint_path"] = config.kl_reference_checkpoint_path

    checkpoints_path = resolved_log_path / "checkpoints.jsonl"
    last_checkpoint = _read_last_jsonl_record(checkpoints_path)
    if last_checkpoint is not None:
        summary["last_checkpoint"] = last_checkpoint

    metrics_path = resolved_log_path / "metrics.jsonl"
    last_metrics = _read_last_jsonl_record(metrics_path)
    if last_metrics is not None:
        maybe_train_loss = _first_present_numeric(
            last_metrics,
            ["train/loss", "optim/loss", "loss", "optim/fwd_loss", "optim/fwd_train_loss"],
        )
        if maybe_train_loss is not None:
            summary["train/loss"] = maybe_train_loss

        metric_aliases = {
            "train/reward_total": ["env/all/reward/total"],
            "train/exec_match": ["env/all/exec_match"],
            "eval/reward_total": ["test/env/all/reward/total"],
            "eval/exec_match": ["test/env/all/exec_match"],
        }
        for summary_key, candidates in metric_aliases.items():
            value = _first_present_numeric(last_metrics, candidates)
            if value is not None:
                summary[summary_key] = value

    return summary



def _first_present_numeric(metrics: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None



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
