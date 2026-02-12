from __future__ import annotations

import asyncio
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import chz
import tinker
from tinker.types import LossFnType
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.train import AsyncConfig, Config as RLTrainConfig, main as rl_train_main
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from bird_scaffold.data_dictionary import DATA_DICTIONARY_MODE_OFF
from bird_scaffold.dataset import filter_examples, load_examples
from bird_scaffold.db import load_database_context
from bird_scaffold.execution import execute_sql, normalize_sql, results_match
from bird_scaffold.prompting import build_system_prompt, build_user_prompt
from bird_scaffold.sql_parsing import extract_sql
from bird_scaffold.types import BirdExample, DatabaseContext


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


class BirdSQLEnv(Env):
    def __init__(
        self,
        *,
        bundle: BirdSQLBundle,
        renderer: renderers.Renderer,
        include_evidence: bool,
        reward_config: RewardConfig,
        query_timeout_seconds: float,
        ordered_result_compare: bool,
        float_precision: int,
    ):
        self.bundle = bundle
        self.renderer = renderer
        self.include_evidence = include_evidence
        self.reward_config = reward_config
        self.query_timeout_seconds = query_timeout_seconds
        self.ordered_result_compare = ordered_result_compare
        self.float_precision = float_precision

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        system_prompt = build_system_prompt(enable_query_tool=False)
        user_prompt = build_user_prompt(
            example=self.bundle.example,
            schema_text=self.bundle.db_context.schema_text,
            include_evidence=self.include_evidence,
            data_dictionary_text=self.bundle.db_context.data_dictionary_text,
            enable_query_tool=False,
        )
        conversation: list[renderers.Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_input = self.renderer.build_generation_prompt(conversation)
        return model_input, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        predicted_sql = extract_sql(content).strip()

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

        metrics: Metrics = {
            "exec_match": float(exec_match),
            "executable": float(executable),
            "exact_match": float(exact_match),
            "parse_success": float(parse_success),
            "prediction_has_sql": float(bool(predicted_sql)),
            "prediction_exec_error": float(prediction_exec.error is not None),
            "gold_exec_error": float(self.bundle.gold_error is not None),
        }

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


@dataclass(frozen=True)
class BirdSQLEnvGroupBuilder(EnvGroupBuilder):
    bundle: BirdSQLBundle
    renderer: renderers.Renderer
    include_evidence: bool
    reward_config: RewardConfig
    query_timeout_seconds: float
    ordered_result_compare: bool
    float_precision: int
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            BirdSQLEnv(
                bundle=self.bundle,
                renderer=self.renderer,
                include_evidence=self.include_evidence,
                reward_config=self.reward_config,
                query_timeout_seconds=self.query_timeout_seconds,
                ordered_result_compare=self.ordered_result_compare,
                float_precision=self.float_precision,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

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

    include_evidence: bool = True
    data_dictionary_mode: str = DATA_DICTIONARY_MODE_OFF
    data_dictionary_max_values: int = 3
    schema_sample_rows: int = 0
    max_columns_per_table: int = 80
    query_timeout_seconds: float = 20.0
    ordered_result_compare: bool = False
    float_precision: int = 6

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

        reward_config = RewardConfig(
            exec_match=self.reward_exec_match,
            executable=self.reward_executable,
            exact_sql=self.reward_exact_sql,
        )

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
                group_size=1,
                groups_per_batch=min(self.groups_per_batch, len(eval_bundles)),
            )

        return train_dataset, eval_dataset


@dataclass(frozen=True)
class TinkerGRPOConfig:
    dataset_root: Path
    split_file: str
    model: str
    renderer_name: str | None
    db_id: str | None

    train_offset: int
    train_limit: int
    eval_offset: int | None
    eval_limit: int
    shuffle: bool
    seed: int

    include_evidence: bool
    data_dictionary_mode: str
    data_dictionary_max_values: int
    schema_sample_rows: int
    max_columns_per_table: int
    query_timeout_seconds: float
    ordered_result_compare: bool
    float_precision: int

    group_size: int
    groups_per_batch: int
    learning_rate: float
    max_tokens: int
    temperature: float
    lora_rank: int
    kl_penalty_coef: float
    num_substeps: int
    loss_fn: LossFnType
    max_steps_off_policy: int | None

    reward_exec_match: float
    reward_executable: float
    reward_exact_sql: float

    eval_every: int
    save_every: int
    log_path: Path | None
    load_checkpoint_path: str | None
    base_url: str | None
    wandb_project: str | None
    wandb_name: str | None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior


def run_tinker_grpo_training(config: TinkerGRPOConfig) -> dict[str, Any]:
    if config.group_size < 2:
        raise ValueError("group_size must be >= 2 for GRPO-style grouped rollouts")

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = config.model.replace("/", "-")
    default_log_path = Path("outputs") / "tinker_grpo" / f"{timestamp}_{model_tag}"
    resolved_log_path = config.log_path or default_log_path

    cli_utils.check_log_dir(
        str(resolved_log_path),
        behavior_if_exists=config.behavior_if_log_dir_exists,
    )

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

    train_config = RLTrainConfig(
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        lora_rank=config.lora_rank,
        kl_penalty_coef=config.kl_penalty_coef,
        num_substeps=config.num_substeps,
        loss_fn=config.loss_fn,
        log_path=str(resolved_log_path),
        load_checkpoint_path=config.load_checkpoint_path,
        base_url=config.base_url,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        eval_every=config.eval_every,
        save_every=config.save_every,
        async_config=async_config,
    )

    asyncio.run(rl_train_main(train_config))

    summary: dict[str, Any] = {
        "log_path": str(resolved_log_path),
        "model": config.model,
        "renderer": renderer_name,
        "group_size": config.group_size,
        "groups_per_batch": config.groups_per_batch,
    }

    checkpoints_path = resolved_log_path / "checkpoints.jsonl"
    if checkpoints_path.exists():
        lines = checkpoints_path.read_text(encoding="utf-8").splitlines()
        if lines:
            try:
                last_checkpoint = json.loads(lines[-1])
                summary["last_checkpoint"] = last_checkpoint
            except json.JSONDecodeError:
                pass

    return summary


def _slice_examples(examples: list[BirdExample], offset: int, limit: int) -> list[BirdExample]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    sliced = examples[offset:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


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
