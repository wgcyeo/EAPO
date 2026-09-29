"""Shared training defaults for EAPO and the GRPO baseline."""

from __future__ import annotations

import argparse
import os
from typing import Any

DATASET_NAME = "open-r1/DAPO-Math-17k-Processed"
DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_DATA_CACHE_DIR = "data"

COMMON_TRAINING_DEFAULTS: dict[str, Any] = {
    "dataset_name": DATASET_NAME,
    "dataset_config": "all",
    "num_prompts": 2048,
    "data_seed": 42,
    "max_completion_length": 10240,
    "soft_overlong_start": 8192,
    "question_batch_size": 128,
    "mini_batch_size": 64,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": None,  # derived from mini_batch_size
    "num_generations": 8,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 10,
    "lr_scheduler_type": "constant_with_warmup",
    "max_grad_norm": 1.0,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "loss_type": "dapo",
    "scale_rewards": "group",
    "beta": 0.0,
    "entropy_coef": 0.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "vllm_importance_sampling_mode": "token_truncate",
    "vllm_importance_sampling_clip_max": 2.0,
    "num_train_epochs": 1.0,
    "max_steps": 200,
    "logging_steps": 1,
    "save_steps": 20,
    "save_total_limit": 10,
    "save_only_model": False,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "vllm_gpu_memory_utilization": 0.5,
    "vllm_tensor_parallel_size": 1,
    "report_to": "wandb",
}

LORA_DEFAULTS: dict[str, Any] = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_target_modules": "all-linear",
    "lora_bias": "none",
}

EAPO_DEFAULTS: dict[str, Any] = {
    "eapo_kappa": 1.3862943611198906,  # log(4)
    "eapo_kappa_decay_steps": 0,  # constant strength by default
}


def merged_defaults(method_defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = {**COMMON_TRAINING_DEFAULTS, **LORA_DEFAULTS}
    if method_defaults:
        defaults.update(method_defaults)
    return defaults


def comma_split(value: str | None) -> list[str] | None:
    if value is None or value.lower() == "none":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def scale_rewards_value(value: str | bool) -> str:
    """Advantage std normalization: ``group`` / ``batch`` / ``none`` (``true``/``false`` also accepted)."""
    if isinstance(value, bool):
        return "group" if value else "none"
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "group"}:
        return "group"
    if lowered in {"0", "false", "no", "n", "none"}:
        return "none"
    if lowered == "batch":
        return "batch"
    raise argparse.ArgumentTypeError(f"Expected group/batch/none, got {value!r}.")


def sanitize_run_name_part(value: str) -> str:
    value = value.strip().replace("/", "_")
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value
    ).strip("_")


def auto_run_name(algorithm: str, args: argparse.Namespace) -> str:
    model = sanitize_run_name_part(
        args.model_name.rstrip("/").split("/")[-1].split(":")[-1]
    )
    lora = f"lora{args.lora_r}"
    return (
        f"{algorithm}_{model}_{lora}_n{args.num_prompts}_seed{args.data_seed}"
        f"_lr{args.learning_rate:g}_mcl{args.max_completion_length}_sop{args.soft_overlong_start}"
    )


def add_common_training_args(
    parser: argparse.ArgumentParser, method_defaults: dict[str, Any] | None = None
) -> None:
    defaults = merged_defaults(method_defaults)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory whose model, optimizer, scheduler, and RNG state should be restored.",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Existing W&B run ID to resume with resume='must'.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model ID or local model directory to train",
    )
    parser.add_argument("--dataset-name", type=str, default=defaults["dataset_name"])
    parser.add_argument("--dataset-config", type=str, default=defaults["dataset_config"])
    parser.add_argument("--num-prompts", type=int, default=defaults["num_prompts"])
    parser.add_argument("--data-seed", type=int, default=defaults["data_seed"])
    parser.add_argument("--train-jsonl", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-cache-dir", type=str, default=DEFAULT_DATA_CACHE_DIR)
    parser.add_argument(
        "--question-batch-size",
        type=int,
        default=defaults["question_batch_size"],
        help="Generation batch, counted in prompt-completion pairs (must be divisible by --num-generations).",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=defaults["mini_batch_size"],
        help=(
            "Sequences consumed per optimizer step. question_batch_size / mini_batch_size optimizer steps "
            "run per generation batch."
        ),
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults["per_device_train_batch_size"])
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults["gradient_accumulation_steps"],
        help="Overrides the value derived from --mini-batch-size.",
    )
    parser.add_argument("--num-generations", type=int, default=defaults["num_generations"])
    parser.add_argument("--max-completion-length", type=int, default=defaults["max_completion_length"])
    parser.add_argument(
        "--soft-overlong-start",
        type=int,
        default=defaults["soft_overlong_start"],
        help=(
            "Completion length where DAPO soft-overlong punishment begins (8192 by default). "
            "The reward decreases linearly from 0 at this length to -1 at "
            "--max-completion-length. Set both values equal to disable soft-overlong punishment."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"LoRA learning rate (default: {defaults['learning_rate']:g}).",
    )
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])
    parser.add_argument("--lr-scheduler-type", type=str, default=defaults["lr_scheduler_type"])
    parser.add_argument("--max-grad-norm", type=float, default=defaults["max_grad_norm"])
    parser.add_argument("--epsilon", type=float, default=defaults["epsilon"])
    parser.add_argument("--epsilon-high", type=float, default=defaults["epsilon_high"])
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=defaults["entropy_coef"],
        help=(
            "Coefficient on TRL's entropy regularization term (mean per-token entropy "
            "added to the loss). 0 disables the bonus."
        ),
    )
    parser.add_argument("--loss-type", type=str, default=defaults["loss_type"])
    parser.add_argument("--scale-rewards", type=scale_rewards_value, default=defaults["scale_rewards"])
    parser.add_argument("--beta", type=float, default=defaults["beta"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--rollout-is-clip", type=float, default=defaults["vllm_importance_sampling_clip_max"])
    parser.add_argument("--rollout-is-mode", type=str, default=defaults["vllm_importance_sampling_mode"], choices=["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"])
    parser.add_argument("--num-train-epochs", type=float, default=defaults["num_train_epochs"])
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument("--logging-steps", type=int, default=defaults["logging_steps"])
    parser.add_argument("--save-steps", type=int, default=defaults["save_steps"])
    parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"])
    parser.add_argument("--save-only-model", type=str_to_bool, default=defaults["save_only_model"])
    parser.add_argument("--bf16", type=str_to_bool, default=defaults["bf16"])
    parser.add_argument("--fp16", type=str_to_bool, default=defaults["fp16"])
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=defaults["gradient_checkpointing"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=defaults["vllm_gpu_memory_utilization"])
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=defaults["vllm_tensor_parallel_size"])
    parser.add_argument("--report-to", type=str, default=os.getenv("REPORT_TO", defaults["report_to"]))
    parser.add_argument("--lora-r", type=int, default=defaults["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=defaults["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=defaults["lora_target_modules"],
        help="all-linear (default), comma-separated module names, or none for PEFT's architecture defaults",
    )
    parser.add_argument("--lora-bias", type=str, default=defaults["lora_bias"])


def resolve_learning_rate(
    args: argparse.Namespace, method_defaults: dict[str, Any] | None = None
) -> float:
    """An explicit ``--learning-rate`` overrides the LoRA default."""
    if args.learning_rate is not None:
        return float(args.learning_rate)
    defaults = merged_defaults(method_defaults)
    return float(defaults["learning_rate"])


def world_size() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def resolve_gradient_accumulation_steps(args: argparse.Namespace) -> int:
    """``mini_batch_size`` sequences per optimizer step, spread over the available processes."""
    if getattr(args, "gradient_accumulation_steps", None):
        return int(args.gradient_accumulation_steps)
    global_batch = args.per_device_train_batch_size * world_size()
    if args.mini_batch_size % global_batch != 0:
        raise ValueError(
            f"--mini-batch-size ({args.mini_batch_size}) must be divisible by "
            f"per_device_train_batch_size * world_size ({global_batch})."
        )
    steps = args.mini_batch_size // global_batch
    if args.question_batch_size % args.mini_batch_size != 0:
        raise ValueError(
            f"--question-batch-size ({args.question_batch_size}) must be divisible by "
            f"--mini-batch-size ({args.mini_batch_size})."
        )
    return steps


def report_to_list(value: str) -> list[str]:
    if value.lower() in {"", "none", "no"}:
        return []
    return comma_split(value) or []


def build_config_kwargs(
    args: argparse.Namespace,
    output_dir: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "run_name": args.run_name,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": resolve_gradient_accumulation_steps(args),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": args.save_only_model,
        "save_strategy": "steps",
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "generation_batch_size": args.question_batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        "entropy_coef": args.entropy_coef,
        "loss_type": args.loss_type,
        "scale_rewards": args.scale_rewards,
        "chat_template_kwargs": {},
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_clip_max": args.rollout_is_clip,
        "vllm_importance_sampling_mode": args.rollout_is_mode,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "disable_dropout": True,
        "remove_unused_columns": False,
        "report_to": report_to_list(args.report_to),
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "use_liger_kernel": False,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def build_training_args(
    config_class: type,
    args: argparse.Namespace,
    output_dir: str,
    extra_kwargs: dict[str, Any] | None = None,
):
    return config_class(**build_config_kwargs(args, output_dir, extra_kwargs))


def build_lora_config(args: argparse.Namespace):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=(
            "all-linear" if args.lora_target_modules == "all-linear"
            else comma_split(args.lora_target_modules)
        ),
    )


def _torch_dtype_from_args(args: argparse.Namespace):
    """Storage dtype for the frozen base; PEFT keeps trainable adapters in fp32."""
    import torch

    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def load_model_for_training(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=_torch_dtype_from_args(args),
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False

    from peft import get_peft_model

    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()
    return model
