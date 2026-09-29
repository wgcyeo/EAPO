from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# vLLM launches worker subprocesses; spawn avoids CUDA re-init failures after fork.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from huggingface_hub import snapshot_download
from peft import PeftConfig
from vllm.lora.request import LoRARequest

from .eval_utils import (
    create_vllm_engine,
    estimate_pass_at_k,
    evaluate_model_on_dataset,
    sanitize_name,
)
from .loaders import (
    load_aime_dataset,
    load_amc_dataset,
    load_hmmt_dataset,
    load_math500_dataset,
)


SUPPORTED_DATASETS = (
    "AIME",
    "AIME24",
    "AIME25",
    "AIME26",
    "HMMT",
    "HMMT25",
    "HMMT26",
    "AMC23",
    "MATH500",
)
NUM_ROLLOUTS = 32
DEFAULT_PROMPT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 16384
RESULT_SCHEMA_VERSION = 2
DATASET_ALIASES = {
    "AIME": "AIME",
    "AIME24": "AIME24",
    "AIME2024": "AIME24",
    "AIME_2024": "AIME24",
    "AIME25": "AIME25",
    "AIME2025": "AIME25",
    "AIME_2025": "AIME25",
    "AIME26": "AIME26",
    "AIME2026": "AIME26",
    "AIME_2026": "AIME26",
    "AMC23": "AMC23",
    "AMC": "AMC23",
    "MATH500": "MATH500",
    "MATH": "MATH500",
    "HMMT": "HMMT",
    "HMMT25": "HMMT25",
    "HMMT2025": "HMMT25",
    "HMMT_2025": "HMMT25",
    "HMMT26": "HMMT26",
    "HMMT2026": "HMMT26",
    "HMMT_2026": "HMMT26",
}
DATASET_GROUPS = {
    "AIME": ("AIME24", "AIME25", "AIME26"),
    "HMMT": ("HMMT25", "HMMT26"),
}
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "AIME24": {
        "loader": load_aime_dataset,
        "data_dir_arg": "aime_data_dir",
        "display_name": "AIME 2024",
        "source_dataset": "aime_2024",
    },
    "AIME25": {
        "loader": load_aime_dataset,
        "data_dir_arg": "aime_data_dir",
        "display_name": "AIME 2025",
        "source_dataset": "aime_2025",
    },
    "AIME26": {
        "loader": load_aime_dataset,
        "data_dir_arg": "aime_data_dir",
        "display_name": "AIME 2026",
        "source_dataset": "aime_2026",
    },
    "AMC23": {
        "loader": load_amc_dataset,
        "data_dir_arg": "amc23_data_dir",
        "display_name": "AMC23 (combined via load_amc_dataset)",
    },
    "MATH500": {
        "loader": load_math500_dataset,
        "data_dir_arg": "math500_data_dir",
        "display_name": "MATH500 (combined via load_math500_dataset)",
    },
    "HMMT25": {
        "loader": load_hmmt_dataset,
        "data_dir_arg": "hmmt_data_dir",
        "display_name": "HMMT 2025",
        "source_dataset": "hmmt_2025",
    },
    "HMMT26": {
        "loader": load_hmmt_dataset,
        "data_dir_arg": "hmmt_data_dir",
        "display_name": "HMMT 2026",
        "source_dataset": "hmmt_2026",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base model or single LoRA adapter with 32 rollouts per math prompt."
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--lora-path",
        type=str,
        help="Path or Hugging Face repo ID of trained LoRA adapter",
    )
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Path or Hugging Face repo ID of the model to evaluate",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="AIME24,AIME25,AIME26,HMMT26,AMC23,MATH500",
        help=(
            "Comma-separated datasets. AIME expands to AIME24/25/26 and HMMT expands to HMMT25/26; "
            "individual years are also accepted. AMC selects AMC23; MATH500 is also supported."
        ),
    )
    parser.add_argument(
        "--aime-data-dir",
        type=str,
        default="./data/aime",
        help="AIME dataset cache directory",
    )
    parser.add_argument(
        "--amc23-data-dir",
        type=str,
        default="./data/amc23",
        help="AMC23 dataset cache directory",
    )
    parser.add_argument(
        "--math500-data-dir",
        type=str,
        default="./data/math500",
        help="MATH500 dataset cache directory",
    )
    parser.add_argument(
        "--hmmt-data-dir",
        type=str,
        default="./data/hmmt",
        help="HMMT dataset cache directory",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/eval_model",
        help="Root directory for eval outputs; each run creates {run_name}_{eval_time}/ under this root.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run/model name used in the output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional exact directory for metrics.json and raw_responses.json; overrides --results-root naming.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N problems per dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore compatible saved results and re-evaluate all requested datasets.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_ROLLOUTS,
        help=f"Number of rollouts per prompt. This script is intended to run exactly {NUM_ROLLOUTS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=NUM_ROLLOUTS,
        help=f"Rollouts per prompt in each vLLM call; total rollouts remain {NUM_ROLLOUTS}.",
    )
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=DEFAULT_PROMPT_BATCH_SIZE,
        help="Problems submitted together so vLLM can continuously batch their rollouts.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.8, help="Top-p for sampling"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k for sampling; -1 disables it",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for vLLM tensor parallelism",
    )
    parser.add_argument(
        "--vllm-performance-mode",
        choices=("balanced", "interactivity", "throughput"),
        default="throughput",
        help="vLLM runtime mode; throughput is best suited to offline evaluation.",
    )
    parser.add_argument(
        "--max-lora-rank", type=int, default=32, help="Maximum LoRA rank passed to vLLM"
    )
    return parser.parse_args()


def parse_dataset_names(raw_value: str) -> list[str]:
    dataset_names: list[str] = []
    for token in raw_value.split(","):
        name = token.strip().upper()
        if not name:
            continue
        selection = DATASET_ALIASES.get(name)
        if selection is None:
            supported = ", ".join(SUPPORTED_DATASETS)
            raise ValueError(
                f"Unsupported dataset: {token!r}. Supported datasets: {supported}"
            )
        for canonical in DATASET_GROUPS.get(selection, (selection,)):
            if canonical not in dataset_names:
                dataset_names.append(canonical)
    if not dataset_names:
        raise ValueError("No valid datasets provided to --datasets")
    return dataset_names


def _split_hf_repo_and_subfolder(model_ref: str) -> tuple[str, str | None]:
    parts = [part for part in model_ref.strip("/").split("/") if part]
    if len(parts) < 2:
        return model_ref, None
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) or None
    return repo_id, subfolder


def resolve_lora_adapter_ref(lora_ref: str) -> tuple[str, str]:
    if os.path.exists(lora_ref):
        return lora_ref, lora_ref

    repo_id, subfolder = _split_hf_repo_and_subfolder(lora_ref)
    snapshot_path = snapshot_download(repo_id=repo_id)
    adapter_path = (
        os.path.join(snapshot_path, subfolder) if subfolder else snapshot_path
    )

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Resolved Hugging Face LoRA ref {lora_ref!r} to {adapter_path!r}, "
            "but that directory does not exist in the downloaded snapshot."
        )
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"Hugging Face LoRA ref {lora_ref!r} did not resolve to a directory containing "
            f"'adapter_config.json': {adapter_path!r}"
        )
    return adapter_path, adapter_path


def infer_run_name(
    model_ref: str,
    explicit_run_name: str | None = None,
) -> str:
    if explicit_run_name:
        return sanitize_name(explicit_run_name)

    path = Path(model_ref)
    if path.exists():
        path = path.resolve()
    parts = list(path.parts)
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            return sanitize_name(parts[idx + 1])

    basename = path.name or model_ref.rstrip("/").split("/")[-1]
    if basename.startswith("checkpoint-") and path.parent.name:
        basename = path.parent.name
    return sanitize_name(basename)


def make_output_dir(*, results_root: str, run_name: str, eval_time: str) -> str:
    output_dir = os.path.join(results_root, f"{sanitize_name(run_name)}_{eval_time}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_metric_notes() -> dict[str, str]:
    return {
        "avg@32": "Mean correctness across the 32 sampled rollouts, averaged over prompts.",
        "pass@32": (
            "Per-problem indicator that at least one of 32 sampled rollouts is correct; "
            "implemented as the Chen et al. pass@k estimator with n=32 and k=32."
        ),
        "is_correct": "Per-problem binary correctness sequence for each sampled completion (1=correct, 0=incorrect).",
    }


def print_summary(
    *,
    dataset_name: str,
    lora_path: str | None,
    model_path: str,
    base_model_name: str,
    metrics: dict[str, Any],
) -> None:
    mode_label = "Single LoRA" if lora_path is not None else "Base Model"
    print(f"\n=== {dataset_name} Evaluation ({mode_label}) ===")
    print(
        "Metric definition: avg@32 and pass@32 are computed from 32 sampled rollouts per problem"
    )
    print(f"Base model: {base_model_name}")
    if lora_path is not None:
        print(f"LoRA path: {lora_path}")
    else:
        print(f"Model path: {model_path}")
    print(f"avg@32: {float(metrics['avg@32']):.4f}")
    print(f"pass@32: {float(metrics['pass@32']):.4f}")


def benchmark_key_for_source(dataset_name: str, sample: dict[str, Any]) -> str:
    source = str(sample.get("benchmark") or sample.get("source_dataset") or "unknown")
    lowered = source.lower()
    dataset = dataset_name.upper()

    if dataset == "AIME":
        if "2024" in lowered:
            return "aime_2024"
        if "2025" in lowered:
            return "aime_2025"
        if "2026" in lowered or "aime26" in lowered:
            return "aime_2026"
    if dataset == "HMMT":
        if "2025" in lowered:
            return "hmmt_2025"
        if "2026" in lowered:
            return "hmmt_2026"

    return source


def rebuild_pass32_avg32_metrics(
    result: dict[str, Any], *, dataset_name: str
) -> dict[str, Any]:
    per_problem = list(result.get("per_problem", []))
    raw_responses = list(result.get("raw_responses", []))
    total = len(per_problem)

    pass32_sum = 0.0
    avg32_sum = 0.0
    source_stats: dict[str, dict[str, float]] = {}

    for item in per_problem:
        source = benchmark_key_for_source(dataset_name, item)
        item["benchmark"] = source
        num_correct = int(item.get("num_correct_out_of_n", 0))
        pass32 = estimate_pass_at_k(
            num_samples=NUM_ROLLOUTS, num_correct=num_correct, k=NUM_ROLLOUTS
        )
        avg32 = float(num_correct / NUM_ROLLOUTS)

        item["avg@32"] = avg32
        item["pass@32"] = float(pass32)
        item.pop("pass@1", None)

        pass32_sum += float(pass32)
        avg32_sum += avg32

        source_stat = source_stats.setdefault(
            source,
            {"count": 0.0, "avg32_sum": 0.0, "pass32_sum": 0.0},
        )
        source_stat["count"] += 1.0
        source_stat["avg32_sum"] += avg32
        source_stat["pass32_sum"] += float(pass32)

    source_breakdown: dict[str, dict[str, float]] = {}
    for source, stats in sorted(source_stats.items()):
        count = max(int(stats["count"]), 1)
        source_breakdown[source] = {
            "count": int(stats["count"]),
            "avg@32": float(stats["avg32_sum"] / count),
            "pass@32": float(stats["pass32_sum"] / count),
        }

    for row in raw_responses:
        row["benchmark"] = benchmark_key_for_source(dataset_name, row)

    metrics = dict(result.get("metrics", {}))
    metrics["num_examples"] = int(total)
    metrics["num_samples_per_problem"] = NUM_ROLLOUTS
    metrics["avg@32"] = float(avg32_sum / max(total, 1))
    metrics.pop("pass@1", None)
    metrics.pop(f"pass@{NUM_ROLLOUTS}", None)
    metrics["pass@32"] = float(pass32_sum / max(total, 1))

    return {
        **result,
        "metrics": metrics,
        "source_breakdown": source_breakdown,
        "benchmark_breakdown": source_breakdown,
        "per_problem": per_problem,
        "raw_responses": raw_responses,
        "metric_notes": build_metric_notes(),
    }


def split_result_payload(
    result: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_result = {k: v for k, v in result.items() if k != "raw_responses"}
    raw_responses = list(result.get("raw_responses", []))
    return metrics_result, raw_responses


RESUME_GENERATION_KEYS = (
    "num_samples_per_problem",
    "batch_size",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "chat_template_kwargs",
    "seed",
)


def _load_json_if_present(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _atomic_json_dump(payload: dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _resume_signature(generation: dict[str, Any]) -> dict[str, Any]:
    return {key: generation.get(key) for key in RESUME_GENERATION_KEYS}


def _normalized_model_ref(model_ref: Any) -> str:
    value = str(model_ref)
    if os.path.exists(value):
        return os.path.realpath(value)
    return value.rstrip("/")


def _payload_is_compatible(
    payload: dict[str, Any],
    *,
    model_ref: str,
    base_model_name: str,
    generation: dict[str, Any],
    result_group_key: str,
) -> bool:
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False
    if str(payload.get("base_model_name")) != str(base_model_name):
        return False
    saved_model_ref = payload.get("model_ref")
    if _normalized_model_ref(saved_model_ref) != _normalized_model_ref(model_ref):
        return False
    if _resume_signature(dict(payload.get("generation", {}))) != _resume_signature(
        generation
    ):
        return False
    if result_group_key not in payload:
        return False
    group = payload[result_group_key]
    if not isinstance(group, dict) or any(name in DATASET_GROUPS for name in group):
        return False
    return all(
        _normalized_model_ref(item.get("model_ref")) == _normalized_model_ref(model_ref)
        for item in group.values()
        if isinstance(item, dict)
    )


def load_resume_state(
    *,
    metrics_json: str,
    raw_responses_json: str,
    model_ref: str,
    base_model_name: str,
    generation: dict[str, Any],
    result_group_key: str,
    force: bool,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[str],
]:
    metrics_payload = _load_json_if_present(metrics_json)
    raw_payload = _load_json_if_present(raw_responses_json)
    present_payloads = [
        payload for payload in (metrics_payload, raw_payload) if payload is not None
    ]

    if present_payloads and not all(
        _payload_is_compatible(
            payload,
            model_ref=model_ref,
            base_model_name=base_model_name,
            generation=generation,
            result_group_key=result_group_key,
        )
        for payload in present_payloads
    ):
        if force:
            return {}, {}, {}, []
        raise RuntimeError(
            f"Existing results in {os.path.dirname(metrics_json) or '.'} have a different "
            "format, model, or generation configuration. Use --force to replace them."
        )

    metrics_results = dict((metrics_payload or {}).get(result_group_key, {}))
    raw_results = dict((raw_payload or {}).get(result_group_key, {}))
    dataset_meta = dict((metrics_payload or raw_payload or {}).get("datasets", {}))
    requested = list((metrics_payload or raw_payload or {}).get("requested_datasets", []))

    return metrics_results, raw_results, dataset_meta, requested


def dataset_result_is_complete(
    *,
    dataset_name: str,
    dataset_meta: dict[str, Any],
    metrics_results: dict[str, dict[str, Any]],
    raw_results: dict[str, dict[str, Any]],
) -> bool:
    metrics_item = metrics_results.get(dataset_name)
    raw_item = raw_results.get(dataset_name)
    if not isinstance(metrics_item, dict) or not isinstance(raw_item, dict):
        return False
    expected_count = int(dataset_meta.get("num_examples", 0))
    metrics = metrics_item.get("metrics", {})
    raw_responses = raw_item.get("raw_responses", [])
    return (
        int(metrics.get("num_examples", -1)) == expected_count
        and isinstance(raw_responses, list)
        and len(raw_responses) == expected_count
    )


def build_output_payloads(
    *,
    eval_time: str,
    run_name: str,
    output_dir: str,
    metrics_json: str,
    raw_responses_json: str,
    args: argparse.Namespace,
    lora_path: str | None,
    model_path: str | None,
    model_ref: str,
    base_model_name: str,
    generation: dict[str, Any],
    requested_datasets: list[str],
    dataset_meta: dict[str, dict[str, Any]],
    result_group_key: str,
    result_item_key: str,
    metrics_results: dict[str, dict[str, Any]],
    raw_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": now,
        "updated_at_utc": now,
        "eval_time": eval_time,
        "run_name": run_name,
        "lora_path": lora_path,
        "model_path": model_path,
        "model_ref": model_ref,
        "base_model_name": base_model_name,
        "requested_datasets": requested_datasets,
        "completed_datasets": [name for name in metrics_results if name in raw_results],
        "generation": generation,
        "datasets": dataset_meta,
    }
    metrics_payload: dict[str, Any] = {
        **common,
        "results_dir": output_dir,
        "raw_responses_json": raw_responses_json,
        "aime_data_dir": args.aime_data_dir,
        "amc23_data_dir": args.amc23_data_dir,
        "math500_data_dir": args.math500_data_dir,
        "hmmt_data_dir": args.hmmt_data_dir,
        "metric_definitions": build_metric_notes(),
        result_group_key: metrics_results,
    }
    raw_payload: dict[str, Any] = {
        **common,
        "metrics_json": metrics_json,
        result_group_key: raw_results,
    }

    if len(metrics_results) == 1:
        only_dataset = next(iter(metrics_results))
        metrics_payload["dataset"] = dataset_meta[only_dataset]
        metrics_payload[result_item_key] = metrics_results[only_dataset]
        raw_payload["dataset"] = dataset_meta[only_dataset]
        raw_payload[result_item_key] = raw_results[only_dataset]

    return metrics_payload, raw_payload


def save_incremental_results(
    *,
    metrics_json: str,
    raw_responses_json: str,
    **payload_kwargs: Any,
) -> None:
    metrics_payload, raw_payload = build_output_payloads(
        metrics_json=metrics_json,
        raw_responses_json=raw_responses_json,
        **payload_kwargs,
    )
    # Write raw responses first; metrics.json acts as the completion marker.
    _atomic_json_dump(raw_payload, raw_responses_json)
    _atomic_json_dump(metrics_payload, metrics_json)


def load_datasets(
    *,
    args: argparse.Namespace,
    dataset_names: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    loaded_datasets: dict[str, list[dict[str, Any]]] = {}
    dataset_meta: dict[str, dict[str, Any]] = {}
    loader_cache: dict[tuple[Any, str], list[dict[str, Any]]] = {}

    for dataset_name in dataset_names:
        conf = DATASET_CONFIGS[dataset_name]
        loader = conf["loader"]
        data_dir_arg = str(conf["data_dir_arg"])
        data_dir = str(getattr(args, data_dir_arg))

        cache_key = (loader, data_dir)
        if cache_key not in loader_cache:
            print(f"Loading source data for {dataset_name} from {data_dir} ...")
            loader_cache[cache_key] = loader(data_dir)
        examples = list(loader_cache[cache_key])
        source_dataset = conf.get("source_dataset")
        if source_dataset is not None:
            examples = [
                example
                for example in examples
                if str(example.get("source_dataset", "")) == str(source_dataset)
            ]
        if args.limit is not None:
            examples = examples[: int(args.limit)]
        if not examples:
            raise RuntimeError(
                f"No examples found for {dataset_name}"
                + (
                    f" with source_dataset={source_dataset!r}"
                    if source_dataset is not None
                    else ""
                )
            )
        print(f"Evaluating {len(examples)} {dataset_name} problems")

        loaded_datasets[dataset_name] = examples
        metadata = {
            "name": str(conf["display_name"]),
            "num_examples": int(len(examples)),
            "limit": int(args.limit) if args.limit is not None else None,
            "data_dir": data_dir,
            "source_dataset": source_dataset,
        }
        dataset_meta[dataset_name] = metadata
    return loaded_datasets, dataset_meta


def main() -> None:
    args = parse_args()
    dataset_names = parse_dataset_names(args.datasets)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.prompt_batch_size < 1:
        raise ValueError("--prompt-batch-size must be >= 1")
    if args.num_samples != NUM_ROLLOUTS:
        raise ValueError(
            f"--num-samples must be {NUM_ROLLOUTS} for avg@32 and pass@32 evaluation"
        )

    lora_path = args.lora_path
    model_path = args.model_path
    if lora_path is not None:
        resolved_lora_path, peft_config_ref = resolve_lora_adapter_ref(lora_path)
        peft_cfg = PeftConfig.from_pretrained(peft_config_ref)
        base_model_name = str(peft_cfg.base_model_name_or_path)
        launch_model_name = base_model_name
        lora_request = LoRARequest("eval", 1, resolved_lora_path)
        model_label = "lora"
        enable_lora = True
        model_ref = lora_path
        result_group_key = "lora_results_by_dataset"
        result_item_key = "lora"
    else:
        base_model_name = model_path
        launch_model_name = model_path
        lora_request = None
        model_label = "base"
        enable_lora = False
        model_ref = model_path
        result_group_key = "base_results_by_dataset"
        result_item_key = "base"

    max_new_tokens = int(args.max_new_tokens)
    chat_template_kwargs = {}
    run_name = infer_run_name(model_ref, args.run_name)
    generation = {
        "num_samples_per_problem": NUM_ROLLOUTS,
        "batch_size": int(args.batch_size),
        "prompt_batch_size": int(args.prompt_batch_size),
        "vllm_performance_mode": str(args.vllm_performance_mode),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "max_new_tokens": max_new_tokens,
        "chat_template_kwargs": chat_template_kwargs,
        "seed": int(args.seed),
    }

    eval_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is not None:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = make_output_dir(
            results_root=args.results_root, run_name=run_name, eval_time=eval_time
        )
    metrics_json = os.path.join(output_dir, "metrics.json")
    raw_responses_json = os.path.join(output_dir, "raw_responses.json")

    print(f"Selected datasets: {', '.join(dataset_names)}")
    print(f"Evaluation run name: {run_name}")
    print(f"Output directory: {output_dir}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"Prompt batch size: {int(args.prompt_batch_size)}")
    datasets, dataset_meta = load_datasets(args=args, dataset_names=dataset_names)

    (
        metrics_results_by_dataset,
        raw_results_by_dataset,
        saved_dataset_meta,
        saved_requested,
    ) = load_resume_state(
        metrics_json=metrics_json,
        raw_responses_json=raw_responses_json,
        model_ref=model_ref,
        base_model_name=base_model_name,
        generation=generation,
        result_group_key=result_group_key,
        force=bool(args.force),
    )
    if args.force:
        for dataset_name in dataset_names:
            metrics_results_by_dataset.pop(dataset_name, None)
            raw_results_by_dataset.pop(dataset_name, None)
    all_dataset_meta = {**saved_dataset_meta, **dataset_meta}
    all_requested_datasets = list(dict.fromkeys([*saved_requested, *dataset_names]))

    pending_datasets: list[str] = []
    for dataset_name in dataset_names:
        complete = dataset_result_is_complete(
            dataset_name=dataset_name,
            dataset_meta=dataset_meta[dataset_name],
            metrics_results=metrics_results_by_dataset,
            raw_results=raw_results_by_dataset,
        )
        if complete and not args.force:
            print(f"Skipping {dataset_name}: compatible saved result already exists")
            print_summary(
                dataset_name=dataset_name,
                lora_path=lora_path,
                model_path=launch_model_name,
                base_model_name=base_model_name,
                metrics=metrics_results_by_dataset[dataset_name]["metrics"],
            )
        else:
            pending_datasets.append(dataset_name)

    if not pending_datasets:
        print("All requested datasets are already complete; vLLM was not launched.")
        print(f"Metrics JSON: {metrics_json}")
        print(f"Raw responses JSON: {raw_responses_json}")
        return

    launch_desc = (
        f"base model (+LoRA): {base_model_name}"
        if lora_path is not None
        else f"base model: {launch_model_name}"
    )
    print(f"Launching vLLM for {launch_desc}")
    llm, tokenizer = create_vllm_engine(
        model_name=launch_model_name,
        enable_lora=enable_lora,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_lora_rank=int(args.max_lora_rank),
        performance_mode=str(args.vllm_performance_mode),
    )

    try:
        for dataset_name in pending_datasets:
            raw_result = evaluate_model_on_dataset(
                model_label=model_label,
                llm=llm,
                tokenizer=tokenizer,
                lora_request=lora_request,
                dataset=datasets[dataset_name],
                num_samples=NUM_ROLLOUTS,
                batch_size=int(args.batch_size),
                prompt_batch_size=int(args.prompt_batch_size),
                max_new_tokens=max_new_tokens,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                seed=int(args.seed),
                dataset_label=dataset_name,
                keep_raw_responses=True,
                chat_template_kwargs=chat_template_kwargs,
            )
            result = rebuild_pass32_avg32_metrics(raw_result, dataset_name=dataset_name)
            metrics_result, raw_responses = split_result_payload(result)
            metrics_results_by_dataset[dataset_name] = {
                "model_ref": model_ref,
                **metrics_result,
            }
            raw_results_by_dataset[dataset_name] = {
                "model_ref": model_ref,
                "raw_responses": raw_responses,
            }
            print_summary(
                dataset_name=dataset_name,
                lora_path=lora_path,
                model_path=launch_model_name,
                base_model_name=base_model_name,
                metrics=result["metrics"],
            )
            save_incremental_results(
                metrics_json=metrics_json,
                raw_responses_json=raw_responses_json,
                eval_time=eval_time,
                run_name=run_name,
                output_dir=output_dir,
                args=args,
                lora_path=lora_path,
                model_path=model_path,
                model_ref=model_ref,
                base_model_name=base_model_name,
                generation=generation,
                requested_datasets=all_requested_datasets,
                dataset_meta=all_dataset_meta,
                result_group_key=result_group_key,
                result_item_key=result_item_key,
                metrics_results=metrics_results_by_dataset,
                raw_results=raw_results_by_dataset,
            )
            print(f"Saved {dataset_name} metrics to {metrics_json}")
            print(f"Saved {dataset_name} raw responses to {raw_responses_json}")
    finally:
        del llm
        gc.collect()

    print(f"Evaluation complete. Metrics JSON: {metrics_json}")
    print(f"Evaluation complete. Raw responses JSON: {raw_responses_json}")


if __name__ == "__main__":
    main()
