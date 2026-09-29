from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl.rewards import get_soft_overlong_punishment

from .math_protocol import canonicalize_math_messages, math_answer_is_correct


def load_training_tokenizer(model_name: str):
    """Load the model's tokenizer and native chat template for training."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def math_reward_func(completions, answer=None, **kwargs):
    if answer is None:
        return [0.0 for _ in completions]

    rewards = []
    for completion, gold in zip(completions, answer):
        text = (
            completion[-1]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        rewards.append(float(math_answer_is_correct(text, str(gold))))
    return rewards


def build_reward_funcs(
    *,
    max_completion_length: int,
    soft_overlong_start: int,
):
    if not 0 <= soft_overlong_start <= max_completion_length:
        raise ValueError(
            "soft_overlong_start must satisfy 0 <= soft_overlong_start <= max_completion_length; "
            f"got {soft_overlong_start} and {max_completion_length}."
        )

    reward_funcs = [math_reward_func]
    # Equal limits disable the soft overlong penalty.
    if soft_overlong_start < max_completion_length:
        reward_funcs.append(
            get_soft_overlong_punishment(
                max_completion_len=max_completion_length,
                soft_punish_cache=max_completion_length - soft_overlong_start,
            )
        )
    return reward_funcs


def _get_ground_truth(example: dict[str, Any]) -> str:
    reward_model = example.get("reward_model") or {}
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    if example.get("answer") is not None:
        return str(example["answer"])
    if example.get("solution") is not None:
        return str(example["solution"])
    return ""


def _get_prompt(example: dict[str, Any], use_source_prompt: bool = True) -> Any:
    source_prompt = example.get("source_prompt")
    if use_source_prompt and source_prompt:
        return source_prompt
    return example["prompt"]


CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _text_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(prompt)


def contains_chinese_character(value: Any) -> bool:
    return CHINESE_CHAR_RE.search(str(value)) is not None


def has_chinese_prompt(example: dict[str, Any], use_source_prompt: bool = True) -> bool:
    return contains_chinese_character(
        _text_from_prompt(_get_prompt(example, use_source_prompt=use_source_prompt))
    )


def normalize_dapo_example(
    example: dict[str, Any], use_source_prompt: bool = True
) -> dict[str, Any]:
    answer = _get_ground_truth(example)
    prompt = canonicalize_math_messages(
        _get_prompt(example, use_source_prompt=use_source_prompt)
    )
    solution = str(example.get("solution") or answer)
    extra_info = example.get("extra_info") or {}
    index = extra_info.get("index") if isinstance(extra_info, dict) else None
    return {
        "prompt": prompt,
        "answer": answer,
        "solution": solution,
        "data_source": example.get("data_source", "math_dapo"),
        "ability": example.get("ability", "MATH"),
        "index": index,
    }


def save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def sample_and_save_dapo_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    num_prompts: int,
    seed: int,
    output_path: str | Path,
    use_source_prompt: bool = True,
) -> Path:
    config = None if dataset_config in {None, "", "none", "None"} else dataset_config
    dataset = (
        load_dataset(dataset_name, config, split="train")
        if config
        else load_dataset(dataset_name, split="train")
    )
    candidate_indices = [
        idx
        for idx in range(len(dataset))
        if not has_chinese_prompt(
            dataset[int(idx)], use_source_prompt=use_source_prompt
        )
    ]
    count = min(num_prompts, len(candidate_indices))
    rng = random.Random(seed)
    indices = rng.sample(candidate_indices, count)
    rows = [
        normalize_dapo_example(dataset[int(i)], use_source_prompt=use_source_prompt)
        for i in indices
    ]
    return save_jsonl(rows, output_path)


def load_or_create_training_dataset(
    *,
    train_jsonl: str | None,
    data_cache_dir: str,
    dataset_name: str,
    dataset_config: str | None,
    num_prompts: int,
    seed: int,
) -> tuple[Dataset, Path]:
    config_name = (
        "default"
        if dataset_config in {None, "", "none", "None"}
        else str(dataset_config).replace("/", "_")
    )
    dataset_slug = dataset_name.rstrip("/").split("/")[-1].replace("/", "_")
    path = (
        Path(train_jsonl)
        if train_jsonl
        else Path(data_cache_dir)
        / f"{dataset_slug}_{config_name}_nozh_n{num_prompts}_seed{seed}.jsonl"
    )
    if not path.exists():
        sample_and_save_dapo_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            num_prompts=num_prompts,
            seed=seed,
            output_path=path,
        )
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.map(
        lambda example: {"prompt": canonicalize_math_messages(example["prompt"])},
        desc="Applying boxed-answer prompt",
    )
    return dataset, path
