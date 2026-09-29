from __future__ import annotations

import json
import os
from typing import Any

import requests
from datasets import load_dataset

from ..math_protocol import boxed_math_prompt


def _make_prompt(problem: str) -> list[dict[str, str]]:
    return [{"content": boxed_math_prompt(problem), "role": "user"}]


def _write_jsonl(rows: list[dict[str, Any]], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _jsonl_has_sources(filepath: str, required_sources: set[str]) -> bool:
    if not os.path.exists(filepath):
        return False

    seen_sources: set[str] = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return False
            seen_sources.add(str(data.get("source_dataset", "unknown")))
    return required_sources.issubset(seen_sources)


def _jsonl_row_count(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    with open(filepath, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _format_examples(filepath: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Error parsing line {line_num + 1}: {exc}")
                continue

            problem = str(data["problem"])
            formatted_example = {
                "global_id": data.get("global_id", line_num),
                "original_id": data.get(
                    "original_id", data.get("id", data.get("problem_idx", line_num))
                ),
                "source_dataset": data.get("source_dataset", "unknown"),
                "problem": problem,
                "answer": str(data["answer"]),
                "solution": data.get("solution", ""),
                "url": data.get("url", ""),
                "prompt": _make_prompt(problem),
            }
            if "split" in data:
                formatted_example["split"] = data.get("split", "unknown")
            if "benchmark" in data:
                formatted_example["benchmark"] = data.get(
                    "benchmark", data.get("source_dataset", "unknown")
                )
            if "problem_type" in data:
                formatted_example["problem_type"] = data.get("problem_type", [])
            examples.append(formatted_example)
    return examples


def _print_source_counts(examples: list[dict[str, Any]]) -> None:
    source_counts: dict[str, int] = {}
    for example in examples:
        source = str(example["source_dataset"])
        source_counts[source] = source_counts.get(source, 0) + 1
    for source, count in source_counts.items():
        print(f"  {source}: {count} problems")


def download_and_combine_aime_datasets(data_dir: str = "./data/aime") -> str:
    preview_datasets = {
        "test2024": {
            "url": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2024.jsonl",
            "benchmark": "aime_2024",
        },
        "test2025-I": {
            "url": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-I.jsonl",
            "benchmark": "aime_2025",
        },
        "test2025-II": {
            "url": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-II.jsonl",
            "benchmark": "aime_2025",
        },
    }
    required_sources = {"aime_2024", "aime_2025", "aime_2026"}

    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "aime.jsonl")
    if _jsonl_has_sources(combined_filepath, required_sources):
        print(f"Combined AIME dataset already exists at {combined_filepath}")
        return combined_filepath
    if os.path.exists(combined_filepath):
        print(
            f"Existing AIME dataset at {combined_filepath} is missing required yearly sources; rebuilding it"
        )

    print("Downloading and combining AIME datasets...")
    all_problems: list[dict[str, Any]] = []
    global_id = 0
    for dataset_name, config in preview_datasets.items():
        url = str(config["url"])
        benchmark = str(config["benchmark"])
        print(f"  Downloading {dataset_name}...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"    Error downloading {dataset_name}: {exc}")
            continue

        for line_num, line in enumerate(response.text.strip().split("\n")):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"    Warning: error parsing line {line_num + 1} in {dataset_name}: {exc}"
                )
                continue
            data["source_dataset"] = benchmark
            data["benchmark"] = benchmark
            data["split"] = dataset_name
            data["original_id"] = data.get("id", line_num)
            data["global_id"] = global_id
            all_problems.append(data)
            global_id += 1

    print("  Downloading aime_2026 from math-ai/aime26...")
    aime26 = load_dataset("math-ai/aime26")
    for split_name, split_data in aime26.items():
        print(f"  Processing aime_2026:{split_name} ({len(split_data)} samples)")
        for i, item in enumerate(split_data):
            all_problems.append(
                {
                    "global_id": global_id,
                    "original_id": item.get("id", item.get("problem_idx", i)),
                    "source_dataset": "aime_2026",
                    "benchmark": "aime_2026",
                    "split": split_name,
                    "problem": item.get("problem", item.get("question", "")),
                    "answer": str(item.get("answer", item.get("final_answer", ""))),
                    "solution": item.get("solution", ""),
                    "url": "https://huggingface.co/datasets/math-ai/aime26",
                }
            )
            global_id += 1

    if not all_problems:
        raise RuntimeError("No AIME problems were successfully downloaded")

    _write_jsonl(all_problems, combined_filepath)
    print(
        f"Combined {len(all_problems)} problems from AIME 2024, AIME 2025, and AIME 2026"
    )
    print(f"Saved to: {combined_filepath}")
    for dataset_name in sorted(required_sources):
        count = sum(
            1 for item in all_problems if item["source_dataset"] == dataset_name
        )
        print(f"  {dataset_name}: {count} problems")
    return combined_filepath


def load_aime_dataset(data_dir: str = "./data/aime") -> list[dict[str, Any]]:
    examples = _format_examples(download_and_combine_aime_datasets(data_dir))
    print(f"Loaded {len(examples)} problems from combined AIME dataset")
    _print_source_counts(examples)
    return examples


def download_and_combine_hmmt_datasets(data_dir: str = "./data/hmmt") -> str:
    datasets = {
        "hmmt_2025": "MathArena/hmmt_nov_2025",
        "hmmt_2026": "MathArena/hmmt_feb_2026",
    }
    required_sources = set(datasets)

    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "hmmt.jsonl")
    if _jsonl_has_sources(combined_filepath, required_sources):
        print(f"Combined HMMT dataset already exists at {combined_filepath}")
        return combined_filepath
    if os.path.exists(combined_filepath):
        print(
            f"Existing HMMT dataset at {combined_filepath} is missing required yearly sources; rebuilding it"
        )

    print("Downloading and combining HMMT datasets from Hugging Face...")
    all_problems: list[dict[str, Any]] = []
    global_id = 0
    for dataset_name, repo_id in datasets.items():
        dataset = load_dataset(repo_id)
        for split_name, split_data in dataset.items():
            print(
                f"  Processing {dataset_name}:{split_name} ({len(split_data)} samples)"
            )
            for i, item in enumerate(split_data):
                problem_types = item.get("problem_type", [])
                if problem_types is None:
                    problem_types = []
                elif not isinstance(problem_types, list):
                    problem_types = [str(problem_types)]
                all_problems.append(
                    {
                        "global_id": global_id,
                        "original_id": item.get("problem_idx", i),
                        "source_dataset": dataset_name,
                        "benchmark": dataset_name,
                        "split": split_name,
                        "problem": item.get("problem", ""),
                        "answer": str(item.get("answer", "")),
                        "problem_type": problem_types,
                        "solution": item.get("solution", ""),
                        "url": f"https://huggingface.co/datasets/{repo_id}",
                    }
                )
                global_id += 1

    _write_jsonl(all_problems, combined_filepath)
    print(f"Combined {len(all_problems)} problems from {len(datasets)} HMMT datasets")
    print(f"Saved to: {combined_filepath}")
    for dataset_name in datasets:
        count = sum(
            1 for item in all_problems if item["source_dataset"] == dataset_name
        )
        print(f"  {dataset_name}: {count} problems")
    return combined_filepath


def load_hmmt_dataset(data_dir: str = "./data/hmmt") -> list[dict[str, Any]]:
    examples = _format_examples(download_and_combine_hmmt_datasets(data_dir))
    print(f"Loaded {len(examples)} problems from combined HMMT dataset")
    _print_source_counts(examples)
    return examples


def download_math500_dataset(data_dir: str = "./data/math500") -> str:
    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "math500.jsonl")
    if _jsonl_row_count(combined_filepath) == 134:
        print(f"MATH-500 dataset already exists at {combined_filepath}")
        return combined_filepath
    if os.path.exists(combined_filepath):
        print(
            f"Existing MATH-500 cache is not the 134-problem level-5 subset; rebuilding it at {combined_filepath}"
        )

    print("Downloading MATH-500 dataset from Hugging Face...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")

    all_problems: list[dict[str, Any]] = []
    global_id = 0
    for split_name, split_data in dataset.items():
        print(f"  Processing split: {split_name} ({len(split_data)} samples)")
        for i, item in enumerate(split_data):
            if item["level"] == 5:
                all_problems.append(
                    {
                        "global_id": global_id,
                        "original_id": i,
                        "source_dataset": split_name,
                        "problem": item.get("problem", ""),
                        "answer": str(item.get("answer", "")),
                        "solution": item.get("solution", ""),
                        "url": item.get("url", ""),
                    }
                )
                global_id += 1

    _write_jsonl(all_problems, combined_filepath)
    print(f"Saved to: {combined_filepath}")
    return combined_filepath


def load_math500_dataset(data_dir: str = "./data/math500") -> list[dict[str, Any]]:
    examples = _format_examples(download_math500_dataset(data_dir))
    print(f"Loaded {len(examples)} problems from combined MATH-500 dataset")
    _print_source_counts(examples)
    return examples


def download_amc23_dataset(data_dir: str = "./data/amc23") -> str:
    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "amc23.jsonl")
    if os.path.exists(combined_filepath):
        print(f"Combined AMC23 dataset already exists at {combined_filepath}")
        return combined_filepath

    print("Downloading AMC23 dataset from Hugging Face...")
    dataset = load_dataset("zwhe99/amc23")

    all_problems: list[dict[str, Any]] = []
    global_id = 0
    for split_name, split_data in dataset.items():
        print(f"  Processing split: {split_name} ({len(split_data)} samples)")
        for i, item in enumerate(split_data):
            answer = item.get("answer", "")
            try:
                answer = int(answer)
            except (TypeError, ValueError):
                pass
            all_problems.append(
                {
                    "global_id": global_id,
                    "original_id": i,
                    "source_dataset": split_name,
                    "problem": item.get("question", ""),
                    "answer": str(answer),
                    "solution": item.get("solution", ""),
                    "url": item.get("url", ""),
                }
            )
            global_id += 1

    _write_jsonl(all_problems, combined_filepath)
    print(f"Combined {len(all_problems)} problems from AMC23 dataset")
    print(f"Saved to: {combined_filepath}")
    return combined_filepath


def load_amc_dataset(data_dir: str = "./data/amc23") -> list[dict[str, Any]]:
    examples = _format_examples(download_amc23_dataset(data_dir))
    print(f"Loaded {len(examples)} problems from combined AMC23 dataset")
    _print_source_counts(examples)
    return examples
