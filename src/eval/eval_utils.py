from __future__ import annotations

import math
import os
import random
import re
import time
from typing import Any

import torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .verify import reward


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_vllm_engine(
    *,
    model_name: str,
    enable_lora: bool,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    max_lora_rank: int = 32,
    performance_mode: str = "throughput",
) -> tuple[LLM, Any]:
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enable_lora=bool(enable_lora),
        max_lora_rank=int(max_lora_rank),
        gpu_memory_utilization=float(gpu_memory_utilization),
        tensor_parallel_size=int(tensor_parallel_size),
        performance_mode=str(performance_mode),
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def _fallback_chat_text(messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip()
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for piece in content:
                if isinstance(piece, dict):
                    txt = piece.get("text", "")
                    if txt:
                        parts.append(str(txt))
            content = "\n".join(parts)
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n\n".join(lines)


def render_chat_prompt(
    tokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render evaluation prompts with the same template kwargs used in training."""
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = dict(chat_template_kwargs or {})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
            if isinstance(text, str):
                return text
        except Exception:
            pass
    return _fallback_chat_text(messages, add_generation_prompt=add_generation_prompt)


def _sampling_params(
    *,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> SamplingParams:
    if float(temperature) <= 0.0:
        return SamplingParams(
            n=int(n),
            temperature=0.0,
            max_tokens=int(max_new_tokens),
            seed=int(seed),
        )
    return SamplingParams(
        n=int(n),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )


def generate_completions_batch_vllm(
    *,
    llm: LLM,
    tokenizer,
    prompt_messages_batch: list[list[dict[str, Any]]],
    lora_request: LoRARequest | None,
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seeds: list[int],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[list[str]]:
    if len(prompt_messages_batch) != len(seeds):
        raise ValueError("prompt_messages_batch and seeds must have the same length")
    if not prompt_messages_batch:
        return []

    prompt_texts = [
        render_chat_prompt(
            tokenizer=tokenizer,
            messages=prompt_messages,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        for prompt_messages in prompt_messages_batch
    ]
    completions_by_prompt: list[list[str]] = [[] for _ in prompt_texts]
    remaining = int(n_samples)
    chunk_idx = 0

    while remaining > 0:
        cur_batch = min(int(batch_size), remaining)
        chunk_sampling_params = [
            _sampling_params(
                n=cur_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=int(prompt_seed) + chunk_idx,
            )
            for prompt_seed in seeds
        ]

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request

        outputs = llm.generate(prompt_texts, chunk_sampling_params, **generate_kwargs)
        if len(outputs) != len(prompt_texts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} prompt outputs for {len(prompt_texts)} input prompts"
            )
        for prompt_idx, req_out in enumerate(outputs):
            chunk_outputs = list(getattr(req_out, "outputs", []))
            if len(chunk_outputs) != cur_batch:
                raise RuntimeError(
                    f"vLLM returned {len(chunk_outputs)} completions for prompt {prompt_idx}; "
                    f"expected {cur_batch}"
                )
            completions_by_prompt[prompt_idx].extend(
                str(getattr(out, "text", "") or "").strip() for out in chunk_outputs
            )
        remaining -= cur_batch
        chunk_idx += 1

    return completions_by_prompt


def estimate_pass_at_k(
    *,
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    n = int(num_samples)
    c = max(0, min(int(num_correct), n))
    k = int(k)

    if n < 1:
        return 0.0
    if k < 1:
        raise ValueError("pass@k requires k >= 1")
    if k > n:
        raise ValueError(f"pass@k requires k <= n, got k={k}, n={n}")
    if c == 0:
        return 0.0
    if k == 1:
        return float(c / n)
    if (n - c) < k:
        return 1.0
    return float(1.0 - (math.comb(n - c, k) / math.comb(n, k)))


def evaluate_model_on_dataset(
    *,
    model_label: str,
    llm: LLM,
    tokenizer,
    lora_request: LoRARequest | None,
    dataset: list[dict[str, Any]],
    num_samples: int,
    batch_size: int,
    prompt_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = -1,
    seed: int,
    dataset_label: str | None = None,
    keep_raw_responses: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be >= 1")

    total = len(dataset)
    avg_n_sum = 0.0
    pass1_sum = 0.0
    passn_sum = 0.0
    per_problem: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, float]] = {}

    progress_desc = f"Eval {model_label}" if not dataset_label else f"Eval {model_label} [{dataset_label}]"
    progress = tqdm(total=total, desc=progress_desc, dynamic_ncols=True)
    start_time = time.time()
    avg_n_key = f"avg@{int(num_samples)}"
    passn_key = f"pass@{int(num_samples)}"
    set_seed(int(seed))

    for batch_start in range(0, total, int(prompt_batch_size)):
        samples = dataset[batch_start : batch_start + int(prompt_batch_size)]
        completions_batch = generate_completions_batch_vllm(
            llm=llm,
            tokenizer=tokenizer,
            prompt_messages_batch=[sample["prompt"] for sample in samples],
            lora_request=lora_request,
            n_samples=num_samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=[int(seed) + (batch_start + idx) * 100 for idx in range(len(samples))],
            chat_template_kwargs=chat_template_kwargs,
        )
        for local_idx, (sample, completions) in enumerate(zip(samples, completions_batch, strict=True)):
            idx = batch_start + local_idx
            scores = [
                int(
                    reward(
                        txt,
                        sample["answer"],
                    )
                )
                for txt in completions
            ]

            num_correct = int(sum(scores))
            avg_n = float(num_correct / num_samples)
            pass1 = estimate_pass_at_k(num_samples=num_samples, num_correct=num_correct, k=1)
            passn = estimate_pass_at_k(num_samples=num_samples, num_correct=num_correct, k=int(num_samples))

            avg_n_sum += avg_n
            pass1_sum += pass1
            passn_sum += passn

            source = str(sample.get("source_dataset", "unknown"))
            stat = source_stats.setdefault(
                source,
                {"count": 0.0, "avg_n_sum": 0.0, "pass1_sum": 0.0, "passn_sum": 0.0},
            )
            stat["count"] += 1.0
            stat["avg_n_sum"] += avg_n
            stat["pass1_sum"] += float(pass1)
            stat["passn_sum"] += float(passn)

            per_problem.append(
                {
                    "global_id": int(sample.get("global_id", idx)),
                    "original_id": sample.get("original_id"),
                    "source_dataset": source,
                    avg_n_key: avg_n,
                    "pass@1": float(pass1),
                    passn_key: float(passn),
                    "num_correct_out_of_n": num_correct,
                    "is_correct": [int(s) for s in scores],
                }
            )
            if keep_raw_responses:
                raw_responses.append(
                    {
                        "global_id": int(sample.get("global_id", idx)),
                        "original_id": sample.get("original_id"),
                        "source_dataset": source,
                        "prompt": sample.get("prompt"),
                        "answer": sample.get("answer"),
                        "responses": completions,
                        "is_correct": [int(s) for s in scores],
                    }
                )

            seen = idx + 1
            progress.set_postfix(
                {
                    avg_n_key: f"{avg_n_sum / seen:.3f}",
                    f"pass@{int(num_samples)}": f"{passn_sum / seen:.3f}",
                },
                refresh=False,
            )
            progress.update(1)

    progress.close()

    elapsed = time.time() - start_time
    metrics = {
        "num_examples": int(total),
        "num_samples_per_problem": int(num_samples),
        "prompt_batch_size": int(prompt_batch_size),
        avg_n_key: float(avg_n_sum / max(total, 1)),
        "pass@1": float(pass1_sum / max(total, 1)),
        passn_key: float(passn_sum / max(total, 1)),
        "avg_correct_count": float(
            sum(item["num_correct_out_of_n"] for item in per_problem) / max(total, 1)
        ),
        "elapsed_sec": float(elapsed),
        "sec_per_problem": float(elapsed / max(total, 1)),
    }

    source_breakdown: dict[str, dict[str, float]] = {}
    for source, stat in sorted(source_stats.items()):
        count = max(int(stat["count"]), 1)
        source_breakdown[source] = {
            "count": int(stat["count"]),
            avg_n_key: float(stat["avg_n_sum"] / count),
            "pass@1": float(stat["pass1_sum"] / count),
            passn_key: float(stat["passn_sum"] / count),
        }

    result: dict[str, Any] = {
        "model_label": model_label,
        "metrics": metrics,
        "source_breakdown": source_breakdown,
        "per_problem": per_problem,
        "metric_notes": {
            avg_n_key: (
                f"Mean correctness across the {int(num_samples)} sampled completions, "
                "averaged over prompts."
            ),
            "pass@1": (
                f"Chen et al. unbiased pass@1 estimate from {int(num_samples)} samples; "
                f"numerically equal to {avg_n_key}."
            ),
            passn_key: (
                f"Chen et al. (2021) unbiased pass@{int(num_samples)} estimate computed as "
                f"1 - C(n-c, k) / C(n, k) with n={int(num_samples)}, k={int(num_samples)}, "
                "c=num_correct_out_of_n."
            ),
            "is_correct": "Per-problem binary correctness sequence for each sampled completion (1=correct, 0=incorrect).",
        },
    }
    if keep_raw_responses:
        result["raw_responses"] = raw_responses
    return result


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "model"
