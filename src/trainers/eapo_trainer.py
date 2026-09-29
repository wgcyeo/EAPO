"""Entropic Advantage Policy Optimization (EAPO).

EAPO turns a sequence-level advantage ``A_i`` into a token-level advantage
without changing its mean within the completion::

    r_{i,t} = exp(kappa * sign(A_i) * h_{i,t})
    w_{i,t} = r_{i,t} / masked_mean_t(r_{i,t})
    A^EAPO_{i,t} = A_i * w_{i,t}

``h`` is the detached, robustly normalized full-vocabulary entropy of the
rollout policy.  The trainer computes it once immediately after generation, so
the credit weights remain fixed when a generation batch is reused for multiple
optimizer steps.  The standard TRL GRPO loss then consumes the resulting
``(B, T)`` advantage tensor unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from trl import GRPOConfig, GRPOTrainer

EAPO_ENTROPY_QUANTILES = (0.10, 0.90)
EAPO_NORMALIZATION_EPS = 1e-8


@dataclass
class EAPOConfig(GRPOConfig):
    """TRL GRPO configuration with EAPO's shaping strength and decay."""

    eapo_kappa: float = field(
        default=math.log(4.0),
        metadata={"help": "Entropy shaping strength; log(4) gives at most a 4x raw credit ratio."},
    )
    eapo_kappa_decay_steps: int = field(
        default=0,
        metadata={"help": "Linearly decay kappa to zero over this many optimizer steps; <=0 disables decay."},
    )


def _linear_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    """Compute a linear quantile without ``torch.quantile``'s 2**24 input limit."""
    if values.numel() == 0:
        return values.new_tensor(float("nan"))
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}.")

    rank = quantile * (values.numel() - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    lower = values.kthvalue(lower_index + 1).values
    if lower_index == upper_index:
        return lower
    upper = values.kthvalue(upper_index + 1).values
    return torch.lerp(lower, upper, rank - lower_index)


@torch.no_grad()
def normalize_token_entropies(
    entropies: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    accelerator: Any | None = None,
    lower_quantile: float = EAPO_ENTROPY_QUANTILES[0],
    upper_quantile: float = EAPO_ENTROPY_QUANTILES[1],
    eps: float = EAPO_NORMALIZATION_EPS,
) -> torch.Tensor:
    """Robustly map valid completion-token entropies to ``[0, 1]``.

    When an Accelerate ``accelerator`` is supplied, quantiles are computed over
    the full distributed rollout batch.  Padding and prompt tokens are excluded.
    The returned normalized tensor is detached and zero at masked positions.
    """
    if entropies.shape != completion_mask.shape:
        raise ValueError(
            f"entropies and completion_mask must have the same shape, got "
            f"{tuple(entropies.shape)} and {tuple(completion_mask.shape)}."
        )
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Entropy quantiles must satisfy 0 <= lower < upper <= 1.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    mask = completion_mask.bool()
    valid_entropies = entropies.detach()[mask].float()
    if accelerator is not None:
        # Entropy is non-negative, so a negative padding sentinel cannot be
        # mistaken for a real value after the cross-process gather.
        pad_value = -1.0
        valid_entropies = accelerator.pad_across_processes(
            valid_entropies,
            dim=0,
            pad_index=pad_value,
        )
        valid_entropies = accelerator.gather(valid_entropies)
        valid_entropies = valid_entropies[valid_entropies != pad_value]

    if valid_entropies.numel() == 0:
        return torch.zeros_like(entropies, dtype=torch.float32)

    q_low = _linear_quantile(valid_entropies, lower_quantile)
    q_high = _linear_quantile(valid_entropies, upper_quantile)
    normalized = ((entropies.detach().float() - q_low) / (q_high - q_low + eps)).clamp_(0.0, 1.0)
    normalized = normalized * mask.to(normalized.dtype)
    return normalized.detach()


@torch.no_grad()
def eapo_token_advantages(
    advantages: torch.Tensor,
    normalized_entropies: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    kappa: float = math.log(4.0),
) -> torch.Tensor:
    """Allocate sequence advantages across tokens while preserving their mean.

    Positive advantages favor high-entropy tokens; negative advantages favor
    low-entropy tokens. Returns detached advantages of shape ``(B, T)``, with
    zeros at masked positions.
    """
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError(f"kappa must be finite and non-negative, got {kappa}.")
    if normalized_entropies.shape != completion_mask.shape:
        raise ValueError(
            "normalized_entropies and completion_mask must have the same shape, "
            f"got {tuple(normalized_entropies.shape)} and {tuple(completion_mask.shape)}."
        )
    if advantages.dim() == 2 and advantages.size(1) == 1:
        advantages = advantages.squeeze(1)
    if advantages.dim() != 1 or advantages.size(0) != normalized_entropies.size(0):
        raise ValueError(
            "advantages must have shape (B,) or (B, 1) matching the entropy batch; "
            f"got {tuple(advantages.shape)}."
        )

    mask = completion_mask.to(normalized_entropies.dtype)
    entropy = normalized_entropies.detach().float().clamp(0.0, 1.0)
    sequence_advantages = advantages.detach().float().unsqueeze(1)
    allocation_sign = torch.sign(sequence_advantages)
    log_raw_weights = kappa * allocation_sign * entropy
    # Subtracting a per-completion constant cancels during mean normalization
    # and prevents overflow for unusually large experimental kappa values.
    log_raw_weights = log_raw_weights - log_raw_weights.max(dim=-1, keepdim=True).values
    raw_weights = torch.exp(log_raw_weights)

    token_counts = mask.sum(dim=-1, keepdim=True)
    mean_weights = (raw_weights * mask).sum(dim=-1, keepdim=True) / token_counts.clamp_min(1.0)
    weights = raw_weights / mean_weights.clamp_min(EAPO_NORMALIZATION_EPS)
    weights = weights * mask
    token_advantages = sequence_advantages * weights
    return token_advantages.detach()


class EAPOTrainer(GRPOTrainer):
    """GRPOTrainer with EAPO token-level credit allocation."""

    _tag_names = ["trl", "grpo", "eapo"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eapo_kappa = float(getattr(self.args, "eapo_kappa", math.log(4.0)))
        if not math.isfinite(self.eapo_kappa) or self.eapo_kappa < 0.0:
            raise ValueError(f"eapo_kappa must be finite and non-negative, got {self.eapo_kappa}.")
        self.eapo_kappa_decay_steps = int(getattr(self.args, "eapo_kappa_decay_steps", 0))
        if self.use_liger_kernel:
            raise NotImplementedError("EAPOTrainer does not support the fused Liger GRPO loss.")

        self._eapo_capture_entropy = False
        self._eapo_captured_entropies: torch.Tensor | None = None

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        **kwargs,
    ):
        """Reuse TRL's old-policy scoring forward to capture rollout entropy."""
        capture = (
            getattr(self, "_eapo_capture_entropy", False)
            and getattr(self, "_eapo_captured_entropies", None) is None
            and model is getattr(self, "model", None)
        )
        logps, entropies, aux_loss = super()._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy or capture,
            **kwargs,
        )
        if capture:
            if entropies is None:
                raise RuntimeError("TRL did not return the entropies requested by EAPO.")
            self._eapo_captured_entropies = entropies.detach()
            self._eapo_capture_entropy = False
        return logps, entropies, aux_loss

    @torch.no_grad()
    def _completion_entropies(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fallback entropy forward for configurations that do not score the old policy."""
        completion_ids = output["completion_ids"]
        input_ids = torch.cat([output["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
        _, entropies, _ = self._get_per_token_logps_and_entropies(
            self.model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
            batch_size=self.args.per_device_train_batch_size,
            compute_entropy=True,
            pixel_values=output.get("pixel_values"),
            image_grid_thw=output.get("image_grid_thw"),
            num_images=output.get("num_images"),
            pixel_attention_mask=output.get("pixel_attention_mask"),
            spatial_shapes=output.get("spatial_shapes"),
            num_tiles=output.get("num_tiles"),
            image_sizes=output.get("image_sizes"),
            token_type_ids=output.get("token_type_ids"),
            mm_token_type_ids=output.get("mm_token_type_ids"),
            image_position_ids=output.get("image_position_ids"),
        )
        if entropies is None:
            raise RuntimeError("TRL did not return the entropies requested by EAPO.")
        return entropies.detach()

    def _old_policy_forward_is_used(self) -> bool:
        generate_every = self.args.steps_per_generation * self.num_iterations
        return self.args.gradient_accumulation_steps % generate_every != 0 or (
            self.use_vllm and self.vllm_importance_sampling_correction
        )

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        # With the repository's default vLLM correction, TRL already performs
        # an old-policy forward after generation.  Ask that same forward for
        # entropy instead of doing another full-vocabulary model pass.
        self._eapo_captured_entropies = None
        self._eapo_capture_entropy = self._old_policy_forward_is_used()
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            self._eapo_capture_entropy = False

        entropies = self._eapo_captured_entropies
        self._eapo_captured_entropies = None
        if entropies is None:
            entropies = self._completion_entropies(output)

        completion_mask = output["completion_mask"]
        normalized_entropies = normalize_token_entropies(
            entropies,
            completion_mask,
            accelerator=self.accelerator,
        )
        # Evaluate the schedule once per rollout batch. Restored global_step
        # resumes the schedule, and cached advantages stay fixed across reuse.
        kappa = self.eapo_kappa
        if self.eapo_kappa_decay_steps > 0:
            progress = min(float(self.state.global_step) / float(self.eapo_kappa_decay_steps), 1.0)
            kappa *= 1.0 - progress
        token_advantages = eapo_token_advantages(
            output["advantages"],
            normalized_entropies,
            completion_mask,
            kappa=kappa,
        )
        output["advantages"] = token_advantages.to(output["advantages"].dtype)
        return output
