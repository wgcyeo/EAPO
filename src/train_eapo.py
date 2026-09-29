from __future__ import annotations

import argparse
from pathlib import Path

from transformers import set_seed

from .env_utils import init_wandb_run, load_repo_dotenv
from .trainers import EAPOConfig, EAPOTrainer
from .training_args_default import (
    EAPO_DEFAULTS,
    add_common_training_args,
    auto_run_name,
    build_training_args,
    load_model_for_training,
    resolve_learning_rate,
)
from .training_utils import build_reward_funcs, load_or_create_training_dataset, load_training_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EAPO on a sampled DAPO-Math-17k subset.")
    add_common_training_args(parser, EAPO_DEFAULTS)
    parser.add_argument(
        "--eapo-kappa",
        type=float,
        default=EAPO_DEFAULTS["eapo_kappa"],
        help="Entropy shaping strength (default: log(4), for at most a 4x raw credit ratio).",
    )
    parser.add_argument(
        "--eapo-kappa-decay-steps",
        type=int,
        default=EAPO_DEFAULTS["eapo_kappa_decay_steps"],
        help="Linearly decay kappa over optimizer steps (default: 0, disabled).",
    )
    return parser.parse_args()


def main() -> None:
    load_repo_dotenv()
    args = parse_args()
    args.learning_rate = resolve_learning_rate(args, EAPO_DEFAULTS)
    if args.run_name is None:
        args.run_name = auto_run_name("eapo", args) + f"_kappa{args.eapo_kappa:g}"
        if args.eapo_kappa_decay_steps > 0:
            args.run_name += f"_decay{args.eapo_kappa_decay_steps}"
    output_dir = str(Path(args.output_root) / args.run_name)
    train_dataset, train_jsonl = load_or_create_training_dataset(
        train_jsonl=args.train_jsonl,
        data_cache_dir=args.data_cache_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_prompts=args.num_prompts,
        seed=args.data_seed,
    )

    tokenizer = load_training_tokenizer(args.model_name)

    training_args = build_training_args(
        EAPOConfig,
        args,
        output_dir,
        extra_kwargs={
            "eapo_kappa": args.eapo_kappa,
            "eapo_kappa_decay_steps": args.eapo_kappa_decay_steps,
        },
    )
    set_seed(training_args.seed)
    model = load_model_for_training(args)
    trainer = EAPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=build_reward_funcs(
            max_completion_length=args.max_completion_length,
            soft_overlong_start=args.soft_overlong_start,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    wandb_run = init_wandb_run(
        args=args,
        report_to=training_args.report_to,
        algorithm="eapo",
        output_dir=output_dir,
        train_jsonl=train_jsonl,
        is_main_process=trainer.accelerator.is_main_process,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    if trainer.accelerator.is_main_process:
        print(f"Saved sampled training set to {train_jsonl}")
        print(f"Saved model/checkpoints under {output_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
