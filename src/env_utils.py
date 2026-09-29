from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


def load_repo_dotenv() -> None:
    """Load the repository's .env, overriding matching environment variables."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing dotenv file at {ENV_PATH}. Copy .env.example to .env and fill it in.")
    load_dotenv(ENV_PATH, override=True)
    logger.info("Loaded environment from {}", ENV_PATH)


def _optional_env(name: str) -> str | None:
    return os.getenv(name, "").strip() or None


def init_wandb_run(
    *,
    args: object,
    report_to: list[str],
    algorithm: str,
    output_dir: str,
    train_jsonl: object,
    is_main_process: bool = True,
):
    if not is_main_process or "wandb" not in report_to:
        return None

    import wandb

    config = vars(args).copy()
    config.update(
        {
            "algorithm": algorithm,
            "output_dir": output_dir,
            "train_jsonl": str(train_jsonl),
        }
    )
    config["chat_template_kwargs"] = {}
    project = _optional_env("WANDB_PROJECT")
    if project is None:
        raise ValueError("WANDB_PROJECT must be set when W&B logging is enabled.")

    init_kwargs = {
        "config": config,
        "entity": _optional_env("WANDB_ENTITY"),
        "project": project,
        "name": getattr(args, "run_name", None),
        "group": _optional_env("WANDB_GROUP"),
        "job_type": _optional_env("WANDB_JOB_TYPE"),
    }
    tags = _optional_env("WANDB_TAGS")
    if tags:
        init_kwargs["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
    run_id = getattr(args, "wandb_run_id", None)
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "must"

    return wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})
