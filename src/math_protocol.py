from __future__ import annotations

import re
from typing import Any

BOXED_PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
TRAINING_SCORE_TAIL_CHARS = 100

_DAPO_LEADING_INSTRUCTION = re.compile(
    r"^Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: \$Answer \(without quotes\) "
    r"where \$Answer is the answer to the problem\.\s*\n*"
)
_DAPO_TRAILING_INSTRUCTION = re.compile(
    r'\s*\n*Remember to put your answer on its own line after "Answer:"\.?\s*$'
)


def strip_legacy_dapo_instructions(text: str) -> str:
    """Recover the bare problem from DAPO's legacy ``Answer:`` prompt."""
    text = _DAPO_LEADING_INSTRUCTION.sub("", str(text))
    text = _DAPO_TRAILING_INSTRUCTION.sub("", text)
    text = text.removesuffix(BOXED_PROMPT_SUFFIX)
    return text.strip()


def boxed_math_prompt(problem: str) -> str:
    """Append the shared boxed-answer instruction to the problem."""
    return strip_legacy_dapo_instructions(problem) + BOXED_PROMPT_SUFFIX


def canonicalize_math_messages(prompt: Any) -> list[dict[str, str]]:
    """Convert a DAPO prompt into one user message with the boxed-answer instruction."""
    if isinstance(prompt, (list, tuple)):
        messages = [message for message in prompt if isinstance(message, dict)]
        user_messages = [
            message for message in messages if message.get("role") == "user"
        ]
        selected = (
            user_messages[0] if user_messages else (messages[0] if messages else {})
        )
        content = selected.get("content", "")
    else:
        content = prompt
    return [{"content": boxed_math_prompt(str(content)), "role": "user"}]


def last_boxed_only_string(text: str, *, tail_chars: int | None = None) -> str | None:
    """Extract the last complete ``\\boxed{...}``, including nested braces."""
    text = str(text)
    if tail_chars is not None:
        text = text[-tail_chars:]

    start = text.rfind(r"\boxed{")
    if start < 0:
        return None

    open_braces = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            open_braces += 1
        elif text[index] == "}":
            open_braces -= 1
            if open_braces == 0:
                return text[start : index + 1]
    return ""


def remove_boxed(boxed: str | None) -> str | None:
    """Remove one outer ``\\boxed{}``; preserve missing versus malformed boxes."""
    if boxed is None:
        return None
    prefix = r"\boxed{"
    if boxed.startswith(prefix) and boxed.endswith("}"):
        return boxed[len(prefix) : -1]
    return ""


def extract_boxed_answer(text: str, *, tail_chars: int | None = None) -> str | None:
    return remove_boxed(last_boxed_only_string(text, tail_chars=tail_chars))


def boxed_answer_is_correct(
    completion: str,
    ground_truth: str,
    *,
    tail_chars: int | None = None,
) -> bool:
    """Extract the boxed answer, then check exact or Math-Verify equivalence."""
    prediction = extract_boxed_answer(completion, tail_chars=tail_chars)
    if prediction is None or prediction == "":
        return False

    ground_truth = str(ground_truth)
    if prediction == ground_truth:
        return True

    try:
        from math_verify import LatexExtractionConfig, parse, verify

        # Parse each complete answer as LaTeX; bare-expression extraction can
        # miss commands such as \sqrt or select only a number from a formula.
        extraction_config = [LatexExtractionConfig()]
        gold_parsed = parse(
            rf"\boxed{{{ground_truth}}}",
            extraction_config=extraction_config,
            fallback_mode="no_fallback",
        )
        prediction_parsed = parse(
            rf"\boxed{{{prediction}}}",
            extraction_config=extraction_config,
            fallback_mode="no_fallback",
        )
        return bool(verify(gold_parsed, prediction_parsed))
    except Exception:  # noqa: BLE001 - verifier parse/timeouts must produce reward zero.
        return False


def math_answer_is_correct(completion: str, ground_truth: str) -> bool:
    """Score the boxed answer in the final 100 characters of a completion."""
    return boxed_answer_is_correct(
        completion,
        ground_truth,
        tail_chars=TRAINING_SCORE_TAIL_CHARS,
    )
