from __future__ import annotations

from .custom_eval import check_is_correct, extract_answer


def reward(txt: str, gt: str) -> int:
    prediction = extract_answer(str(txt))
    return int(check_is_correct(prediction, str(gt)))
