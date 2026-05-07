"""Reward functions and answer extractors for SCoRe training.

`train.py` looks both up by string name from the YAML config:

    extract = ANSWER_EXTRACTORS[cfg.reward.answer_extractor]
    reward  = REWARD_FNS[cfg.reward.fn]

Composition in `train.py`:

    extracted_pred   = [extract(p) for p in predictions]
    extracted_target = [extract(t) for t in targets]
    rewards = reward(extracted_pred, extracted_target)  # list[float] in [0, 1]

To support a new task, add one extractor and (optionally) one reward
function below, decorate, then reference the names from your YAML.
"""

from __future__ import annotations

import re
from typing import Callable

RewardFn = Callable[[list[str], list[str]], list[float]]
AnswerExtractor = Callable[[str], str]

REWARD_FNS: dict[str, RewardFn] = {}
ANSWER_EXTRACTORS: dict[str, AnswerExtractor] = {}


def register_reward(name: str) -> Callable[[RewardFn], RewardFn]:
    def decorator(fn: RewardFn) -> RewardFn:
        if name in REWARD_FNS:
            raise ValueError(f"Reward function '{name}' already registered")
        REWARD_FNS[name] = fn
        return fn

    return decorator


def register_extractor(name: str) -> Callable[[AnswerExtractor], AnswerExtractor]:
    def decorator(fn: AnswerExtractor) -> AnswerExtractor:
        if name in ANSWER_EXTRACTORS:
            raise ValueError(f"Answer extractor '{name}' already registered")
        ANSWER_EXTRACTORS[name] = fn
        return fn

    return decorator


# --- Reward functions ---------------------------------------------------------


@register_reward("exact_match")
def exact_match(predictions: list[str], targets: list[str]) -> list[float]:
    """1.0 if normalized strings match else 0.0. Whitespace-stripped, lowercased."""
    return [
        1.0 if _normalize(p) == _normalize(t) else 0.0
        for p, t in zip(predictions, targets)
    ]


def _normalize(s: str) -> str:
    return s.strip().lower()


# --- Answer extractors --------------------------------------------------------


@register_extractor("identity")
def identity(text: str) -> str:
    return text


# Patterns are tried in order; first match wins. Falls back to any \boxed{...}
# anywhere in the text, then to the empty string.
_FINAL_ANSWER_PATTERNS = (
    r"final answer is:?\s*\$([^$]+)\$",
    r"final answer is:?\s*(?:\$\$)?\\boxed\{([^}]+)\}",
    r"final answer is:?\s*([^\n.]+)",
)


@register_extractor("math_final_answer")
def math_final_answer(text: str) -> str:
    """Extract the final answer from math-style solutions.

    Handles common forms: 'final answer is: 42', 'final answer is: $y = 2x$',
    'final answer is: \\boxed{42}', or a bare \\boxed{X} anywhere in the text.
    """
    lower = text.lower()
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = re.search(pattern, lower, flags=re.DOTALL)
        if match:
            return match.group(1).strip().rstrip(".")
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    return ""
