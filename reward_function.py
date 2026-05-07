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
from collections.abc import Callable

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
        1.0 if p.strip().lower() == t.strip().lower() else 0.0
        for p, t in zip(predictions, targets)
    ]


# --- Answer extractors --------------------------------------------------------


@register_extractor("identity")
def identity(text: str) -> str:
    return text


def _extract_boxed(text: str) -> str | None:
    """Return contents of the first ``\\boxed{...}`` in ``text``, or None.

    Walks braces with a depth counter so nested LaTeX like ``\\boxed{\\frac{1}{2}}``
    captures ``\\frac{1}{2}`` instead of stopping at the first ``}``.
    """
    idx = text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


_DOLLAR_WRAPPER = re.compile(r"^\$+|\$+$")


def _peel_math_wrappers(text: str) -> str:
    """Strip outer ``$...$`` and ``\\boxed{...}`` wrappers from a bare answer."""
    text = _DOLLAR_WRAPPER.sub("", text).strip()
    if text.startswith(r"\boxed{"):
        boxed = _extract_boxed(text)
        if boxed is not None:
            text = boxed.strip()
    return text


@register_extractor("math_final_answer")
def math_final_answer(text: str) -> str:
    """Extract the final answer from a model output or reference solution.

    Order of attempts:
      1. ``<answer>...</answer>`` tags (the modern reasoning-model format the
         shipped prompts request). The tag content has ``$...$`` and
         ``\\boxed{...}`` wrappers peeled, so ``<answer>$\\boxed{42}$</answer>``,
         ``<answer>$42$</answer>``, and ``<answer>42</answer>`` all yield ``42``.
         When multiple ``<answer>`` tags appear, the last one wins.
      2. After a ``final answer is:`` marker (legacy dataset format):
         ``\\boxed{...}`` (balanced braces, optional ``$``/``$$`` wrap),
         ``$...$``, then plain text up to a sentence-ending period or newline
         (decimals preserved).
      3. Any ``\\boxed{...}`` anywhere in the text.
    """
    tag_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if tag_matches:
        return _peel_math_wrappers(tag_matches[-1].rstrip("."))

    marker = re.search(r"final answer is:?\s*", text, flags=re.IGNORECASE)
    if marker:
        rest = text[marker.end():]
        if rest.lstrip("$").startswith(r"\boxed{"):
            content = _extract_boxed(rest)
            if content is not None:
                return content.strip().rstrip(".")
        m = re.match(r"\$([^$\n]+)\$", rest)
        if m:
            return m.group(1).strip().rstrip(".")
        m = re.match(r"([^\n]+?)(?=\.\s|\.$|\n|$)", rest)
        if m:
            return m.group(1).strip().rstrip(".")
    content = _extract_boxed(text)
    if content is not None:
        return content.strip().rstrip(".")
    return ""
