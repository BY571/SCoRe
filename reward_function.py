"""Reward functions and answer extractors for SCoRe training.

`train.py` looks both up by string name from the YAML config:

    extract = ANSWER_EXTRACTORS[cfg.reward.answer_extractor]
    reward  = REWARD_FNS[cfg.reward.fn]

Reward functions receive the **full** prediction and target text plus the
configured extractor, and return per-example floats in [0, 1]:

    rewards = reward(predictions, targets, extract)

This signature lets format-aware rewards (e.g. `format_and_match`) inspect
tag structure on the prediction while still using the extractor for the
answer-correctness check. Format-only rewards apply checks only to the
prediction; targets typically come from a dataset and may not carry the
same format conventions.

To support a new task, add one extractor and (optionally) one reward
function below, decorate, then reference the names from your YAML.
"""

from __future__ import annotations

import re
from collections.abc import Callable

AnswerExtractor = Callable[[str], str]
RewardFn = Callable[[list[str], list[str], AnswerExtractor], list[float]]

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


def _normalize(s: str) -> str:
    return s.strip().lower()


@register_reward("exact_match")
def exact_match(
    predictions: list[str], targets: list[str], extractor: AnswerExtractor
) -> list[float]:
    """1.0 if extracted, normalized strings match else 0.0."""
    return [
        1.0 if _normalize(extractor(p)) == _normalize(extractor(t)) else 0.0
        for p, t in zip(predictions, targets)
    ]


_THINK_PAIR = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ANSWER_PAIR = re.compile(r"<answer>.*?</answer>", flags=re.DOTALL)


def _has_balanced_think(text: str) -> bool:
    """Exactly one well-formed ``<think>...</think>`` pair somewhere in the text."""
    return len(_THINK_PAIR.findall(text)) >= 1


def _has_one_answer_tag(text: str) -> bool:
    """Exactly one well-formed ``<answer>...</answer>`` pair (no more, no less)."""
    return len(_ANSWER_PAIR.findall(text)) == 1


@register_reward("format_and_match")
def format_and_match(
    predictions: list[str], targets: list[str], extractor: AnswerExtractor
) -> list[float]:
    """Compound reward: loose tag format compliance + answer correctness.

    Per prediction:
      - 0.25 for a balanced ``<think>...</think>`` pair (reasoning closed).
      - 0.25 for exactly one balanced ``<answer>...</answer>`` pair.
      - 0.50 for the extracted answer matching the target after normalization.

    "Loose": the tags only need to appear *somewhere* — arbitrary prose is
    tolerated before, between, and after the two blocks. For a version that
    requires the whole output to be exactly the two blocks, see
    ``strict_format_and_match``.

    Returns floats in [0.0, 1.0]. Format checks apply to predictions only;
    targets may not follow the same format conventions (e.g. dataset
    solutions in legacy ``"final answer is:"`` form).
    """
    rewards: list[float] = []
    for pred, target in zip(predictions, targets):
        r = 0.0
        if _has_balanced_think(pred):
            r += 0.25
        if _has_one_answer_tag(pred):
            r += 0.25
        if _normalize(extractor(pred)) == _normalize(extractor(target)):
            r += 0.5
        rewards.append(r)
    return rewards


_STRICT_FORMAT = re.compile(
    r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",
    flags=re.DOTALL,
)


def _has_strict_format(text: str) -> bool:
    """The ENTIRE output is exactly ``<think>...</think><answer>...</answer>``.

    Only whitespace is permitted before ``<think>``, between ``</think>`` and
    ``<answer>``, and after ``</answer>`` — no prose anywhere outside the two
    blocks, and exactly one of each tag.

    The ``.count()`` guards are not redundant with the regex: ``.*?`` with
    ``DOTALL`` backtracks across a second ``<think>`` block to reach a later
    ``</think>``, so a pure regex would accept duplicated blocks. Counting
    tags first rejects those; the regex then enforces ordering + no-prose.
    """
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return False
    return _STRICT_FORMAT.match(text) is not None


@register_reward("strict_format_and_match")
def strict_format_and_match(
    predictions: list[str], targets: list[str], extractor: AnswerExtractor
) -> list[float]:
    """Stricter compound reward: whole-output format compliance + answer correctness.

    Per prediction:
      - 0.50 for STRICT format: the entire output is exactly one
        ``<think>...</think>`` block immediately followed by exactly one
        ``<answer>...</answer>`` block, with only whitespace before, between,
        and after. All-or-nothing.
      - 0.50 for the extracted answer matching the target after normalization.

    Contrast with ``format_and_match``, which splits the format reward into
    two loose 0.25 checks (tags exist *somewhere*) and tolerates arbitrary
    prose around and between the blocks. Select this via ``reward.fn`` in the
    YAML when you want the model to learn the exact output shape — no prose
    leaking between ``</think>`` and ``<answer>``, nothing after ``</answer>``.

    Returns floats in [0.0, 1.0].
    """
    rewards: list[float] = []
    for pred, target in zip(predictions, targets):
        r = 0.0
        if _has_strict_format(pred):
            r += 0.5
        if _normalize(extractor(pred)) == _normalize(extractor(target)):
            r += 0.5
        rewards.append(r)
    return rewards


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
_DISPLAY_BRACKETS = re.compile(r"^\\\[(.*)\\\]$", flags=re.DOTALL)


def _peel_math_wrappers(text: str) -> str:
    """Strip outer ``$...$``, ``\\[...\\]`` display brackets, and ``\\boxed{...}`` wrappers.

    Repeatedly peels until no wrapper remains, so combinations like
    ``$\\[\\boxed{42}\\]$`` reduce to ``42``.
    """
    prev = None
    while text != prev:
        prev = text
        text = _DOLLAR_WRAPPER.sub("", text).strip()
        m = _DISPLAY_BRACKETS.match(text)
        if m:
            text = m.group(1).strip()
            continue
        if text.startswith(r"\boxed{"):
            boxed = _extract_boxed(text)
            if boxed is not None:
                text = boxed.strip()
    return text


_GSM8K_HASH = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(s: str) -> str:
    return s.strip().rstrip(".").replace(",", "").strip()


@register_extractor("gsm8k_hash")
def gsm8k_hash(text: str) -> str:
    """Extract the final numeric answer from a GSM8K example or model output.

    Handles both input shapes the training loop encounters:
      - Model predictions: ``<answer>...</answer>`` (the format the system
        prompt asks for). Last tag wins, contents normalized to a bare number.
      - Dataset targets: GSM8K's native ``#### N`` marker at the end of the
        reasoning trace. Last marker wins so intermediate ``<<48/2=24>>``
        annotations don't trip it.
      - Lenient last-resort: the final numeric literal in the text. Lets the
        format reward differ from the correctness reward — a model that gets
        the number right without using ``<answer>`` tags still scores 0.5.

    Returns the normalized number as a string (commas stripped, sign and
    decimals preserved), or ``""`` if nothing parseable was found.
    """
    tag_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if tag_matches:
        return _normalize_number(tag_matches[-1])

    hash_matches = _GSM8K_HASH.findall(text)
    if hash_matches:
        return _normalize_number(hash_matches[-1])

    nums = _NUMBER.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return ""


@register_extractor("math_final_answer")
def math_final_answer(text: str) -> str:
    """Extract the final answer from a model output or reference solution.

    Order of attempts:
      1. ``<answer>...</answer>`` tags (the modern reasoning-model format the
         shipped prompts request). The tag content has wrappers peeled, so
         ``<answer>$\\boxed{42}$</answer>``, ``<answer>$42$</answer>``, and
         ``<answer>42</answer>`` all yield ``42``. When multiple ``<answer>``
         tags appear, the last one wins.
      2. Any ``\\boxed{...}`` anywhere in the text. Preferred over the
         "final answer is:" marker because the dataset's relabel script
         doubly-wrapped many targets ("the final answer is $the final answer
         is $\\boxed{X}$.$"), and ``\\boxed{X}`` is the unambiguous source of
         truth in math problems.
      3. After a ``final answer is:`` marker (last-resort fallback for
         solutions without a ``\\boxed{...}``): ``$...$`` then plain text up
         to a sentence-ending period or newline (decimals preserved).
    """
    tag_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.DOTALL,
    )
    if tag_matches:
        return _peel_math_wrappers(tag_matches[-1].rstrip("."))

    boxed = _extract_boxed(text)
    if boxed is not None:
        return _peel_math_wrappers(boxed.rstrip("."))

    marker = re.search(r"final answer is:?\s*", text, flags=re.IGNORECASE)
    if marker:
        rest = text[marker.end():]
        m = re.match(r"\$([^$\n]+)\$", rest)
        if m:
            return _peel_math_wrappers(m.group(1).rstrip("."))
        m = re.match(r"([^\n]+?)(?=\.\s|\.$|\n|$)", rest)
        if m:
            return _peel_math_wrappers(m.group(1).rstrip("."))
    return ""
