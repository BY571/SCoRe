"""Diagnostic run: profile SCoRe prompt and generation lengths on N sample questions.

Run BEFORE training to validate `max_new_tokens_*`, `max_seq_length`, and
`max_prompt_length_attempt{1,2}` against the model's actual output distribution
on the configured task — and to spot-check rewards / extraction.

Usage:
    python profile_run.py --config configs/math.yaml [--n 10] [--seed 42]

Outputs:
- Stdout: per-question metrics, full-dataset prompt-length distribution,
  summary table, and concrete YAML-budget recommendations.
- Markdown file `profile_<run_name>.md` with the full text of every question
  and both attempts so the actual model output can be inspected later.

Generation uses `cfg.train.generation_temperature` (sampling matches training).
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
from unsloth import FastLanguageModel

import reward_function as rf
from train import (
    get_eos_mask,
    load_and_prepare_data,
    load_config,
    load_model_and_tokenizer,
    thinking_to_history,
)


def percentiles(values: list[int]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def fmt_pct(p: dict[str, float]) -> str:
    return (
        f"n={p['n']:<5}  min={p['min']:<6}  p50={p['p50']:<6}  "
        f"p95={p['p95']:<6}  p99={p['p99']:<6}  max={p['max']:<6}  "
        f"mean={p['mean']:<.1f}"
    )


def count_think_tokens(text: str, tokenizer) -> tuple[int, int]:
    """Return ``(think_tokens, post_think_tokens)``.

    Handles three cases:
      - Closed ``<think>...</think>``: split at the closing tag.
      - Open-only ``<think>...`` (budget hit before close): everything after
        ``<think>`` is reasoning, post-think is empty.
      - No ``<think>`` tag: post-think is the whole text.
    """
    closed = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if closed:
        think_text = closed.group(1)
        post_text = text[closed.end():].strip()
    elif "<think>" in text:
        think_text = text.split("<think>", 1)[1]
        post_text = ""
    else:
        think_text, post_text = "", text
    enc = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    return enc(think_text), enc(post_text)


def count_generated(tokens: torch.Tensor, tokenizer) -> int:
    """Tokens up to (and including) the first EOS, capped at the tensor length."""
    mask = get_eos_mask(tokens.unsqueeze(0), tokenizer)
    return int(mask.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--n", type=int, default=10, help="Number of questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(f"profile_{cfg.run_name}.md")

    print(f"=== Profile run for {cfg.run_name} ===")
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Sampling {args.n} questions (seed={args.seed})\n")

    print("--- Loading model + tokenizer ---")
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    print("--- Loading dataset (this also pre-builds chat templates) ---")
    train_dataset, _ = load_and_prepare_data(cfg.dataset, cfg.prompts, tokenizer)

    extractor = rf.ANSWER_EXTRACTORS[cfg.reward.answer_extractor]
    reward_fn = rf.REWARD_FNS[cfg.reward.fn]

    # --- Phase 1: prompt-length distribution across the full train set -------
    print("\n=== Phase 1: attempt-1 prompt token distribution across full train set ===")
    prompt_lens = [
        len(tokenizer.encode(ex["text"], add_special_tokens=False))
        for ex in train_dataset
    ]
    p1 = percentiles(prompt_lens)
    print(f"  prompt tokens: {fmt_pct(p1)}")
    cfg_a1 = cfg.train.max_prompt_length_attempt1
    if p1["max"] > cfg_a1:
        print(f"  WARNING: max ({p1['max']}) > max_prompt_length_attempt1 ({cfg_a1}) — some prompts will be truncated.")
    elif p1["p99"] < cfg_a1 // 2:
        print(f"  Note: p99 ({p1['p99']}) is well under {cfg_a1}; could reduce to ~{int(p1['p99'] * 1.3)}.")
    else:
        print(f"  OK: p99 ({p1['p99']}) fits within max_prompt_length_attempt1 ({cfg_a1}).")

    # --- Phase 2: rollout N random examples ---------------------------------
    print(f"\n=== Phase 2: rollout {args.n} random examples ===")
    indices = random.sample(range(len(train_dataset)), args.n)

    a1_total: list[int] = []
    a1_think: list[int] = []
    a1_post: list[int] = []
    a2_total: list[int] = []
    a2_think: list[int] = []
    a2_post: list[int] = []
    a2_prompt: list[int] = []
    rewards1: list[float] = []
    rewards2: list[float] = []
    a1_budget_hits = 0
    a2_budget_hits = 0
    a1_extract_fails = 0
    a2_extract_fails = 0

    md_lines: list[str] = [
        f"# SCoRe profile run — `{cfg.run_name}`",
        "",
        f"- Config: `{args.config}`",
        f"- Model: `{cfg.model.name}`",
        f"- Dataset: `{cfg.dataset.name}`",
        f"- N: {args.n} (seed={args.seed})",
        "",
        "## Phase 1: dataset prompt-length distribution",
        "",
        f"`{fmt_pct(p1)}`",
        "",
        "## Phase 2: per-question rollouts",
        "",
    ]

    FastLanguageModel.for_inference(model)
    do_sample = cfg.train.generation_temperature > 0
    gen_kwargs = {
        "do_sample": do_sample,
        "temperature": cfg.train.generation_temperature if do_sample else 1.0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for i, idx in enumerate(indices, 1):
        ex = train_dataset[idx]
        question_text = ex[cfg.dataset.input_field]
        target = ex["target"]
        target_extracted = extractor(target)

        x1 = tokenizer(ex["text"], padding=False, return_tensors="pt").to(model.device)
        x1_len = x1.input_ids.shape[1]

        action1 = model.generate(
            x1.input_ids,
            attention_mask=x1.attention_mask,
            max_new_tokens=cfg.train.max_new_tokens_attempt1,
            **gen_kwargs,
        )
        a1_new = action1[:, x1_len:][0]
        a1_text = tokenizer.decode(a1_new, skip_special_tokens=True)
        n1 = count_generated(a1_new, tokenizer)
        n1_think, n1_post = count_think_tokens(a1_text, tokenizer)
        a1_total.append(n1)
        a1_think.append(n1_think)
        a1_post.append(n1_post)
        if n1 >= cfg.train.max_new_tokens_attempt1:
            a1_budget_hits += 1
        a1_extracted = extractor(a1_text)
        if not a1_extracted:
            a1_extract_fails += 1
        r1 = reward_fn([a1_text], [target], extractor)[0]
        rewards1.append(r1)

        # Build attempt 2 — convert <think> the same way train.py does so the
        # chat template doesn't strip it.
        messages = list(ex["messages"])
        messages.append({"role": "assistant", "content": thinking_to_history(a1_text)})
        messages.append({"role": "user", "content": cfg.prompts.self_correction})
        x2_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        x2 = tokenizer(x2_text, padding=False, return_tensors="pt").to(model.device)
        x2_len = x2.input_ids.shape[1]
        a2_prompt.append(x2_len)

        action2 = model.generate(
            x2.input_ids,
            attention_mask=x2.attention_mask,
            max_new_tokens=cfg.train.max_new_tokens_attempt2,
            **gen_kwargs,
        )
        a2_new = action2[:, x2_len:][0]
        a2_text = tokenizer.decode(a2_new, skip_special_tokens=True)
        n2 = count_generated(a2_new, tokenizer)
        n2_think, n2_post = count_think_tokens(a2_text, tokenizer)
        a2_total.append(n2)
        a2_think.append(n2_think)
        a2_post.append(n2_post)
        if n2 >= cfg.train.max_new_tokens_attempt2:
            a2_budget_hits += 1
        a2_extracted = extractor(a2_text)
        if not a2_extracted:
            a2_extract_fails += 1
        r2 = reward_fn([a2_text], [target], extractor)[0]
        rewards2.append(r2)

        print(
            f"  Q{i:>2} idx={idx:<5}  "
            f"x1={x1_len:<4}  a1=[total={n1}, think={n1_think}, post={n1_post}]  "
            f"x2={x2_len:<5}  a2=[total={n2}, think={n2_think}, post={n2_post}]  "
            f"r1={r1:.2f} r2={r2:.2f}  "
            f"e1={a1_extracted!r:<20}  e2={a2_extracted!r:<20}  target={target_extracted!r}"
        )

        md_lines += [
            f"### Q{i} (idx={idx})",
            "",
            f"- **Target answer:** `{target_extracted}`",
            f"- **Attempt 1:** prompt={x1_len} tokens, generated={n1} (think={n1_think}, post={n1_post}), extracted={a1_extracted!r}, reward={r1:.2f}",
            f"- **Attempt 2:** prompt={x2_len} tokens, generated={n2} (think={n2_think}, post={n2_post}), extracted={a2_extracted!r}, reward={r2:.2f}",
            "",
            "**Question:**",
            "",
            "```",
            question_text,
            "```",
            "",
            "**Attempt 1 output:**",
            "",
            "```",
            a1_text,
            "```",
            "",
            "**Attempt 2 output:**",
            "",
            "```",
            a2_text,
            "```",
            "",
            "---",
            "",
        ]

    # --- Summary ------------------------------------------------------------
    print(f"\n=== Summary across {args.n} examples ===")
    summary_rows = [
        ("attempt-1 generated total", a1_total),
        ("attempt-1 think tokens", a1_think),
        ("attempt-1 post-think tokens", a1_post),
        ("attempt-2 prompt tokens", a2_prompt),
        ("attempt-2 generated total", a2_total),
        ("attempt-2 think tokens", a2_think),
        ("attempt-2 post-think tokens", a2_post),
    ]
    print(f"{'metric':<30}  {'min':>5} {'p50':>5} {'p95':>5} {'p99':>5} {'max':>5} {'mean':>7}")
    md_lines += [
        "## Summary",
        "",
        "| metric | min | p50 | p95 | p99 | max | mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, vals in summary_rows:
        p = percentiles(vals)
        print(f"{name:<30}  {p['min']:>5} {p['p50']:>5} {p['p95']:>5} {p['p99']:>5} {p['max']:>5} {p['mean']:>7.1f}")
        md_lines.append(
            f"| {name} | {p['min']} | {p['p50']} | {p['p95']} | {p['p99']} | {p['max']} | {p['mean']:.1f} |"
        )
    md_lines.append("")

    headline = [
        ("Reward attempt 1", f"{np.mean(rewards1):.2f}"),
        ("Reward attempt 2", f"{np.mean(rewards2):.2f}"),
        ("Delta (a2 - a1)", f"{np.mean(rewards2) - np.mean(rewards1):+.2f}"),
        ("Attempt 1 hit max_new_tokens", f"{a1_budget_hits}/{args.n}"),
        ("Attempt 2 hit max_new_tokens", f"{a2_budget_hits}/{args.n}"),
        ("Attempt 1 extraction failures", f"{a1_extract_fails}/{args.n}"),
        ("Attempt 2 extraction failures", f"{a2_extract_fails}/{args.n}"),
    ]
    print()
    for label, value in headline:
        print(f"  {label}: {value}")
    md_lines += [f"- {label}: **{value}**" for label, value in headline]
    md_lines.append("")

    # --- Recommendations ---------------------------------------------------
    print("\n=== Recommendations ===")
    md_lines += ["## Recommendations", ""]

    def gen_budget_rec(name: str, current: int, observed_max: int, hits: int) -> str:
        if hits > 0:
            return f"{name}: {current} — INCREASE (hit {hits}/{args.n} times)."
        if observed_max < current * 0.5 and args.n >= 5:
            return f"{name}: {current} — could reduce to ~{max(int(observed_max * 1.3), 64)} (observed max {observed_max} over n={args.n})."
        return f"{name}: {current} — looks right (observed max {observed_max})."

    def prompt_budget_rec(name: str, current: int, observed_max: int) -> str:
        if observed_max > current:
            return f"{name}: {current} — INCREASE (observed prompt of {observed_max} would be truncated)."
        if observed_max < current * 0.5:
            return f"{name}: {current} — could reduce to ~{int(observed_max * 1.3)} (observed max {observed_max})."
        return f"{name}: {current} — looks right (observed max {observed_max})."

    rec_lines = [
        gen_budget_rec("max_new_tokens_attempt1", cfg.train.max_new_tokens_attempt1, max(a1_total), a1_budget_hits),
        gen_budget_rec("max_new_tokens_attempt2", cfg.train.max_new_tokens_attempt2, max(a2_total), a2_budget_hits),
        prompt_budget_rec("max_prompt_length_attempt1", cfg.train.max_prompt_length_attempt1, p1["max"]),
        prompt_budget_rec("max_prompt_length_attempt2", cfg.train.max_prompt_length_attempt2, max(a2_prompt)),
    ]
    for line in rec_lines:
        print(f"  {line}")
        md_lines.append(f"- {line}")

    if a1_extract_fails > 0 or a2_extract_fails > 0:
        warn = (
            f"WARNING: {a1_extract_fails + a2_extract_fails} extraction failures — "
            "the model's answer format isn't matching the regex. Tighten the "
            "system prompt or extend the extractor in reward_function.py."
        )
        print(f"  {warn}")
        md_lines.append(f"- {warn}")

    out_path.write_text("\n".join(md_lines))
    print(f"\nFull per-question text written to: {out_path}")


if __name__ == "__main__":
    main()
