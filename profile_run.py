"""Diagnostic run: profile SCoRe prompt and generation lengths on N sample questions.

Use this BEFORE a full training run to validate that `max_new_tokens_*`,
`max_seq_length`, and `max_prompt_length_attempt{1,2}` in your YAML are
appropriately sized — and to spot-check the model's outputs and reward extraction.

Usage:
    python profile_run.py --config configs/math.yaml [--n 10] [--seed 42] [--out profile.md]

Outputs:
- Stdout: per-question metrics, full-dataset prompt-length distribution, summary
  table, and concrete recommendations for tightening or widening the YAML budgets.
- Markdown file (default `profile.md`): full text of every question, attempt 1, and
  attempt 2 so you can scroll the actual model output later.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from unsloth import FastLanguageModel

import reward_function as rf
from train import load_and_prepare_data, load_config, load_model_and_tokenizer


def percentiles(values: list[int]) -> dict[str, int]:
    arr = np.array(values)
    return {
        "n": len(arr),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": int(arr.mean()),
    }


def fmt_pct(p: dict[str, int]) -> str:
    return (
        f"n={p['n']:<5}  min={p['min']:<6}  p50={p['p50']:<6}  "
        f"p95={p['p95']:<6}  p99={p['p99']:<6}  max={p['max']:<6}  mean={p['mean']:<6}"
    )


def count_think_tokens(text: str, tokenizer: Any) -> tuple[int, int]:
    """Return (think_tokens, post_think_tokens). Closes-tag-only matches still count."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return 0, len(tokenizer.encode(text, add_special_tokens=False))
    think_text = match.group(1)
    post = text[match.end():].strip()
    n_think = len(tokenizer.encode(think_text, add_special_tokens=False))
    n_post = len(tokenizer.encode(post, add_special_tokens=False))
    return n_think, n_post


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--n", type=int, default=10, help="Number of questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="profile.md", help="Markdown output file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    print(f"\n=== Phase 1: attempt-1 prompt token distribution across full train set ===")
    prompt_lens: list[int] = []
    for ex in train_dataset:
        prompt_lens.append(len(tokenizer.encode(ex["text"], add_special_tokens=False)))
    p1 = percentiles(prompt_lens)
    print(f"  prompt tokens: {fmt_pct(p1)}")
    cfg_a1 = cfg.train.max_prompt_length_attempt1
    if p1["max"] > cfg_a1:
        print(f"  WARNING: max ({p1['max']}) > max_prompt_length_attempt1 ({cfg_a1}) — some prompts will be truncated.")
    elif p1["p99"] < cfg_a1 // 2:
        print(f"  Note: p99 ({p1['p99']}) is well under max_prompt_length_attempt1 ({cfg_a1}); could reduce to ~{int(p1['p99'] * 1.3)}.")
    else:
        print(f"  OK: p99 ({p1['p99']}) fits within max_prompt_length_attempt1 ({cfg_a1}).")

    # --- Phase 2: rollout N random examples ---------------------------------
    print(f"\n=== Phase 2: rollout {args.n} random examples ===")
    indices = random.sample(range(len(train_dataset)), args.n)

    a1_total, a1_think, a1_post = [], [], []
    a2_total, a2_think, a2_post = [], [], []
    a2_prompt = []
    rewards1, rewards2 = [], []
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
    gen_temp = cfg.train.generation_temperature if do_sample else 1.0

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
            do_sample=do_sample,
            temperature=gen_temp,
        )
        a1_new = action1[:, x1_len:][0]
        a1_text = tokenizer.decode(a1_new, skip_special_tokens=True)
        n1 = int((a1_new != tokenizer.pad_token_id).sum().item()) if tokenizer.pad_token_id is not None else len(a1_new)
        n1_think, n1_post = count_think_tokens(a1_text, tokenizer)
        a1_total.append(n1)
        a1_think.append(n1_think)
        a1_post.append(n1_post)
        if n1 >= cfg.train.max_new_tokens_attempt1 - 1:
            a1_budget_hits += 1
        a1_extracted = extractor(a1_text)
        if not a1_extracted:
            a1_extract_fails += 1
        r1 = reward_fn([a1_extracted], [target_extracted])[0]
        rewards1.append(r1)

        # Build attempt 2
        messages = list(ex["messages"])
        messages.append({"role": "assistant", "content": a1_text})
        messages.append({"role": "user", "content": cfg.prompts.self_correction})
        x2_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        x2 = tokenizer(x2_text, padding=False, return_tensors="pt").to(model.device)
        x2_len = x2.input_ids.shape[1]
        a2_prompt.append(x2_len)

        action2 = model.generate(
            x2.input_ids,
            attention_mask=x2.attention_mask,
            max_new_tokens=cfg.train.max_new_tokens_attempt2,
            do_sample=do_sample,
            temperature=gen_temp,
        )
        a2_new = action2[:, x2_len:][0]
        a2_text = tokenizer.decode(a2_new, skip_special_tokens=True)
        n2 = int((a2_new != tokenizer.pad_token_id).sum().item()) if tokenizer.pad_token_id is not None else len(a2_new)
        n2_think, n2_post = count_think_tokens(a2_text, tokenizer)
        a2_total.append(n2)
        a2_think.append(n2_think)
        a2_post.append(n2_post)
        if n2 >= cfg.train.max_new_tokens_attempt2 - 1:
            a2_budget_hits += 1
        a2_extracted = extractor(a2_text)
        if not a2_extracted:
            a2_extract_fails += 1
        r2 = reward_fn([a2_extracted], [target_extracted])[0]
        rewards2.append(r2)

        # Compact stdout line per question
        print(
            f"  Q{i:>2} idx={idx:<5}  "
            f"x1={x1_len:<4}  a1=[total={n1}, think={n1_think}, post={n1_post}]  "
            f"x2={x2_len:<5}  a2=[total={n2}, think={n2_think}, post={n2_post}]  "
            f"r1={r1:.0f} r2={r2:.0f}  "
            f"e1={a1_extracted!r:<20}  e2={a2_extracted!r:<20}  target={target_extracted!r}"
        )

        # Full text for the markdown file
        md_lines += [
            f"### Q{i} (idx={idx})",
            "",
            f"- **Target answer:** `{target_extracted}`",
            f"- **Attempt 1:** prompt={x1_len} tokens, generated={n1} (think={n1_think}, post={n1_post}), extracted={a1_extracted!r}, reward={r1:.0f}",
            f"- **Attempt 2:** prompt={x2_len} tokens, generated={n2} (think={n2_think}, post={n2_post}), extracted={a2_extracted!r}, reward={r2:.0f}",
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
        ("attempt-1 think tokens",    a1_think),
        ("attempt-1 post-think tokens", a1_post),
        ("attempt-2 prompt tokens",   a2_prompt),
        ("attempt-2 generated total", a2_total),
        ("attempt-2 think tokens",    a2_think),
        ("attempt-2 post-think tokens", a2_post),
    ]
    print(f"{'metric':<30}  {'min':>5} {'p50':>5} {'p95':>5} {'p99':>5} {'max':>5} {'mean':>5}")
    md_lines += ["## Summary", "", "| metric | min | p50 | p95 | p99 | max | mean |", "|---|---|---|---|---|---|---|"]
    for name, vals in summary_rows:
        p = percentiles(vals)
        print(f"{name:<30}  {p['min']:>5} {p['p50']:>5} {p['p95']:>5} {p['p99']:>5} {p['max']:>5} {p['mean']:>5}")
        md_lines.append(f"| {name} | {p['min']} | {p['p50']} | {p['p95']} | {p['p99']} | {p['max']} | {p['mean']} |")
    md_lines.append("")

    print(f"\n  Reward attempt 1: {np.mean(rewards1):.2f}")
    print(f"  Reward attempt 2: {np.mean(rewards2):.2f}")
    print(f"  Delta (a2-a1):    {np.mean(rewards2) - np.mean(rewards1):+.2f}")
    print(f"  Attempt 1 hit max_new_tokens: {a1_budget_hits}/{args.n}")
    print(f"  Attempt 2 hit max_new_tokens: {a2_budget_hits}/{args.n}")
    print(f"  Attempt 1 extraction failures: {a1_extract_fails}/{args.n}")
    print(f"  Attempt 2 extraction failures: {a2_extract_fails}/{args.n}")

    md_lines += [
        f"- Reward attempt 1: **{np.mean(rewards1):.2f}**",
        f"- Reward attempt 2: **{np.mean(rewards2):.2f}**",
        f"- Delta: **{np.mean(rewards2) - np.mean(rewards1):+.2f}**",
        f"- Attempt 1 hit `max_new_tokens`: {a1_budget_hits}/{args.n}",
        f"- Attempt 2 hit `max_new_tokens`: {a2_budget_hits}/{args.n}",
        f"- Attempt 1 extraction failures: {a1_extract_fails}/{args.n}",
        f"- Attempt 2 extraction failures: {a2_extract_fails}/{args.n}",
        "",
    ]

    # --- Recommendations ---------------------------------------------------
    print(f"\n=== Recommendations ===")
    md_lines += ["## Recommendations", ""]

    def rec(name: str, current: int, observed_max: int, hit_count: int) -> str:
        if hit_count > 0:
            msg = f"  {name}: {current} — INCREASE (hit {hit_count}/{args.n} times)."
        elif observed_max < current * 0.5:
            new = max(int(observed_max * 1.3), 64)
            msg = f"  {name}: {current} — could reduce to ~{new} (observed max {observed_max})."
        else:
            msg = f"  {name}: {current} — looks right (observed max {observed_max})."
        return msg

    for line in [
        rec("max_new_tokens_attempt1", cfg.train.max_new_tokens_attempt1, max(a1_total), a1_budget_hits),
        rec("max_new_tokens_attempt2", cfg.train.max_new_tokens_attempt2, max(a2_total), a2_budget_hits),
        rec("max_prompt_length_attempt1", cfg.train.max_prompt_length_attempt1, max(p1["max"], 0), 0),
        rec("max_prompt_length_attempt2", cfg.train.max_prompt_length_attempt2, max(a2_prompt), 0),
    ]:
        print(line)
        md_lines.append(line)

    if a1_extract_fails > 0 or a2_extract_fails > 0:
        warn = (
            f"  WARNING: {a1_extract_fails + a2_extract_fails} extraction failures — "
            "the model's answer format is not matching the regex. "
            "Either tighten the system prompt or extend the extractor in reward_function.py."
        )
        print(warn)
        md_lines.append(warn)

    Path(args.out).write_text("\n".join(md_lines))
    print(f"\nFull per-question text written to: {args.out}")


if __name__ == "__main__":
    main()
