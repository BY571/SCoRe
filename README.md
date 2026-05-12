# SCoRe: Self-Correct via Reinforcement Learning

Task-agnostic, **SCoRe-inspired** implementation of [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) (Kumar et al., ICLR 2025), with practical modifications so it runs cleanly on small open-weight models with [Unsloth](https://github.com/unslothai/unsloth) LoRA.

The core algorithm — Stage I with a KL anchor on the first attempt, Stage II with the reward-shaping bonus `α · (r(y2) − r(y1))` — is faithful to the paper. Deviations:

- **K3 KL estimator (Schulman).** Forward KL is approximated as `K3 = exp(log π_ref − log π) − (log π_ref − log π) − 1`, which is unbiased, non-negative, and requires only the per-token log-probabilities of *sampled* tokens. This eliminates the `[B, T, V]` log-softmax tensor that the naive KL form needs — critical for Stage II, which holds two log-prob graphs simultaneously.
- **Reasoning-model integration.** The shipped prompts ask the model to reason inside `<think>...</think>` and put its final answer in `<answer>...</answer>`. For Qwen3 (native thinking mode), the chat template silently strips raw `<think>` tags from prior assistant turns; SCoRe needs the corrector to see the prior reasoning, so attempt 1's `<think>` content is rewritten as plain-text `[Prior reasoning: ...]` before being appended as history. Qwen2.5 does *not* strip them, so the rewrite is a harmless no-op there.
- **Compound reward.** Default reward `format_and_match`: 0.25 for a balanced `<think>...</think>` pair, 0.25 for exactly one `<answer>...</answer>` pair, 0.5 for extracted-answer match. Gives Stage II's α-bonus more signal than a pure binary reward and explicitly anchors format compliance.
- **Unsloth + LoRA.** The reference policy is the same model with the LoRA adapter disabled (`with model.disable_adapter():`) — no second model in VRAM. Full fine-tuning is out of scope here; the paper trains much larger base models.

## Layout

```
train.py              # task-agnostic entry point
reward_function.py    # @register_reward / @register_extractor + shipped fns
configs/gsm8k.yaml    # primary target: GSM8K + Qwen2.5-1.5B-Instruct
configs/math.yaml     # math dataset baseline (Qwen3-4B-bnb-4bit + MATH-final-answer)
profile_run.py        # optional standalone diagnostic — profile lengths before training
```

`train.py` never references a specific task. Task-specific knobs (model, dataset fields, prompts, reward fn / extractor) all live in the YAML.

## Setup

Follow the [Unsloth install guide](https://github.com/unslothai/unsloth#-installation-instructions) for your CUDA version, then:

```bash
pip install -r requirements.txt
```

## Run

```bash
# primary target
python train.py --config configs/gsm8k.yaml

# end-to-end smoke (N examples, caps eval, validates the full loop in ~1 min)
python train.py --config configs/gsm8k.yaml --smoke 8
```

Logs go to W&B (project from `wandb_project` in the config). LoRA adapters are saved to `outputs/{run_name}/stage{1,2}/`.

## What gets logged to W&B

Per training step, both stages log:

- `stageN/reward_attempt1`, `stageN/reward_attempt2`, `stageN/diff_attempt2_minus_attempt1` — the key SCoRe-deltas
- `stageN/loss`, `stageN/kl_div[_attemptK]`
- `stageN/attemptK_{len_mean,len_p95,len_max,truncated_frac}` — generation-length stats per attempt
- `stageN/attemptK_lens_hist` — per-step length histogram
- `stageN/rollouts` — wandb Table with up to 4 sampled rollouts: prompt tail, target, attempt 1 text, attempt 2 text, rewards, lengths, truncation flags. Sort by `trunc1=True` to see what the model produces when it runs out of budget.

Per eval, the same `rollouts` table is logged (up to 8 samples) plus scalar accuracy + length stats.

## Model-specific notes

The two shipped configs target different model families.

**Qwen2.5-Instruct (gsm8k.yaml, primary):**
- `chat_template: ""` — empty uses the tokenizer's built-in ChatML.
- `eval.generation_temperature: 0.0` — greedy is fine, no thinking-mode loop pathology.
- Native thinking mode is *not* present; the `<think>` blocks come from format-following the system prompt. Verified that Qwen2.5's chat template preserves `<think>` in multi-turn history out of the box.

**Qwen3 (math.yaml, baseline):**
- Same `chat_template: ""`.
- `eval.generation_temperature > 0` is required — greedy on Qwen3 in thinking mode can produce infinite loops per the model's docs.
- The chat template strips `<think>` from history; the `thinking_to_history` helper in `train.py` rewrites them to plain text before re-rendering.

## Sizing the token budgets

Three knobs that need to be sized together (in YAML):

- `model.max_seq_length` — total cap (prompt + generation) the model accepts.
- `train.max_new_tokens_attempt{1,2}` — generation caps per attempt.
- `train.max_prompt_length_attempt2` — must hold the attempt-1 prompt + attempt-1 output + self-correction wrapper.

The `stageN/attemptK_truncated_frac` metric tells you when these are too small: if a high fraction of samples hit the cap without emitting EOS, the reward is being silently truncated to 0 (no `<answer>` reached), and the policy will drift. Bump `max_new_tokens` and re-run. The shipped `gsm8k.yaml` uses 1024 / 1024 with a 6144 total seq length after observing 50% truncation at 512.

## Adapt to a new task

Everything task-specific lives in the YAML and the reward registry.

1. **Push your dataset to the HuggingFace Hub** with `train` and `test` splits, an input column and a target column. For datasets that require a sub-config (e.g. `openai/gsm8k` has `main` / `socratic`), set `dataset.config_name` in the YAML.
2. **Add an answer extractor** to `reward_function.py` if the shipped `math_final_answer`, `gsm8k_hash`, or `identity` extractors don't match your target format:
    ```python
    @register_extractor("my_extractor")
    def my_extractor(text: str) -> str:
        ...
    ```
3. **Add a reward function** if `format_and_match` or `exact_match` don't fit.
4. **Copy `configs/gsm8k.yaml`**, change `model.*`, `dataset.*`, `prompts.*`, and `reward.*` to match your task.
5. `python train.py --config configs/yours.yaml --smoke 8` first, then full.

## Dataset format contract

The dataset must have a `train` and `test` split, with:

- A column named by `dataset.input_field` — the question / prompt (string).
- A column named by `dataset.target_field` — ground-truth answer (string) in a form the chosen `answer_extractor` can parse.

`train.py` only reads these two fields. Anything else in the dataset is ignored.

### Shipped extractors

| Extractor | Handles |
|---|---|
| `gsm8k_hash` | `<answer>N</answer>` (predictions) or `#### N` (GSM8K targets); falls back to the last number in text. Strips commas, preserves sign and decimals. |
| `math_final_answer` | `<answer>...</answer>` first, then any `\boxed{...}` anywhere, then a `final answer is:` marker. Peels `$...$`, `\[...\]`, `\boxed{...}` wrappers iteratively. |
| `identity` | Passes the text through as-is. |

### Math baseline dataset

`Sebasdi/math_final_answer` is `lighteval/MATH` with each solution suffixed by an extracted final-answer marker (so the extractor has something to parse). The relabel script used GPT-3.5 to extract the final answer from each MATH solution; it is preserved in git history if regeneration is ever needed.
