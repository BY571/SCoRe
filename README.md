# SCoRe: Self-Correct via Reinforcement Learning

Task-agnostic implementation of [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) (Kumar et al., ICLR 2025), built on [Unsloth](https://github.com/unslothai/unsloth) LoRA for small open-weight models.

Core algorithm is as in the paper: Stage I anchors attempt 1 to the reference policy via KL, Stage II adds the reward-shaping bonus `α · (r(y2) − r(y1))`. Adaptations we introduced:

- **Reasoning-tag format from [DeepSeek-R1](https://arxiv.org/abs/2501.12948)** (the GRPO paper): model reasons inside `<think>...</think>` and gives the final answer inside `<answer>...</answer>`. The compound reward below scores this format directly.
- **K3 KL estimator** (Schulman): `K3 = exp(log π_ref − log π) − (log π_ref − log π) − 1`. Unbiased forward KL from per-token log-probs only — no `[B, T, V]` log-softmax tensor, which is what makes Stage II's two-graph backward fit in memory.
- **Optional Dr.GRPO-style length normalization** (opt-in via `train.length_norm`, **not** the default): the default `"sequence"` divides the PG and K3 KL terms by the actual per-sample generated length — the original form. Setting `"constant"` instead divides by `max_new_tokens` ([Dr.GRPO](https://arxiv.org/abs/2503.20783)), which avoids the length bias where the policy games the loss by inflating generation length (each extra low-KL / low-grad token otherwise dilutes the per-token signal). Flip it in the YAML if you want it.
- **Compound reward** (`format_and_match`): 0.25 for a `<think>...</think>` pair, 0.25 for one `<answer>...</answer>` pair, 0.5 for extracted-answer match. Gives the α-bonus more signal than pure binary and explicitly anchors format. A stricter `strict_format_and_match` variant (0.5 format / 0.5 match) requires the *whole* output to be exactly the two blocks — no prose between `</think>` and `<answer>`, nothing after `</answer>`. Select either via `reward.fn` in the YAML.
- **LoRA-only**: reference policy = same model with `model.disable_adapter()`. No second model in VRAM.

## Layout

```
train.py                 # task-agnostic entry point
reward_function.py       # @register_reward / @register_extractor + shipped fns
configs/gsm8k.yaml       # primary target
configs/arithmetic.yaml  # fast toy task — 5-digit subtraction Qwen3-0.6B can't solve; full run < 1h
configs/math.yaml        # math baseline
build_hard_arithmetic.py # builds the toy-task dataset (greedy-decode, keep the failures)
profile_run.py           # optional length/extraction diagnostic
```

`train.py` never references a specific task — model, dataset, prompts, reward, and extractor are all YAML.

## Run

```bash
pip install -r requirements.txt

# full run
python train.py --config configs/gsm8k.yaml

# fast toy task (~1h two-stage run) — good for debugging the loop end-to-end
python train.py --config configs/arithmetic.yaml

# 1-minute end-to-end smoke (N examples, capped eval)
python train.py --config configs/gsm8k.yaml --smoke 8

# skip stage 1 and resume stage 2 from a saved stage-1 adapter
python train.py --config configs/gsm8k.yaml \
    --start-stage 2 --resume-adapter outputs/score-gsm8k/stage1/step_200
```

Logs go to W&B (`wandb_project` in the config). End-of-stage adapters save to `outputs/{run_name}/stage{1,2}/`. Set `train.checkpoint_every: N` in the YAML to also save mid-stage to `outputs/{run_name}/stage{1,2}/step_N/` every N optimizer steps — useful for resuming or for picking the best checkpoint by reward curve.

`configs/arithmetic.yaml` uses [`torchtrade/arithmetic-hard-qwen3-0.6b`](https://huggingface.co/datasets/torchtrade/arithmetic-hard-qwen3-0.6b) — 200 train / 50 eval 5-digit subtractions that Qwen3-0.6B fails greedy-decoded (built by `build_hard_arithmetic.py`; see the dataset card for source + method). It's a debug/shake-out task, not a benchmark: difficulty is defined relative to that one model, so a full two-stage run finishes in ~1h and stresses every part of the loop.

## Adapt to a new task

1. Push your dataset to the HF Hub with `train` and `test` splits. For datasets needing a sub-config (e.g. `openai/gsm8k` has `main`/`socratic`), set `dataset.config_name`.
2. Add an answer extractor in `reward_function.py` if the shipped ones don't fit (`gsm8k_hash`, `math_final_answer`, `identity`).
3. Add a reward function if none of the shipped ones fit.
4. Copy `configs/gsm8k.yaml`, edit `model.*` / `dataset.*` / `prompts.*` / `reward.*`.
5. Smoke first (`--smoke 8`), then full.
