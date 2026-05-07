# SCoRe Cleanup + Unsloth Refactor — Design

**Date:** 2026-05-07
**Branch target:** new branch off `main`, opened as a PR
**Goal:** Replace the three near-duplicate training scripts with a single task-agnostic `train.py` driven by a YAML config and a registry of named reward functions, using Unsloth for efficient LoRA training. Fix two correctness issues found in the existing code.

## 1. Scope

**In:**
- New `train.py` (task-agnostic; reads YAML config).
- New `reward_function.py` (registry of reward functions and answer extractors).
- New `configs/math.yaml` (the only shipped config; serves as the documented example).
- Unsloth `FastLanguageModel` for model + LoRA loading; reference policy via `model.disable_adapter()` (no second model in VRAM).
- Eval pass after each stage logging attempt-1/attempt-2 accuracy to W&B.
- LoRA-adapter checkpointing at the end of each stage.
- Two correctness fixes (see §6).
- Updated `README.md`, `requirements.txt`, `.gitignore`.

**Out:**
- Multi-GPU / DeepSpeed / FSDP.
- Hyperparameter sweeps (Hydra, sweep configs).
- Replaying prior runs / additional task configs (toy task removed; the README explains how to add one).
- Unit tests (this is a research repo; ship a working `train.py` and one config).
- Maintaining a non-unsloth model-loading path.

## 2. File Layout

```
SCoRe/
├── train.py
├── reward_function.py
├── configs/
│   └── math.yaml
├── README.md
├── requirements.txt
└── .gitignore
```

**Files removed in this PR:** `score_math.py`, `score_toy.py`, `string_matcher.py`, `dataset_relabel.py`, `test_chat.py`, `test_batch_request.py`, `test.ipynb`.

**Files un-staged / gitignored:** `wandb/`, `__pycache__/`, `outputs/`, `data/`, `api.txt`, `*.ipynb`.

## 3. Config Schema (`configs/math.yaml`)

```yaml
run_name: score-math
wandb_project: SCoRe-Math

model:
  name: unsloth/Llama-3.2-1B-Instruct
  max_seq_length: 2048
  load_in_4bit: true
  chat_template: llama-3.1     # passed to unsloth.chat_templates.get_chat_template
  lora:
    r: 16
    alpha: 16
    dropout: 0.0
    target_modules:
      [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

dataset:
  name: Sebasdi/math_final_answer
  input_field: problem
  target_field: final_answer_solution
  test_split_size: 200          # examples taken from `test` split for eval

prompts:
  system: |
    You are a math expert. Solve problems step by step, showing all work.
    Final Answer Format:
    - For numbers: "The final answer is: 42"
    - For equations: "The final answer is: $y = 2x - 13$"
    Always clearly show your reasoning and conclusion.
  self_correction: |
    There might be an error in the solution above because of lack of understanding
    of the question. Please correct the error, if any, and rewrite the solution.
    At the end of the Solution, when you give your final answer indicate your final
    answer by writing "the final answer is: <your answer>".

reward:
  fn: exact_match                # name from reward_function.py registry
  answer_extractor: math_final_answer  # applied to both prediction and target before scoring

train:
  batch_size: 2
  learning_rate: 5.0e-6
  stage1_epochs: 1
  stage2_epochs: 1
  beta1: 0.01                    # KL coeff in Stage II
  beta2: 0.1                     # KL coeff in Stage I (anchors first attempt to base)
  alpha: 10.0                    # Stage II reward shaping bonus multiplier
  max_new_tokens_attempt1: 1000
  max_new_tokens_attempt2: 1000
  attempt2_max_input_length: 1500
  generation_temperature: 0.5
  seed: 3407

output:
  dir: outputs                   # adapter saved to outputs/{run_name}/stage{1,2}/
  save_after_stage1: true
  save_after_stage2: true
```

Loaded into a typed dataclass (`SCoReConfig`) at the top of `train.py` via `pyyaml`. Nested dataclasses for `model`, `dataset`, `prompts`, `reward`, `train`, `output`. Unknown keys raise — keeps configs honest.

CLI: `python train.py --config configs/math.yaml`. A single `--override key.path=value` flag on top is nice but **deferred** unless trivial.

## 4. `reward_function.py`

Two registries, one decorator each:

```python
REWARD_FNS: dict[str, Callable[[list[str], list[str]], list[float]]] = {}
ANSWER_EXTRACTORS: dict[str, Callable[[str], str]] = {}

def register_reward(name): ...
def register_extractor(name): ...
```

Shipped functions:

| Name                  | Type      | Behaviour |
|-----------------------|-----------|-----------|
| `exact_match`         | reward    | `1.0` if normalized strings match (whitespace-stripped, lowercase) else `0.0`. |
| `math_final_answer`   | extractor | regex with fallbacks (handles `"final answer is: X"`, `$X$`, `\boxed{X}`, plain numbers). Ports the `extract_final_answer` from `score_math_unsloth.py`. |
| `identity`            | extractor | returns the input unchanged; used when no extraction is needed. |

**Composition.** `train.py` always pipes through `extractor → reward_fn`:

```python
extracted_pred   = [extractor(p) for p in predictions]
extracted_target = [extractor(t) for t in targets]
rewards = reward_fn(extracted_pred, extracted_target)
```

This keeps reward functions task-agnostic. The math task is just `exact_match` after `math_final_answer` extraction.

`train.py` looks the registry up by string name from the config — no `importlib`, no dotted paths.

**Adding a task** = write at most one extractor + one reward function in this file (often just an extractor), decorate, reference by name from YAML.

## 5. `train.py` Shape

Top-down, ~250 lines total. Single file. Functions in this order:

1. `load_config(path) -> SCoReConfig` — pyyaml load, dataclass cast, validation.
2. `load_model_and_tokenizer(cfg.model)` — `FastLanguageModel.from_pretrained` + `get_peft_model` + `get_chat_template`. Returns `(model, tokenizer)`.
3. `load_and_prepare_data(cfg.dataset, cfg.prompts, tokenizer)` — pulls dataset, splits per `test_split_size`, applies chat template using `prompts.system` and `examples[input_field]`. Returns `(train_iter, test_iter)` (HF dataset `.iter(batch_size=...)`). Carries `messages`, `text`, and the raw `target` through to the training loop.
4. `rollout(model, tokenizer, batch, cfg) -> RolloutResult` — runs both attempts; returns generated tokens, decoded text, attempt-2 prompt boundary, and the per-example rewards from the configured reward fn. Used by both stages and by eval. **Refactored once, shared.**
5. `stage1_step(model, batch, cfg, optimizer)` — implements `L1 = -E[r(y2) * log π(y2)] + β2 * KL(π(y1)||π_ref(y1))` using `model.disable_adapter()` for the reference distribution. Computes log-probs only over the **generated** attempt-2 tokens (not the prompt).
6. `stage2_step(model, batch, cfg, optimizer)` — implements `L2 = -E[(r(y1)+r(y2)+α(r(y2)-r(y1))) * (logπ(y1)+logπ(y2))] + β1 * (KL_1 + KL_2)`. Same disable_adapter trick. Log-probs over generated tokens only for both attempts (fixes the Stage II bug from the current code).
7. `evaluate(model, tokenizer, test_iter, cfg) -> dict` — runs `rollout` greedily over test set, returns `{"acc_attempt1": ..., "acc_attempt2": ..., "delta": ...}`.
8. `save_adapter(model, tokenizer, path)` — `model.save_pretrained(path); tokenizer.save_pretrained(path)`.
9. `main()` — wandb init, build everything, run Stage I (loop epochs → loop batches → step → end-of-stage eval → save), then Stage II identically.

**Reference policy** — Unsloth doesn't expose a separate base model; we use `with model.disable_adapter():` to compute base logits with LoRA off. This halves VRAM vs. loading a second copy.

**Padding side, dtype, autocast** — left padding for generation; bfloat16 autocast wrapped around forward + loss; gradient checkpointing on (unsloth's). Same pattern the experimental unsloth file already uses.

## 6. Correctness fixes carried in this PR

These are small deltas relative to the existing scripts; documented so they don't surprise reviewers.

### 6.1 Already fixed on main

- ~~**Stage II log-probs over generated tokens only**~~ — already fixed on main (PR #1, merged 2026-05-07). The shared `get_log_probs(model, ids, prompt_len)` helper slices the prompt off and shifts logits by one. We keep the same approach (now vectorized — see §6.3).

### 6.2 Algorithm fixes vs the paper

Verified against ICLR 2025 PDF (`arXiv:2409.12917`). Equation references are to the paper.

- **Math reward = binary final-answer match, not similarity-based.** Main used `LLMAnswerComparator` (BERT cosine, threshold=0.9). Paper uses 0/1 correctness; "41" vs "42" should score 0. Replaced with `exact_match` after the configurable answer extractor.
- **Stage I PG drops the first-attempt log-probs.** Main's loss had `(log π(y1) + log π(y2)) · r(y2)`. Paper's Stage I objective (Eq. 3) only rewards `r(y2)`; with the "instantaneous reward, γ=0" convention (Appendix A.4 line 1130), `log π(y1)` gets weighted by 0 and falls out. Stage I leaves the first-attempt distribution shaped only by the KL anchor.
- **KL is forward `D_KL(π_θ || π_ref)` and summed over vocab.** The original `get_kl_div` had three independent issues:
  1. `F.kl_div(input=log_π, target=log_base, log_target=True)` computes `KL(base || π)` (reverse KL). Paper Eq. 3/4 use forward KL.
  2. `kl.mean(-1)` divides by `vocab_size` (~128k for Llama-3); KL must be **summed** over the support, so β values were effectively scaled by `1/vocab_size`.
  3. Callers passed `log_π.detach()` to `get_kl_div`, so β·KL contributed **no gradient** at all — Stage I had no first-attempt anchor in practice.

  Fix: compute KL manually as `Σ exp(log_π)·(log_π − log_base)`, summed over vocab, with `log_base` from `model.disable_adapter()` under `torch.no_grad()` (no grad needed there) and **no `.detach()`** on `log_π` (gradient must flow).

- **`math_final_answer` regex.** The original third pattern `[^\n.]+` excluded `.`, silently truncating decimals (`"3.14"` → `"3"`). The boxed pattern `[^}]+` stopped at the first `}`, breaking nested LaTeX (`\boxed{\frac{1}{2}}` → `\frac{1`). Replaced with: case-insensitive marker search, `$...$`, balanced-brace `\boxed{...}` scan, and a non-greedy plain-text capture that allows decimals (`(.+?)(?=\.\s|\.$|\n|$)`).

### 6.3 Other fixes from PR review

- **`optimizer.zero_grad / backward / step` moved outside the `autocast` block** (the standard pattern; works for bf16 today, won't surprise anyone who ever switches to fp16 with a grad scaler).
- **W&B logging is per-step**, not a smeared running mean of batch means accumulated across the epoch.
- **`evaluate` slices the test set upfront** with `test_dataset.select(range(min(...)))` instead of using a per-batch `seen` counter that overshoots.
- **`get_log_probs` is vectorized**: dropped the Python batch loop (which served no memory purpose since rows were `torch.stack`-ed at the end anyway), and dropped the `return_probs` flag (every Stage I/II caller wanted both returns).
- **`RewardConfig.__post_init__`** validates `fn` and `answer_extractor` against the registry, so config typos fail at config-load instead of after model+dataset load.
- **`TrainConfig.__post_init__`** asserts `max_prompt_length_attempt2 ≥ max_prompt_length_attempt1`.
- **`load_and_prepare_data`** raises if `test_split_size > len(test_dataset)`.
- **`Rollout`** drops the redundant `attempt{1,2}_answer_tokens` fields; they're now `@property` views over `action{1,2}_tokens` so they can't desynchronize.
- **Seed coverage**: `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all` all set.
- **YAGNI drops**: `EvalConfig.enabled` (set `max_examples: 0` to skip eval) and `OutputConfig.save_after_stage1`/`save_after_stage2` (always save — they're a few hundred MB and the only artifacts the script produces).
- **`identity` extractor** kept (3 LOC) so users have a documented "no extraction" option in the registry.
- **`_normalize` in `reward_function.py`** inlined into `exact_match` (one `strip().lower()` per side).

## 7. README contents (rewritten)

Sections, brief:

1. **What this is** — one paragraph + link to paper.
2. **Install** — `pip install -r requirements.txt`. Note Unsloth's CUDA requirement.
3. **Run** — `python train.py --config configs/math.yaml`.
4. **Adapt to your task** — a 5-step recipe:
   1. Push your dataset to HF Hub with `train` and `test` splits, an input column, and a target column.
   2. If your target's final-answer format differs, add an extractor to `reward_function.py` with `@register_extractor("my_extractor")`.
   3. If your reward needs custom logic, add a reward fn with `@register_reward("my_reward")`.
   4. Copy `configs/math.yaml`, edit `dataset.*`, `prompts.*`, `reward.*`.
   5. `python train.py --config configs/yours.yaml`.
5. **Dataset format contract** — required splits and fields.
6. **Outputs** — where adapters and W&B logs land.

## 8. requirements.txt

```
torch
transformers
datasets
unsloth
peft
trl
accelerate
bitsandbytes
pyyaml
wandb
tqdm
```

Removed: `sentence-transformers`, `scikit-learn`, `nltk` (string_matcher gone).

## 9. .gitignore

```
__pycache__/
*.pyc
wandb/
outputs/
data/
api.txt
*.ipynb
.venv/
```

## 10. Acceptance criteria

- `python train.py --config configs/math.yaml` runs end-to-end on a single GPU and:
  - Loads `unsloth/Llama-3.2-1B-Instruct` 4-bit + LoRA.
  - Logs `mr_attempt1`, `mr_attempt2`, `difference_at1_at2`, `loss` to W&B in both stages.
  - Logs `eval/acc_attempt1`, `eval/acc_attempt2`, `eval/delta` after each stage.
  - Writes `outputs/score-math/stage1/` and `outputs/score-math/stage2/` containing the LoRA adapter and tokenizer.
- No reference to "math" in `train.py`.
- `train.py` + `reward_function.py` together stay under ~400 lines.
- Old scripts and `string_matcher.py` deleted; tracked-but-unwanted artifacts (`__pycache__/`, `wandb/`, `api.txt`) are gitignored and untracked.
