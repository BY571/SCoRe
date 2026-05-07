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

- ~~**Stage II log-probs over generated tokens only**~~ — already fixed on main (PR #1, merged 2026-05-07). The new `score_math.py` uses a shared `get_log_probs(model, ids, prompt_len)` helper that slices the prompt off and shifts logits by one. We will keep the same approach.
- **Math reward = binary final-answer match, not similarity-based** — main still uses `LLMAnswerComparator` (BERT cosine similarity, threshold=0.9). The paper uses 0/1 correctness (final answer string matches). With our extractor in place, normalized string equality on the extracted final answer is the right reward and removes the silent failure mode where "41" vs "42" can score 0.95. Drop `LLMAnswerComparator`. `bert_similarity` is **not** ported.

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
