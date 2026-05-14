"""Task-agnostic SCoRe training entry point.

Run with:
    python train.py --config configs/math.yaml

The script never references a specific task. Everything task-specific lives
in the YAML (model, dataset fields, prompts) and the reward registry.

Algorithm reference: Kumar et al., "Training Language Models to Self-Correct
via Reinforcement Learning" (arXiv:2409.12917, ICLR 2025).

  Stage I  (Eq. 3):  max E[ r(y2, y*) - beta2 * D_KL(pi_theta(.|x1) || pi_ref(.|x1)) ]
  Stage II (Eq. 4):  max E[ sum_i r(yi, y*) - beta1 * D_KL(pi_theta(.|xi) || pi_ref(.|xi)) ]
                     with reward shaping bonus b(y2|y1, y*) = alpha * (r(y2, y*) - r(y1, y*))
                     added to r(y2) (Eq. 5).

The paper uses the "instantaneous reward" form (gamma = 0, Appendix A.4), so the
policy gradient credits each attempt's log-probs only with that attempt's reward.
Stage I has no first-attempt reward, so log pi(y1) drops out of the PG term and
the first-attempt distribution is shaped only by the KL anchor.

Reference policy: same model with the LoRA adapter disabled
(`with model.disable_adapter():`).
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from datasets import concatenate_datasets, load_dataset
from torch import Tensor
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import reward_function as rf


# --- Config ------------------------------------------------------------------


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class ModelConfig:
    name: str
    max_seq_length: int
    load_in_4bit: bool
    chat_template: str
    lora: LoRAConfig
    fast_inference: bool = False           # use vLLM for generation (Unsloth `fast_inference=True`)
    fast_inference_gpu_mem: float = 0.5    # vLLM GPU memory share when fast_inference is on


@dataclass
class DatasetConfig:
    name: str
    input_field: str
    target_field: str
    test_split_size: int
    config_name: str | None = None    # for HF datasets that require a sub-config (e.g. openai/gsm8k "main")
    split: str | None = None          # if set, load just this one split spec (slicing allowed, e.g.
                                      # "validation[:2000]") and partition it into train + held-out
                                      # eval. None = expect separate "train" and "test" splits.
    revision: str | None = None       # HF dataset revision — e.g. "refs/convert/parquet" to use the
                                      # auto-generated parquet export of a legacy script dataset.


@dataclass
class PromptsConfig:
    system: str
    self_correction: str


@dataclass
class RewardConfig:
    fn: str
    answer_extractor: str

    def __post_init__(self) -> None:
        if self.fn not in rf.REWARD_FNS:
            raise ValueError(
                f"Unknown reward fn '{self.fn}'. "
                f"Registered: {sorted(rf.REWARD_FNS)}"
            )
        if self.answer_extractor not in rf.ANSWER_EXTRACTORS:
            raise ValueError(
                f"Unknown answer_extractor '{self.answer_extractor}'. "
                f"Registered: {sorted(rf.ANSWER_EXTRACTORS)}"
            )


@dataclass
class TrainConfig:
    stage1_batch_size: int
    stage2_batch_size: int
    learning_rate: float
    stage1_epochs: int
    stage2_epochs: int
    beta1: float
    beta2: float
    alpha: float
    max_new_tokens_attempt1: int
    max_new_tokens_attempt2: int
    max_prompt_length_attempt1: int
    max_prompt_length_attempt2: int
    generation_temperature: float
    seed: int
    checkpoint_every: int = 0    # save LoRA every N optimizer steps within a stage; 0 = end-of-stage only
    length_norm: str = "sequence"   # PG/KL length normalization. "sequence": divide by actual
                                    # generated length (per-sample mean — original). "constant":
                                    # divide by max_new_tokens (Dr.GRPO — avoids the length bias
                                    # where the policy games per-token gradient by inflating length).

    def __post_init__(self) -> None:
        if self.max_prompt_length_attempt2 < self.max_prompt_length_attempt1:
            raise ValueError(
                "max_prompt_length_attempt2 must be >= max_prompt_length_attempt1 "
                "(attempt 2 prompt = attempt 1 prompt + answer + self-correction)"
            )
        if self.length_norm not in ("sequence", "constant"):
            raise ValueError(
                f"length_norm must be 'sequence' or 'constant', got {self.length_norm!r}"
            )


@dataclass
class EvalConfig:
    batch_size: int = 1
    max_examples: int = 200
    generation_temperature: float = 0.0


@dataclass
class OutputConfig:
    dir: str = "outputs"


@dataclass
class SCoReConfig:
    run_name: str
    wandb_project: str
    model: ModelConfig
    dataset: DatasetConfig
    prompts: PromptsConfig
    reward: RewardConfig
    train: TrainConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str | Path) -> SCoReConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return SCoReConfig(
        run_name=raw["run_name"],
        wandb_project=raw["wandb_project"],
        model=ModelConfig(
            **{k: v for k, v in raw["model"].items() if k != "lora"},
            lora=LoRAConfig(**raw["model"]["lora"]),
        ),
        dataset=DatasetConfig(**raw["dataset"]),
        prompts=PromptsConfig(**raw["prompts"]),
        reward=RewardConfig(**raw["reward"]),
        train=TrainConfig(**raw["train"]),
        eval=EvalConfig(**raw.get("eval", {})),
        output=OutputConfig(**raw.get("output", {})),
    )


# --- Model and data ----------------------------------------------------------


def load_model_and_tokenizer(cfg: ModelConfig) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
        fast_inference=cfg.fast_inference,
        gpu_memory_utilization=cfg.fast_inference_gpu_mem if cfg.fast_inference else None,
        max_lora_rank=cfg.lora.r if cfg.fast_inference else None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        use_gradient_checkpointing=True,  # NOT "unsloth" — that mode offloads gradients to host RAM, saturates Spark's unified memory bandwidth and wedges sshd. In-VRAM checkpointing (True) leaves the host kernel responsive.
        random_state=3407,
    )
    if cfg.chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    return model, tokenizer


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def thinking_to_history(text: str) -> str:
    """Rewrite ``<think>...</think>`` blocks as plain-text reasoning notes.

    Qwen3's bundled chat template **silently strips** ``<think>...</think>``
    from prior assistant turns when re-rendering multi-turn history. SCoRe
    self-correction needs the corrector to see the reasoning that produced
    attempt 1, so we wrap the reasoning in plain-text markers (which the
    template won't strip) before appending attempt 1 as the assistant turn
    for attempt 2. No-op if no ``<think>`` tags are present.
    """
    return text #_THINK_BLOCK.sub(r"", text).strip()


def load_and_prepare_data(
    dataset_cfg: DatasetConfig,
    prompts: PromptsConfig,
    tokenizer: Any,
) -> tuple[Any, Any]:
    """Load dataset and pre-build chat-templated prompts.

    Two loading modes, picked by ``dataset_cfg.split``:

      - ``split is None`` (default): expects separate ``train`` and ``test``
        splits. The first ``test_split_size`` test examples become the held-out
        eval set; the rest is concatenated into train so little data is wasted.
      - ``split`` set: a single-split dataset (slicing allowed, e.g.
        ``"validation[:2000]"``). The first ``test_split_size`` examples become
        the held-out eval set; the remainder is train.

    Each output example carries:
      - text:     fully chat-templated prompt string ready for tokenization
      - target:   raw ground-truth string from `target_field`
      - messages: the messages list, used to append assistant + self-correction
    """
    load_kw = {"name": dataset_cfg.config_name, "revision": dataset_cfg.revision}
    if dataset_cfg.split is not None:
        full = load_dataset(dataset_cfg.name, split=dataset_cfg.split, **load_kw)
        if dataset_cfg.test_split_size >= len(full):
            raise ValueError(
                f"test_split_size={dataset_cfg.test_split_size} must be < "
                f"single-split dataset size {len(full)}"
            )
        train_dataset = full.select(range(dataset_cfg.test_split_size, len(full)))
        test_dataset = full.select(range(dataset_cfg.test_split_size))
    else:
        dataset = load_dataset(dataset_cfg.name, split="train", **load_kw)
        test_dataset = load_dataset(dataset_cfg.name, split="test", **load_kw)
        if dataset_cfg.test_split_size > len(test_dataset):
            raise ValueError(
                f"test_split_size={dataset_cfg.test_split_size} exceeds "
                f"test set size {len(test_dataset)}"
            )
        held_out = test_dataset.select(range(dataset_cfg.test_split_size))
        extra_train = test_dataset.select(
            range(dataset_cfg.test_split_size, len(test_dataset))
        )
        train_dataset = concatenate_datasets([dataset, extra_train])
        test_dataset = held_out

    def build(examples: dict[str, list]) -> dict[str, list]:
        messages = [
            [
                {"role": "system", "content": prompts.system},
                {"role": "user", "content": "Question:\n" + q},
            ]
            for q in examples[dataset_cfg.input_field]
        ]
        text = [
            tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages
        ]
        return {
            "text": text,
            "target": list(examples[dataset_cfg.target_field]),
            "messages": messages,
        }

    train_dataset = train_dataset.map(build, batched=True)
    test_dataset = test_dataset.map(build, batched=True)
    return train_dataset, test_dataset


# --- Log-prob and KL helpers --------------------------------------------------


def get_log_probs(model: Any, input_ids: Tensor, prompt_len: int) -> Tensor:
    """Per-token log probabilities of the *generated* tokens only.

    Returns ``[B, gen_len]`` — log π(y_t | y_<t, x) for each generated token y_t.

    Two memory tricks:
      1. ``logits_to_keep`` (when supported by the model's forward) makes the model
         skip computing logits for the prompt portion entirely. For long prompts
         this is a 4-10× reduction in the peak logits tensor.
      2. ``selective_log_softmax`` via ``gather - logsumexp``: we only need the
         log-prob of the SAMPLED token, not the full vocab distribution. So we
         compute ``logits[token] - logsumexp(logits, dim=-1)`` directly, never
         materializing the full ``[B, gen_len, V]`` log-softmax tensor.

    Together these eliminate the ``[B, T, V]`` tensor that triggered OOM on
    Stage 2 backward. KL is computed via the K3 estimator (see ``get_kl_div``)
    which also only needs per-token log-probs — no full vocab anywhere.
    """
    gen_len = input_ids.size(1) - prompt_len
    # Some HF models accept `logits_to_keep` to skip prompt-side logit compute.
    # We pass gen_len + 1 because the last logit of the sequence is the next-token
    # prediction (excluded), and we need gen_len log-probs aligned with targets.
    try:
        logits = model(input_ids, logits_to_keep=gen_len + 1).logits
    except TypeError:
        logits = model(input_ids).logits
    # Align logits with next-token targets: drop the last position (no target after EOS),
    # then slice to the generated span.
    logits = logits[:, -(gen_len + 1):-1, :]    # [B, gen_len, V]
    targets = input_ids[:, prompt_len:]         # [B, gen_len]
    # Selective log-softmax: log π(y_t) = logits[y_t] - logsumexp(logits, V)
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1)


def get_eos_mask(answer_ids: Tensor, tokenizer: Any) -> Tensor:
    """1 for tokens up to and including the first EOS, 0 after."""
    is_eos = answer_ids == tokenizer.eos_token_id
    return (torch.cumsum(is_eos, dim=1) <= 1).int()


def _length_normalize(per_token: Tensor, mask: Tensor, mode: str, max_new: int) -> Tensor:
    """Reduce a per-token ``[B, gen_len]`` quantity to per-sample ``[B]`` by summing
    over the masked span, then normalizing per ``mode``:

      - ``"sequence"``: divide by the actual masked length (per-sample mean). The
        original form — but it lets the policy game per-token gradient weight by
        inflating generation length.
      - ``"constant"``: divide by ``max_new`` (the generation cap). Dr.GRPO fix —
        per-token weight is fixed across samples regardless of rollout length.

    For SCoRe's two-attempt structure, ``max_new`` is the *per-attempt* cap
    (``max_new_tokens_attempt{1,2}``). When both caps are equal this is exactly
    vanilla Dr.GRPO; if they differ, each attempt's tokens are normalized by that
    attempt's own budget.
    """
    summed = (per_token * mask).sum(-1)
    if mode == "constant":
        return summed / max_new
    return summed / mask.sum(-1).clamp(min=1)


def get_kl_div(
    ref_log_probs: Tensor,
    policy_log_probs: Tensor,
    mask: Tensor,
    mode: str,
    max_new: int,
) -> Tensor:
    """K3 KL estimator (Schulman), length-normalized per ``mode`` (see ``_length_normalize``).

    ``K3 = exp(log π_ref - log π) - (log π_ref - log π) - 1``. Always ≥ 0, unbiased
    forward KL. Requires only per-token log-probs of sampled tokens — no ``[B, T, V]``.

    Both inputs are shape ``[B, gen_len]``. ``ref_log_probs`` is from the disabled-adapter
    forward (no grad); ``policy_log_probs`` carries the gradient.
    """
    diff = ref_log_probs - policy_log_probs
    kl_per_token = torch.exp(diff) - diff - 1
    return _length_normalize(kl_per_token, mask, mode, max_new)


# --- Rollout -----------------------------------------------------------------


@dataclass
class Rollout:
    action1_tokens: Tensor
    answer1_text: list[str]
    reward1: list[float]
    x1_len: int
    attempt1_answer_mask: Tensor
    action2_tokens: Tensor
    answer2_text: list[str]
    reward2: list[float]
    x2_len: int
    attempt2_answer_mask: Tensor

    @property
    def attempt1_answer_tokens(self) -> Tensor:
        return self.action1_tokens[:, self.x1_len:]

    @property
    def attempt2_answer_tokens(self) -> Tensor:
        return self.action2_tokens[:, self.x2_len:]


def two_attempt_rollout(
    model: Any,
    tokenizer: Any,
    batch: dict[str, list],
    cfg: SCoReConfig,
    *,
    temperature: float | None = None,
) -> Rollout:
    """Run both attempts, decode, score with the configured reward.

    Note: leaves the model in inference mode (`FastLanguageModel.for_inference`).
    Callers that train must restore it with `FastLanguageModel.for_training(model)`.
    """
    train_cfg = cfg.train
    if temperature is None:
        temperature = train_cfg.generation_temperature
    extractor = rf.ANSWER_EXTRACTORS[cfg.reward.answer_extractor]
    reward_fn = rf.REWARD_FNS[cfg.reward.fn]
    targets: list[str] = list(batch["target"])
    do_sample = temperature > 0.0
    gen_temp = temperature if do_sample else 1.0

    x1 = tokenizer(
        batch["text"],
        padding="longest",
        truncation=True,
        max_length=train_cfg.max_prompt_length_attempt1,
        padding_side="left",
        return_tensors="pt",
    ).to(model.device)
    x1_len = x1["input_ids"].shape[1]

    FastLanguageModel.for_inference(model)
    action1_tokens = model.generate(
        x1["input_ids"],
        attention_mask=x1["attention_mask"],
        max_new_tokens=train_cfg.max_new_tokens_attempt1,
        do_sample=do_sample,
        temperature=gen_temp,
    )
    attempt1_answer_tokens = action1_tokens[:, x1_len:]
    attempt1_answer_mask = get_eos_mask(attempt1_answer_tokens, tokenizer)
    answer1_text = tokenizer.batch_decode(attempt1_answer_tokens, skip_special_tokens=True)
    reward1 = reward_fn(answer1_text, targets, extractor)

    messages = [list(m) for m in batch["messages"]]
    for m, a in zip(messages, answer1_text):
        # Convert <think>...</think> to plain-text [Prior reasoning: ...] so the
        # chat template doesn't strip it from history (Qwen3 does this by default).
        m.append({"role": "assistant", "content": thinking_to_history(a)})
        m.append({"role": "user", "content": cfg.prompts.self_correction})
    init_x2 = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in messages
    ]
    x2 = tokenizer(
        init_x2,
        padding="longest",
        truncation=True,
        max_length=train_cfg.max_prompt_length_attempt2,
        padding_side="left",
        return_tensors="pt",
    ).to(model.device)
    x2_len = x2["input_ids"].shape[1]

    action2_tokens = model.generate(
        x2["input_ids"],
        attention_mask=x2["attention_mask"],
        max_new_tokens=train_cfg.max_new_tokens_attempt2,
        do_sample=do_sample,
        temperature=gen_temp,
    )
    attempt2_answer_tokens = action2_tokens[:, x2_len:]
    attempt2_answer_mask = get_eos_mask(attempt2_answer_tokens, tokenizer)
    answer2_text = tokenizer.batch_decode(attempt2_answer_tokens, skip_special_tokens=True)
    reward2 = reward_fn(answer2_text, targets, extractor)

    return Rollout(
        action1_tokens=action1_tokens,
        answer1_text=answer1_text,
        reward1=reward1,
        x1_len=x1_len,
        attempt1_answer_mask=attempt1_answer_mask,
        action2_tokens=action2_tokens,
        answer2_text=answer2_text,
        reward2=reward2,
        x2_len=x2_len,
        attempt2_answer_mask=attempt2_answer_mask,
    )


def _masked_norm_logp(log_probs: Tensor, mask: Tensor, mode: str, max_new: int) -> Tensor:
    """Per-sample log-prob of the generated span, length-normalized per ``mode``
    (see ``_length_normalize``). Used to weight the policy-gradient term."""
    return _length_normalize(log_probs, mask, mode, max_new)


def _log_rollouts_table(
    stage: str,
    batch: dict[str, list],
    roll: Rollout,
    max_new1: int,
    max_new2: int,
    k: int = 4,
) -> None:
    """Log up to k rollouts as a wandb Table for debugging generation quality.

    Per-step scalars (reward, KL, length) tell you *that* something changed.
    A few raw samples tell you *what* — whether the model is truncating before
    `<answer>`, breaking format, or actually self-correcting. The truncation
    columns let you sort by ``trunc1=True`` to see what the model produces
    when it runs out of token budget.

    ``problem_tail`` shows only the last ~1200 chars of the (chat-templated)
    prompt — enough to see the question without flooding the cell with the
    system prompt.
    """
    n = min(k, len(roll.answer1_text))
    if n == 0:
        return
    lens1 = roll.attempt1_answer_mask.sum(-1).cpu().tolist()
    lens2 = roll.attempt2_answer_mask.sum(-1).cpu().tolist()
    targets = list(batch["target"])
    table = wandb.Table(columns=[
        "problem_tail", "target",
        "attempt1", "attempt2",
        "reward1", "reward2",
        "len1", "len2", "trunc1", "trunc2",
    ])
    for i in range(n):
        prompt = batch["text"][i]
        table.add_data(
            prompt[-1200:] if len(prompt) > 1200 else prompt,
            targets[i],
            roll.answer1_text[i],
            roll.answer2_text[i],
            float(roll.reward1[i]),
            float(roll.reward2[i]),
            int(lens1[i]),
            int(lens2[i]),
            bool(lens1[i] >= max_new1),
            bool(lens2[i] >= max_new2),
        )
    wandb.log({f"{stage}/rollouts": table})


def _length_stats(mask: Tensor, max_new: int) -> tuple[dict[str, float], np.ndarray]:
    """Per-sample generated lengths from the EOS mask.

    The mask is 1 up-to-and-including the first EOS, 0 after, so ``mask.sum(-1)``
    is the per-sample generation length. ``len >= max_new`` means the sample
    never produced an EOS and was truncated at the cap — track this fraction
    to right-size ``max_new_tokens_*`` in the config.

    Returns ``(scalar_stats_dict, lens_array)``; caller decides whether to log
    the array as a histogram.
    """
    lens = mask.sum(-1).cpu().numpy()
    stats = {
        "len_mean": float(lens.mean()),
        "len_p95": float(np.percentile(lens, 95)),
        "len_max": float(lens.max()),
        "truncated_frac": float(np.mean(lens >= max_new)),
    }
    return stats, lens


# --- Stage I -----------------------------------------------------------------


def train_stage_1(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    cfg: SCoReConfig,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path | None = None,
) -> Any:
    """Stage I per Eq. 3: PG on log pi(y2) * r(y2), KL anchor on first attempt.

    If ``checkpoint_dir`` is provided and ``cfg.train.checkpoint_every > 0``, the
    LoRA adapter is saved to ``checkpoint_dir / step_{N}/`` every N optimizer
    steps (cumulative across epochs). The end-of-stage save is the caller's job.
    """
    train_cfg = cfg.train
    step_count = 0
    for epoch in range(train_cfg.stage1_epochs):
        loader = train_dataset.iter(batch_size=train_cfg.stage1_batch_size)
        for batch in tqdm(loader, desc=f"Stage 1 epoch {epoch + 1}/{train_cfg.stage1_epochs}"):
            roll = two_attempt_rollout(model, tokenizer, batch, cfg)

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # .clone() strips the inference-mode flag inherited from the rollout
                # (FastLanguageModel.for_inference). Without it, get_log_probs's forward
                # produces inference tensors that can't enter autograd:
                # "Inference tensors cannot be saved for backward."
                action1 = roll.action1_tokens.clone()
                action2 = roll.action2_tokens.clone()

                with torch.no_grad(), model.disable_adapter():
                    base_logp_a1 = get_log_probs(model, action1, roll.x1_len)

                logp_a1 = get_log_probs(model, action1, roll.x1_len)
                kl_div = get_kl_div(
                    base_logp_a1, logp_a1, roll.attempt1_answer_mask,
                    mode=train_cfg.length_norm,
                    max_new=train_cfg.max_new_tokens_attempt1,
                )

                logp_a2 = get_log_probs(model, action2, roll.x2_len)
                reward2_t = torch.tensor(roll.reward2, device=model.device, dtype=logp_a2.dtype)
                pg_term = _masked_norm_logp(
                    logp_a2, roll.attempt2_answer_mask,
                    mode=train_cfg.length_norm,
                    max_new=train_cfg.max_new_tokens_attempt2,
                ) * reward2_t

                loss = (-pg_term + train_cfg.beta2 * kl_div).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            a1_stats, a1_lens = _length_stats(roll.attempt1_answer_mask, train_cfg.max_new_tokens_attempt1)
            a2_stats, a2_lens = _length_stats(roll.attempt2_answer_mask, train_cfg.max_new_tokens_attempt2)
            wandb.log({
                "stage1/reward_attempt1": float(np.mean(roll.reward1)),
                "stage1/reward_attempt2": float(np.mean(roll.reward2)),
                "stage1/diff_attempt2_minus_attempt1": float(
                    np.mean(roll.reward2) - np.mean(roll.reward1)
                ),
                "stage1/loss": float(loss.item()),
                "stage1/kl_div": float(kl_div.mean().item()),
                **{f"stage1/attempt1_{k}": v for k, v in a1_stats.items()},
                **{f"stage1/attempt2_{k}": v for k, v in a2_stats.items()},
                "stage1/attempt1_lens_hist": wandb.Histogram(a1_lens),
                "stage1/attempt2_lens_hist": wandb.Histogram(a2_lens),
            })
            _log_rollouts_table(
                "stage1", batch, roll,
                train_cfg.max_new_tokens_attempt1, train_cfg.max_new_tokens_attempt2,
            )

            step_count += 1
            if (
                checkpoint_dir is not None
                and train_cfg.checkpoint_every > 0
                and step_count % train_cfg.checkpoint_every == 0
            ):
                save_adapter(model, tokenizer, checkpoint_dir / f"step_{step_count}")

    return model


# --- Stage II ----------------------------------------------------------------


def train_stage_2(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    cfg: SCoReConfig,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path | None = None,
) -> Any:
    """Stage II per Eq. 4 + Eq. 5: PG on both attempts with reward shaping bonus on r(y2).

    Mid-stage adapter checkpointing follows the same scheme as ``train_stage_1``.
    """
    train_cfg = cfg.train
    step_count = 0
    for epoch in range(train_cfg.stage2_epochs):
        loader = train_dataset.iter(batch_size=train_cfg.stage2_batch_size)
        for batch in tqdm(loader, desc=f"Stage 2 epoch {epoch + 1}/{train_cfg.stage2_epochs}"):
            roll = two_attempt_rollout(model, tokenizer, batch, cfg)

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # .clone() strips the inference-mode flag inherited from the rollout
                # (FastLanguageModel.for_inference). Without it, get_log_probs's forward
                # produces inference tensors that can't enter autograd:
                # "Inference tensors cannot be saved for backward."
                action1 = roll.action1_tokens.clone()
                action2 = roll.action2_tokens.clone()

                with torch.no_grad(), model.disable_adapter():
                    base_logp_a1 = get_log_probs(model, action1, roll.x1_len)
                logp_a1 = get_log_probs(model, action1, roll.x1_len)
                kl_1 = get_kl_div(
                    base_logp_a1, logp_a1, roll.attempt1_answer_mask,
                    mode=train_cfg.length_norm,
                    max_new=train_cfg.max_new_tokens_attempt1,
                )
                reward1_t = torch.tensor(roll.reward1, device=model.device, dtype=logp_a1.dtype)
                pg_attempt1 = _masked_norm_logp(
                    logp_a1, roll.attempt1_answer_mask,
                    mode=train_cfg.length_norm,
                    max_new=train_cfg.max_new_tokens_attempt1,
                ) * reward1_t

                with torch.no_grad(), model.disable_adapter():
                    base_logp_a2 = get_log_probs(model, action2, roll.x2_len)
                logp_a2 = get_log_probs(model, action2, roll.x2_len)
                kl_2 = get_kl_div(
                    base_logp_a2, logp_a2, roll.attempt2_answer_mask,
                    mode=train_cfg.length_norm,
                    max_new=train_cfg.max_new_tokens_attempt2,
                )
                reward2_t = torch.tensor(roll.reward2, device=model.device, dtype=logp_a2.dtype)
                shaped_reward2 = reward2_t + train_cfg.alpha * (reward2_t - reward1_t)
                pg_attempt2 = _masked_norm_logp(
                    logp_a2, roll.attempt2_answer_mask,
                    mode=train_cfg.length_norm,
                    max_new=train_cfg.max_new_tokens_attempt2,
                ) * shaped_reward2

                loss = (
                    -pg_attempt1 - pg_attempt2
                    + train_cfg.beta1 * kl_1
                    + train_cfg.beta1 * kl_2
                ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            a1_stats, a1_lens = _length_stats(roll.attempt1_answer_mask, train_cfg.max_new_tokens_attempt1)
            a2_stats, a2_lens = _length_stats(roll.attempt2_answer_mask, train_cfg.max_new_tokens_attempt2)
            wandb.log({
                "stage2/reward_attempt1": float(np.mean(roll.reward1)),
                "stage2/reward_attempt2": float(np.mean(roll.reward2)),
                "stage2/diff_attempt2_minus_attempt1": float(
                    np.mean(roll.reward2) - np.mean(roll.reward1)
                ),
                "stage2/loss": float(loss.item()),
                "stage2/kl_div_attempt1": float(kl_1.mean().item()),
                "stage2/kl_div_attempt2": float(kl_2.mean().item()),
                **{f"stage2/attempt1_{k}": v for k, v in a1_stats.items()},
                **{f"stage2/attempt2_{k}": v for k, v in a2_stats.items()},
                "stage2/attempt1_lens_hist": wandb.Histogram(a1_lens),
                "stage2/attempt2_lens_hist": wandb.Histogram(a2_lens),
            })
            _log_rollouts_table(
                "stage2", batch, roll,
                train_cfg.max_new_tokens_attempt1, train_cfg.max_new_tokens_attempt2,
            )

            step_count += 1
            if (
                checkpoint_dir is not None
                and train_cfg.checkpoint_every > 0
                and step_count % train_cfg.checkpoint_every == 0
            ):
                save_adapter(model, tokenizer, checkpoint_dir / f"step_{step_count}")

    return model


# --- Eval and save -----------------------------------------------------------


def evaluate(
    model: Any, tokenizer: Any, test_dataset: Any, cfg: SCoReConfig
) -> dict[str, float]:
    """Two-attempt rollout over the test set at `cfg.eval.generation_temperature` (defaults to greedy)."""
    n = min(len(test_dataset), cfg.eval.max_examples)
    if n == 0:
        return {"acc_attempt1": 0.0, "acc_attempt2": 0.0, "delta": 0.0}
    subset = test_dataset.select(range(n))
    rewards1: list[float] = []
    rewards2: list[float] = []
    lens1: list[int] = []
    lens2: list[int] = []
    loader = subset.iter(batch_size=cfg.eval.batch_size)
    first_batch = True
    for batch in tqdm(loader, desc="Eval", total=(n + cfg.eval.batch_size - 1) // cfg.eval.batch_size):
        roll = two_attempt_rollout(
            model, tokenizer, batch, cfg, temperature=cfg.eval.generation_temperature
        )
        rewards1.extend(roll.reward1)
        rewards2.extend(roll.reward2)
        lens1.extend(roll.attempt1_answer_mask.sum(-1).cpu().tolist())
        lens2.extend(roll.attempt2_answer_mask.sum(-1).cpu().tolist())
        if first_batch:
            _log_rollouts_table(
                "eval", batch, roll,
                cfg.train.max_new_tokens_attempt1, cfg.train.max_new_tokens_attempt2,
                k=8,
            )
            first_batch = False

    lens1_arr = np.array(lens1)
    lens2_arr = np.array(lens2)
    metrics: dict[str, float] = {
        "acc_attempt1": float(np.mean(rewards1)),
        "acc_attempt2": float(np.mean(rewards2)),
        "attempt1_len_mean": float(lens1_arr.mean()),
        "attempt1_len_p95": float(np.percentile(lens1_arr, 95)),
        "attempt1_len_max": float(lens1_arr.max()),
        "attempt1_truncated_frac": float(np.mean(lens1_arr >= cfg.train.max_new_tokens_attempt1)),
        "attempt2_len_mean": float(lens2_arr.mean()),
        "attempt2_len_p95": float(np.percentile(lens2_arr, 95)),
        "attempt2_len_max": float(lens2_arr.max()),
        "attempt2_truncated_frac": float(np.mean(lens2_arr >= cfg.train.max_new_tokens_attempt2)),
    }
    metrics["delta"] = metrics["acc_attempt2"] - metrics["acc_attempt1"]
    return metrics


def save_adapter(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))


# --- Main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Smoke run: limit train to N examples and cap eval.max_examples at N. "
             "Use to verify the loop end-to-end before a full run.",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="1 (default): full run, train both stages. "
             "2: skip stage 1 entirely; requires --resume-adapter pointing at a saved LoRA.",
    )
    parser.add_argument(
        "--resume-adapter",
        type=str,
        default=None,
        help="Path to a saved LoRA adapter directory (output of save_adapter). "
             "Loaded into the 'default' adapter slot after model init. "
             "Required when --start-stage 2.",
    )
    args = parser.parse_args()

    if args.start_stage == 2 and args.resume_adapter is None:
        parser.error("--resume-adapter PATH is required when --start-stage 2")

    cfg = load_config(args.config)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)

    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=_flatten(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg.model)

    if args.resume_adapter is not None:
        adapter_path = Path(args.resume_adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"--resume-adapter path does not exist: {adapter_path}")
        print(f"Loading adapter weights from {adapter_path} into 'default' slot")
        # Overwrites the freshly-initialized LoRA from get_peft_model with saved weights.
        model.load_adapter(str(adapter_path), adapter_name="default", is_trainable=True)

    train_dataset, test_dataset = load_and_prepare_data(cfg.dataset, cfg.prompts, tokenizer)

    if args.smoke > 0:
        n_train = min(len(train_dataset), args.smoke)
        train_dataset = train_dataset.select(range(n_train))
        cfg.eval.max_examples = min(cfg.eval.max_examples, args.smoke)
        print(
            f"SMOKE RUN: train subset to {n_train} examples, "
            f"eval cap to {cfg.eval.max_examples}"
        )

    # bitsandbytes 8-bit AdamW: ~4× smaller optimizer state vs fp32 Adam. Same convergence
    # behavior on LoRA finetuning per the Unsloth/TRL reference RL notebooks.
    from bitsandbytes.optim import AdamW8bit
    optimizer = AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate)

    out_root = Path(cfg.output.dir) / cfg.run_name

    # Pretrain eval = eval before any training in *this run*. If resuming from a
    # stage-1 adapter, this reflects the loaded checkpoint's accuracy, which is
    # the right baseline to compare stage 2's effect against.
    wandb.log({f"eval_pretrain/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})

    if args.start_stage <= 1:
        model = train_stage_1(
            model, tokenizer, train_dataset, cfg, optimizer,
            checkpoint_dir=out_root / "stage1",
        )
        wandb.log({f"eval_after_stage1/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})
        save_adapter(model, tokenizer, out_root / "stage1")
    else:
        print("Skipping stage 1 (--start-stage 2)")

    model = train_stage_2(
        model, tokenizer, train_dataset, cfg, optimizer,
        checkpoint_dir=out_root / "stage2",
    )
    wandb.log({f"eval_after_stage2/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})
    save_adapter(model, tokenizer, out_root / "stage2")


def _flatten(cfg: SCoReConfig) -> dict[str, Any]:
    """Flatten nested dataclasses into a single dict for W&B config logging.

    Recurses only into nested dataclasses; lists/dicts are logged as-is.
    """
    out: dict[str, Any] = {}

    def walk(prefix: str, obj: Any) -> None:
        if hasattr(obj, "__dataclass_fields__"):
            for k in obj.__dataclass_fields__:
                walk(f"{prefix}.{k}" if prefix else k, getattr(obj, k))
        else:
            out[prefix] = obj

    walk("", cfg)
    return out


if __name__ == "__main__":
    main()
