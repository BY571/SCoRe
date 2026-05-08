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
    batch_size: int
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

    def __post_init__(self) -> None:
        if self.max_prompt_length_attempt2 < self.max_prompt_length_attempt1:
            raise ValueError(
                "max_prompt_length_attempt2 must be >= max_prompt_length_attempt1 "
                "(attempt 2 prompt = attempt 1 prompt + answer + self-correction)"
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
        use_gradient_checkpointing=True,  # was "unsloth" — that mode offloads gradients to RAM and wedges Spark's unified memory
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
    return _THINK_BLOCK.sub(r"[Prior reasoning: \1]", text).strip()


def load_and_prepare_data(
    dataset_cfg: DatasetConfig,
    prompts: PromptsConfig,
    tokenizer: Any,
) -> tuple[Any, Any]:
    """Load dataset and pre-build chat-templated prompts.

    The test split is partitioned: the first `test_split_size` examples become the
    held-out eval set; the remainder is concatenated into train so that little of
    the available data is wasted on eval. Each output example carries:
      - text:     fully chat-templated prompt string ready for tokenization
      - target:   raw ground-truth string from `target_field`
      - messages: the messages list, used to append assistant + self-correction
    """
    dataset = load_dataset(dataset_cfg.name, split="train")
    test_dataset = load_dataset(dataset_cfg.name, split="test")
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


def get_log_probs(
    model: Any, input_ids: Tensor, prompt_len: int
) -> tuple[Tensor, Tensor]:
    """Per-token + per-vocab log probs over the *generated* portion only.

    Slices the prompt off and shifts logits by one so they align with the next-token
    targets. Returns ``(token_log_probs, full_log_probs)`` both shaped over the
    generated span: ``[B, gen_len]`` and ``[B, gen_len, V]``.

    Crucially: slices logits to the generated span *before* ``log_softmax`` to
    avoid materializing a ``[B, full_seq, V]`` log-prob tensor — that tripped
    OOM on long attempt-2 prompts (~6K tokens) at batch=8.
    """
    logits = model(input_ids).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    gen_logits = logits[:, prompt_len - 1:, :]
    gen_targets = targets[:, prompt_len - 1:]
    full_log_probs = gen_logits.log_softmax(dim=-1)
    token_log_probs = full_log_probs.gather(2, gen_targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, full_log_probs


def get_eos_mask(answer_ids: Tensor, tokenizer: Any) -> Tensor:
    """1 for tokens up to and including the first EOS, 0 after."""
    is_eos = answer_ids == tokenizer.eos_token_id
    return (torch.cumsum(is_eos, dim=1) <= 1).int()


def get_kl_div(base_log_probs: Tensor, log_probs: Tensor, mask: Tensor) -> Tensor:
    """Forward KL ``D_KL(pi_theta || pi_ref)`` summed over vocab and answer-masked.

    Computed as ``sum_v exp(log_pi) * (log_pi - log_pi_ref)``. ``base_log_probs``
    is from the disabled-adapter forward pass and carries no gradient; gradient
    flows back to the policy through ``log_probs``.
    """
    kl = (log_probs.exp() * (log_probs - base_log_probs)).sum(-1)
    return (kl * mask).sum(-1) / mask.sum(-1).clamp(min=1)


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
        padding="max_length",
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
        padding="max_length",
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


def _masked_mean_logp(log_probs: Tensor, mask: Tensor) -> Tensor:
    return (log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)


# --- Stage I -----------------------------------------------------------------


def train_stage_1(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    cfg: SCoReConfig,
    optimizer: torch.optim.Optimizer,
) -> Any:
    """Stage I per Eq. 3: PG on log pi(y2) * r(y2), KL anchor on first attempt."""
    train_cfg = cfg.train
    for epoch in range(train_cfg.stage1_epochs):
        loader = train_dataset.iter(batch_size=train_cfg.batch_size)
        for batch in tqdm(loader, desc=f"Stage 1 epoch {epoch + 1}/{train_cfg.stage1_epochs}"):
            roll = two_attempt_rollout(model, tokenizer, batch, cfg)

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action1 = roll.action1_tokens.clone()
                action2 = roll.action2_tokens.clone()

                with torch.no_grad(), model.disable_adapter():
                    _, base_full_logp = get_log_probs(model, action1, roll.x1_len)

                _, full_logp_a1 = get_log_probs(model, action1, roll.x1_len)
                kl_div = get_kl_div(base_full_logp, full_logp_a1, roll.attempt1_answer_mask)

                logp_a2, _ = get_log_probs(model, action2, roll.x2_len)
                reward2_t = torch.tensor(roll.reward2, device=model.device, dtype=logp_a2.dtype)
                pg_term = _masked_mean_logp(logp_a2, roll.attempt2_answer_mask) * reward2_t

                loss = (-pg_term + train_cfg.beta2 * kl_div).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "stage1/reward_attempt1": float(np.mean(roll.reward1)),
                "stage1/reward_attempt2": float(np.mean(roll.reward2)),
                "stage1/diff_attempt2_minus_attempt1": float(
                    np.mean(roll.reward2) - np.mean(roll.reward1)
                ),
                "stage1/loss": float(loss.item()),
                "stage1/kl_div": float(kl_div.mean().item()),
            })

    return model


# --- Stage II ----------------------------------------------------------------


def train_stage_2(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    cfg: SCoReConfig,
    optimizer: torch.optim.Optimizer,
) -> Any:
    """Stage II per Eq. 4 + Eq. 5: PG on both attempts with reward shaping bonus on r(y2)."""
    train_cfg = cfg.train
    for epoch in range(train_cfg.stage2_epochs):
        loader = train_dataset.iter(batch_size=train_cfg.batch_size)
        for batch in tqdm(loader, desc=f"Stage 2 epoch {epoch + 1}/{train_cfg.stage2_epochs}"):
            roll = two_attempt_rollout(model, tokenizer, batch, cfg)

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action1 = roll.action1_tokens.clone()
                action2 = roll.action2_tokens.clone()

                with torch.no_grad(), model.disable_adapter():
                    _, base_full_logp_1 = get_log_probs(model, action1, roll.x1_len)
                logp_a1, full_logp_a1 = get_log_probs(model, action1, roll.x1_len)
                kl_1 = get_kl_div(base_full_logp_1, full_logp_a1, roll.attempt1_answer_mask)
                reward1_t = torch.tensor(roll.reward1, device=model.device, dtype=logp_a1.dtype)
                pg_attempt1 = _masked_mean_logp(logp_a1, roll.attempt1_answer_mask) * reward1_t

                with torch.no_grad(), model.disable_adapter():
                    _, base_full_logp_2 = get_log_probs(model, action2, roll.x2_len)
                logp_a2, full_logp_a2 = get_log_probs(model, action2, roll.x2_len)
                kl_2 = get_kl_div(base_full_logp_2, full_logp_a2, roll.attempt2_answer_mask)
                reward2_t = torch.tensor(roll.reward2, device=model.device, dtype=logp_a2.dtype)
                shaped_reward2 = reward2_t + train_cfg.alpha * (reward2_t - reward1_t)
                pg_attempt2 = _masked_mean_logp(logp_a2, roll.attempt2_answer_mask) * shaped_reward2

                loss = (
                    -pg_attempt1 - pg_attempt2
                    + train_cfg.beta1 * kl_1
                    + train_cfg.beta1 * kl_2
                ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "stage2/reward_attempt1": float(np.mean(roll.reward1)),
                "stage2/reward_attempt2": float(np.mean(roll.reward2)),
                "stage2/diff_attempt2_minus_attempt1": float(
                    np.mean(roll.reward2) - np.mean(roll.reward1)
                ),
                "stage2/loss": float(loss.item()),
                "stage2/kl_div_attempt1": float(kl_1.mean().item()),
                "stage2/kl_div_attempt2": float(kl_2.mean().item()),
            })

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
    loader = subset.iter(batch_size=cfg.eval.batch_size)
    for batch in tqdm(loader, desc="Eval", total=(n + cfg.eval.batch_size - 1) // cfg.eval.batch_size):
        roll = two_attempt_rollout(
            model, tokenizer, batch, cfg, temperature=cfg.eval.generation_temperature
        )
        rewards1.extend(roll.reward1)
        rewards2.extend(roll.reward2)
    metrics = {
        "acc_attempt1": float(np.mean(rewards1)),
        "acc_attempt2": float(np.mean(rewards2)),
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)

    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=_flatten(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    train_dataset, test_dataset = load_and_prepare_data(cfg.dataset, cfg.prompts, tokenizer)

    if args.smoke > 0:
        n_train = min(len(train_dataset), args.smoke)
        train_dataset = train_dataset.select(range(n_train))
        cfg.eval.max_examples = min(cfg.eval.max_examples, args.smoke)
        print(
            f"SMOKE RUN: train subset to {n_train} examples, "
            f"eval cap to {cfg.eval.max_examples}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    out_root = Path(cfg.output.dir) / cfg.run_name

    wandb.log({f"eval_pretrain/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})

    model = train_stage_1(model, tokenizer, train_dataset, cfg, optimizer)
    wandb.log({f"eval_after_stage1/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})
    save_adapter(model, tokenizer, out_root / "stage1")

    model = train_stage_2(model, tokenizer, train_dataset, cfg, optimizer)
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
