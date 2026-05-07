"""Task-agnostic SCoRe training entry point.

Run with:
    python train.py --config configs/math.yaml

The script never references a specific task. Everything task-specific lives
in the YAML (model, dataset fields, prompts) and the reward registry.

Algorithm reference: Kumar et al., "Training Language Models to Self-Correct
via Reinforcement Learning" (arXiv:2409.12917).

  Stage I:  L1 = -E[r(y2) * log pi(y2 | y1, x)]
                 + beta2 * KL(pi(y1 | x) || pi_ref(y1 | x))
  Stage II: L2 = -E[r(y1) * log pi(y1 | x) + r2_total * log pi(y2 | y1, x)]
                 + beta1 * (KL_1 + KL_2)
            where r2_total = r(y2) + alpha * (r(y2) - r(y1)).

Reference policy: same model with the LoRA adapter disabled
(`with model.disable_adapter():`).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from datasets import concatenate_datasets, load_dataset
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


@dataclass
class EvalConfig:
    enabled: bool = True
    batch_size: int = 1
    max_examples: int = 200
    generation_temperature: float = 0.0


@dataclass
class OutputConfig:
    dir: str = "outputs"
    save_after_stage1: bool = True
    save_after_stage2: bool = True


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
    with open(path) as f:
        raw = yaml.safe_load(f)
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


def load_model_and_tokenizer(cfg: ModelConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    return model, tokenizer


def load_and_prepare_data(
    dataset_cfg: DatasetConfig,
    prompts: PromptsConfig,
    tokenizer,
):
    """Load dataset and pre-build chat-templated prompts.

    Returns (train_dataset, test_dataset). Each example has:
      - text:     fully chat-templated prompt string ready for tokenization
      - target:   raw ground-truth string from `target_field`
      - messages: the messages list, used to append assistant + self-correction
    """
    dataset = load_dataset(dataset_cfg.name, split="train")
    test_dataset = load_dataset(dataset_cfg.name, split="test")

    held_out = test_dataset.select(range(dataset_cfg.test_split_size))
    extra_train = test_dataset.select(
        range(dataset_cfg.test_split_size, len(test_dataset))
    )
    train_dataset = concatenate_datasets([dataset, extra_train])
    test_dataset = held_out

    def build(examples):
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


def get_log_probs(model, input_ids, prompt_len, return_probs=False):
    """Per-token log probs for the *generated* portion only.

    Slices off the prompt and shifts logits so they align with the next-token
    targets. Loops over the batch dim to keep peak memory manageable.
    """
    logits = model(input_ids).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    per_token_logps = []
    full_log_probs = []
    for logits_row, target_row in zip(logits, targets):
        log_probs = logits_row.log_softmax(dim=-1)
        if return_probs:
            full_log_probs.append(log_probs)
        token_lp = torch.gather(log_probs, dim=1, index=target_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_lp)
    gen_logps = torch.stack(per_token_logps)[:, prompt_len - 1:]
    if not return_probs:
        return gen_logps
    return gen_logps, torch.stack(full_log_probs)[:, prompt_len - 1:]


def get_eos_mask(answer_ids, tokenizer):
    """1 for tokens up to and including the first EOS, 0 after."""
    is_eos = answer_ids == tokenizer.eos_token_id
    return (torch.cumsum(is_eos, dim=1) <= 1).int()


def get_kl_div(base_log_probs, log_probs, mask):
    """Token-level forward KL averaged over the answer span."""
    kl = F.kl_div(log_probs, base_log_probs, reduction="none", log_target=True)
    return (kl.mean(-1) * mask).sum(-1) / mask.sum(-1).clamp(min=1)


# --- Rollout -----------------------------------------------------------------


@dataclass
class Rollout:
    action1_tokens: torch.Tensor
    attempt1_answer_tokens: torch.Tensor
    attempt1_answer_mask: torch.Tensor
    answer1_text: list[str]
    reward1: list[float]
    x1_len: int
    action2_tokens: torch.Tensor
    attempt2_answer_tokens: torch.Tensor
    attempt2_answer_mask: torch.Tensor
    answer2_text: list[str]
    reward2: list[float]
    x2_len: int


def two_attempt_rollout(model, tokenizer, batch, cfg: SCoReConfig, *, temperature: float | None = None) -> Rollout:
    train_cfg = cfg.train
    if temperature is None:
        temperature = train_cfg.generation_temperature
    extractor = rf.ANSWER_EXTRACTORS[cfg.reward.answer_extractor]
    reward_fn = rf.REWARD_FNS[cfg.reward.fn]

    targets = list(batch["target"])

    x1 = tokenizer.batch_encode_plus(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=train_cfg.max_prompt_length_attempt1,
        padding_side="left",
        return_tensors="pt",
    ).to(model.device)
    x1_len = x1["input_ids"].shape[1]

    FastLanguageModel.for_inference(model)
    do_sample = temperature > 0.0
    action1_tokens = model.generate(
        x1["input_ids"],
        attention_mask=x1["attention_mask"],
        max_new_tokens=train_cfg.max_new_tokens_attempt1,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
    )
    attempt1_answer_tokens = action1_tokens[:, x1_len:]
    attempt1_answer_mask = get_eos_mask(attempt1_answer_tokens, tokenizer)
    answer1_text = tokenizer.batch_decode(attempt1_answer_tokens, skip_special_tokens=True)
    reward1 = reward_fn(
        [extractor(a) for a in answer1_text],
        [extractor(t) for t in targets],
    )

    messages = [list(m) for m in batch["messages"]]
    for m, a in zip(messages, answer1_text):
        m.append({"role": "assistant", "content": a})
        m.append({"role": "user", "content": cfg.prompts.self_correction})
    init_x2 = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in messages
    ]
    x2 = tokenizer.batch_encode_plus(
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
        temperature=temperature if do_sample else 1.0,
    )
    attempt2_answer_tokens = action2_tokens[:, x2_len:]
    attempt2_answer_mask = get_eos_mask(attempt2_answer_tokens, tokenizer)
    answer2_text = tokenizer.batch_decode(attempt2_answer_tokens, skip_special_tokens=True)
    reward2 = reward_fn(
        [extractor(a) for a in answer2_text],
        [extractor(t) for t in targets],
    )

    return Rollout(
        action1_tokens=action1_tokens,
        attempt1_answer_tokens=attempt1_answer_tokens,
        attempt1_answer_mask=attempt1_answer_mask,
        answer1_text=answer1_text,
        reward1=reward1,
        x1_len=x1_len,
        action2_tokens=action2_tokens,
        attempt2_answer_tokens=attempt2_answer_tokens,
        attempt2_answer_mask=attempt2_answer_mask,
        answer2_text=answer2_text,
        reward2=reward2,
        x2_len=x2_len,
    )


def _masked_mean_logp(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)


# --- Stage I -----------------------------------------------------------------


def train_stage_1(model, tokenizer, train_dataset, cfg: SCoReConfig, optimizer):
    train_cfg = cfg.train
    for epoch in range(train_cfg.stage1_epochs):
        loader = train_dataset.iter(batch_size=train_cfg.batch_size)
        epoch_loss = []
        rewards1, rewards2 = [], []
        for batch in tqdm(loader, desc=f"Stage 1 epoch {epoch + 1}/{train_cfg.stage1_epochs}"):
            roll = two_attempt_rollout(model, tokenizer, batch, cfg)
            rewards1.append(np.mean(roll.reward1))
            rewards2.append(np.mean(roll.reward2))

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action1 = roll.action1_tokens.clone()
                action2 = roll.action2_tokens.clone()

                with torch.no_grad(), model.disable_adapter():
                    _, base_full_logp = get_log_probs(
                        model, action1, roll.x1_len, return_probs=True
                    )

                _, full_logp_a1 = get_log_probs(
                    model, action1, roll.x1_len, return_probs=True
                )
                kl_div = get_kl_div(base_full_logp, full_logp_a1.detach(), roll.attempt1_answer_mask)

                logp_a2 = get_log_probs(model, action2, roll.x2_len)
                reward2_t = torch.tensor(roll.reward2, device=model.device, dtype=logp_a2.dtype)
                pg_term = _masked_mean_logp(logp_a2, roll.attempt2_answer_mask) * reward2_t

                loss = (-pg_term + train_cfg.beta2 * kl_div).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            wandb.log({
                "stage1/mean_reward_attempt1": float(np.mean(rewards1)),
                "stage1/mean_reward_attempt2": float(np.mean(rewards2)),
                "stage1/diff_attempt2_minus_attempt1": float(np.mean(rewards2) - np.mean(rewards1)),
                "stage1/loss": float(np.mean(epoch_loss)),
                "stage1/kl_div": float(kl_div.mean().item()),
            })

    return model


# --- Stage II ----------------------------------------------------------------


def train_stage_2(model, tokenizer, train_dataset, cfg: SCoReConfig, optimizer):
    train_cfg = cfg.train
    for epoch in range(train_cfg.stage2_epochs):
        loader = train_dataset.iter(batch_size=train_cfg.batch_size)
        epoch_loss = []
        rewards1, rewards2 = [], []
        for batch in tqdm(loader, desc=f"Stage 2 epoch {epoch + 1}/{train_cfg.stage2_epochs}"):
            roll = two_attempt_rollout(model, tokenizer, batch, cfg)
            rewards1.append(np.mean(roll.reward1))
            rewards2.append(np.mean(roll.reward2))

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action1 = roll.action1_tokens.clone()
                action2 = roll.action2_tokens.clone()

                # Attempt 1: log probs + KL to base.
                with torch.no_grad(), model.disable_adapter():
                    _, base_full_logp_1 = get_log_probs(
                        model, action1, roll.x1_len, return_probs=True
                    )
                logp_a1, full_logp_a1 = get_log_probs(
                    model, action1, roll.x1_len, return_probs=True
                )
                kl_1 = get_kl_div(base_full_logp_1, full_logp_a1.detach(), roll.attempt1_answer_mask)
                reward1_t = torch.tensor(roll.reward1, device=model.device, dtype=logp_a1.dtype)
                pg_attempt1 = _masked_mean_logp(logp_a1, roll.attempt1_answer_mask) * reward1_t

                # Attempt 2: log probs + KL + shaped reward.
                with torch.no_grad(), model.disable_adapter():
                    _, base_full_logp_2 = get_log_probs(
                        model, action2, roll.x2_len, return_probs=True
                    )
                logp_a2, full_logp_a2 = get_log_probs(
                    model, action2, roll.x2_len, return_probs=True
                )
                kl_2 = get_kl_div(base_full_logp_2, full_logp_a2.detach(), roll.attempt2_answer_mask)

                reward2_t = torch.tensor(roll.reward2, device=model.device, dtype=logp_a2.dtype)
                bonus = train_cfg.alpha * (reward2_t - reward1_t)
                shaped_reward2 = reward2_t + bonus
                pg_attempt2 = _masked_mean_logp(logp_a2, roll.attempt2_answer_mask) * shaped_reward2

                loss = (
                    -pg_attempt1 - pg_attempt2
                    + train_cfg.beta1 * kl_1
                    + train_cfg.beta1 * kl_2
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            wandb.log({
                "stage2/mean_reward_attempt1": float(np.mean(rewards1)),
                "stage2/mean_reward_attempt2": float(np.mean(rewards2)),
                "stage2/diff_attempt2_minus_attempt1": float(np.mean(rewards2) - np.mean(rewards1)),
                "stage2/loss": float(np.mean(epoch_loss)),
                "stage2/kl_div_attempt1": float(kl_1.mean().item()),
                "stage2/kl_div_attempt2": float(kl_2.mean().item()),
            })

    return model


# --- Eval and save -----------------------------------------------------------


def evaluate(model, tokenizer, test_dataset, cfg: SCoReConfig) -> dict[str, float]:
    """Greedy two-attempt rollout over the test set; returns mean rewards."""
    rewards1, rewards2 = [], []
    loader = test_dataset.iter(batch_size=cfg.eval.batch_size)
    seen = 0
    for batch in tqdm(loader, desc="Eval"):
        if seen >= cfg.eval.max_examples:
            break
        roll = two_attempt_rollout(
            model, tokenizer, batch, cfg, temperature=cfg.eval.generation_temperature
        )
        rewards1.extend(roll.reward1)
        rewards2.extend(roll.reward2)
        seen += len(roll.reward1)

    metrics = {
        "acc_attempt1": float(np.mean(rewards1)) if rewards1 else 0.0,
        "acc_attempt2": float(np.mean(rewards2)) if rewards2 else 0.0,
    }
    metrics["delta"] = metrics["acc_attempt2"] - metrics["acc_attempt1"]
    return metrics


def save_adapter(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))


# --- Main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=_flatten(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    train_dataset, test_dataset = load_and_prepare_data(cfg.dataset, cfg.prompts, tokenizer)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    out_root = Path(cfg.output.dir) / cfg.run_name

    if cfg.eval.enabled:
        wandb.log({f"eval_pretrain/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})

    model = train_stage_1(model, tokenizer, train_dataset, cfg, optimizer)
    if cfg.eval.enabled:
        wandb.log({f"eval_after_stage1/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})
    if cfg.output.save_after_stage1:
        save_adapter(model, tokenizer, out_root / "stage1")

    model = train_stage_2(model, tokenizer, train_dataset, cfg, optimizer)
    if cfg.eval.enabled:
        wandb.log({f"eval_after_stage2/{k}": v for k, v in evaluate(model, tokenizer, test_dataset, cfg).items()})
    if cfg.output.save_after_stage2:
        save_adapter(model, tokenizer, out_root / "stage2")


def _flatten(cfg: SCoReConfig) -> dict[str, Any]:
    """Flatten nested dataclasses into a single dict for W&B config logging."""
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
