"""Build a small "hard arithmetic" dataset — 5-digit subtraction Qwen3-0.6B can't solve.

Greedy-decodes a pool of 5-digit subtraction problems (the hardest
EleutherAI/arithmetic subset for this model, ~52% greedy-fail), keeps the
ones the model gets wrong, cleans the question text, and saves a HF
DatasetDict with train/test splits via save_to_disk.

Run inside the unsloth-dgx-spark container on Spark (GPU). Push to the Hub
happens separately from a machine with an HF token.

Usage:
    python build_hard_arithmetic.py
"""
import re
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from unsloth import FastLanguageModel
import reward_function as rf

POOL_OFFSET = 18000            # rows 18000-19999 of the parquet export = arithmetic_5ds
POOL_SIZE = 700                # ~52% greedy-fail -> ~360 hard, plenty for 250
GEN_BATCH = 128                # inference-only, greedy
MAX_NEW = 768
N_TRAIN, N_EVAL = 200, 50
OUT = "data/arithmetic_hard"
MODEL = "Qwen/Qwen3-0.6B"
SYSTEM = (
    "You are an arithmetic solver. Output EXACTLY this format and nothing else:\n"
    "<think>\na brief calculation — just the arithmetic steps, no commentary\n</think>\n"
    "<answer>the final number</answer>\n"
    "Keep the reasoning short. Put only the bare number between the answer tags."
)

_QPREFIX = re.compile(r"^\s*Question:\s*", re.IGNORECASE)
_ASUFFIX = re.compile(r"\s*Answer:\s*$", re.IGNORECASE)


def clean_question(ctx: str) -> str:
    """EleutherAI ships 'Question: What is X?\\nAnswer:' — strip the framing so
    build() in train.py doesn't double the 'Question:' prefix."""
    return _ASUFFIX.sub("", _QPREFIX.sub("", ctx)).strip()


def main() -> None:
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=2048, dtype=None, load_in_4bit=False)
    FastLanguageModel.for_inference(model)
    gsm = rf.ANSWER_EXTRACTORS["gsm8k_hash"]

    ds = load_dataset("EleutherAI/arithmetic", "default", revision="refs/convert/parquet",
                      split=f"validation[{POOL_OFFSET}:{POOL_OFFSET + POOL_SIZE}]")
    rows = [{"question": clean_question(ex["context"]), "answer": ex["completion"].strip()}
            for ex in ds]
    print(f"pooled {len(rows)} candidate 5-digit-subtraction questions")

    # Greedy-decode in big batches; "hard" = greedy gets it wrong (or emits no number)
    hard = []
    for i in range(0, len(rows), GEN_BATCH):
        batch = rows[i:i + GEN_BATCH]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Question:\n" + r["question"]}],
                tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        x = tok(prompts, padding="longest", padding_side="left", return_tensors="pt").to(model.device)
        out = model.generate(x["input_ids"], attention_mask=x["attention_mask"],
                             max_new_tokens=MAX_NEW, do_sample=False)
        gen = tok.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(batch, gen):
            if gsm(g) == "" or gsm(g) != gsm(r["answer"]):
                hard.append(r)
        print(f"  scored {min(i + GEN_BATCH, len(rows))}/{len(rows)}  hard so far: {len(hard)}")

    fail = 100 * len(hard) / len(rows)
    print(f"\n{len(hard)}/{len(rows)} greedy-wrong = {fail:.0f}% fail rate")
    if len(hard) < N_TRAIN + N_EVAL:
        raise SystemExit(f"only {len(hard)} hard questions, need {N_TRAIN + N_EVAL}")

    rng = np.random.default_rng(3407)
    rng.shuffle(hard)
    sel = hard[: N_TRAIN + N_EVAL]
    dd = DatasetDict({
        "train": Dataset.from_list(sel[:N_TRAIN]),
        "test": Dataset.from_list(sel[N_TRAIN:]),
    })
    print(f"\nfinal: train={len(dd['train'])}  test={len(dd['test'])}")
    print(f"  sample: {dd['train'][0]}")
    dd.save_to_disk(OUT)
    print(f"saved to {OUT}/  — scp to a machine with an HF token, then push_to_hub")


if __name__ == "__main__":
    main()
