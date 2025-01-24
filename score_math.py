import re

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import concatenate_datasets, load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from string_matcher import LLMAnswerComparator
from torch.utils.data import DataLoader
from tqdm import tqdm
from unsloth import FastLanguageModel


# Prompts
math_prompt = """You are a math expert. Solve problems step by step, showing all work. 

Final Answer Format:
- For numbers: "The final answer is: 42"
- For equations: "The final answer is: $y = 2x - 13$"

Always clearly show your reasoning and conclusion."""
# 76 token
self_correction_prompt = """There might be an error in the solution above because of lack of understanding of the question. Please correct
the error, if any, and rewrite the solution. At the end of the Solution, when you give your final answer indicate your final answer by writing the 'final answer is: <your answer>'.
"""
# 62 token

# max problem length is 744 token + math_prompt 76 token + self_correction_prompt 62 token = 882 token
# max length can be savely set to 900 token.
# Hyperparameters
DATASET_NAME = "Sebasdi/math_final_answer"
BATCH_SIZE = 2
LEARNING_RATE = 5e-6
MAX_PROMPT_LEN1 = 1000
mnt_attempt1 = 1000
mnt_attempt2 = 1000
MAX_PROMPT_LEN2 = 2000
stage_1_epochs = 10
stage_2_epochs = 10
BETA1 = 0.01
BETA2 = 0.1
ALPHA = 10  # 𝛼 is a positive constant multiplier, ideally larger than 1.0
TEMP = 1.0
comparator = LLMAnswerComparator(threshold=0.9)


def extract_final_answer(text):
    # Try multiple patterns in order
    patterns = [
        r"The final answer is:?\s*\$([^$]+)\$",  # Matches $equation$ after "The final answer is:"
        r"The final answer is:?\s*(?:\$\$)?\\boxed{([^}]+)}",  # Boxed pattern
        r"The final answer is:?\s*(\d+)",  # Numeric pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback to \boxed{} pattern anywhere in text
    boxed_match = re.search(r"\\boxed{([^}]+)}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    return ""


def eval_answers(answers, solutions):
    rewards = []
    corrects = []
    try:
        for a, s in zip(answers, solutions):
            reward1, correct1 = estimate_reward(
                extract_final_answer(a),
                extract_final_answer(s),
            )
            rewards.append(reward1)
            corrects.append(correct1)
        return rewards, corrects
    except Exception as e:
        print(e)
        return [0.0 for _ in range(answers)], [False for _ in range(len(solutions))]


def load_model_and_tokenizer():
    max_seq_length = 3000  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # for parameter https://www.reddit.com/r/LocalLLaMA/comments/1gpwrq1/how_to_use_qwen25coderinstruct_without/
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer


def load_and_prepare_data(dataset_name: str, tokenizer):

    dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")
    test_size = 200

    # update sizes
    # Take 500 examples from the test set for our new test set
    new_test_dataset = test_dataset.select(range(test_size))

    # Use the remaining 4500 examples from the test set
    remaining_test = test_dataset.select(range(test_size, len(test_dataset)))

    # Combine the original train set with the remaining 4500 from the test set
    dataset = concatenate_datasets([dataset, remaining_test])
    test_dataset = new_test_dataset

    def tokenize_function(examples):

        messages = [
            [
                {"role": "system", "content": math_prompt},
                {"role": "user", "content": "Question:\n" + problem},
            ]
            for problem in examples["problem"]
        ]
        solutions = [solution for solution in examples["final_answer_solution"]]
        text = [
            tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            for message in messages
        ]

        return {"text": text, "solution": solutions, "messages": messages}

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_dataset.select(range(8)), tokenized_test_dataset


def reward_bonus(y2s: list, y1s: list, y_stars: list) -> float:
    """
    Compute the reward bonus for the second attempt.
    Based on: reward_bonus = (r(y2, y_star) - r(y1, y_star))
    """
    similarities1 = []
    similarities2 = []
    for y2, y1, y_star in zip(y2s, y1s, y_stars):
        # Compute the similarity between the first attempt and the correct answer
        similarity, _ = estimate_reward(y1, y_star)

        # Compute the similarity between the second attempt and the correct answer
        similarity2, _ = estimate_reward(y2, y_star)
        similarities1.append(similarity)
        similarities2.append(similarity2)

    # Compute the reward bonus
    return torch.tensor(similarities2) - torch.tensor(similarities1)


def estimate_reward(y2, y_star) -> torch.Tensor:
    return comparator.compare(y2, y_star, method="bert")


def get_log_probs(model, input_ids, prompt_len, return_probs=False):
    logits = model(input_ids).logits
    logits = logits[:, :-1, :]  # shift the logits to the right by one token
    input_ids = input_ids[:, 1:]  # remove the first token
    # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
    per_token_logps = []
    if return_probs:
        log_probs_ = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        if return_probs:
            log_probs_.append(log_probs)
        token_log_prob = torch.gather(
            log_probs, dim=1, index=input_ids_row.unsqueeze(1)
        ).squeeze(1)
        per_token_logps.append(token_log_prob)
    # stack and take off prompt token, -1 for remove by shifting logits
    if not return_probs:
        return torch.stack(per_token_logps)[:, prompt_len - 1 :]
    else:
        return (
            torch.stack(per_token_logps)[:, prompt_len - 1 :],
            torch.stack(log_probs_)[:, prompt_len - 1 :],
        )


def get_eos_mask(answer_ids, tokenizer):
    # gets where the eos token is in the answer_ids
    is_eos = answer_ids == tokenizer.eos_token_id
    # Set EOS index to first EOS position if found, otherwise use max sequence length
    # eos_idx = is_eos.int().argmax(dim=1) if is_eos.any() else torch.full((is_eos.size(0),), is_eos.size(1), device=is_eos.device)
    # Create mask using cumulative sum: 1s before first EOS, 0s after
    mask = (torch.cumsum(is_eos, dim=1) <= 1).int()
    return mask


def get_kl_div(base_logprobs, logprobs, mask):
    kl_div = F.kl_div(logprobs, base_logprobs, reduction="none", log_target=True)
    return (kl_div.mean(-1) * mask).sum(-1) / mask.sum(-1)


def train_stage_1(
    model,
    tokenizer,
    train_dataset,
    num_epochs,
    optimizer,
    beta2,
):
    for epoch in range(num_epochs):
        dataloader = train_dataset.iter(batch_size=BATCH_SIZE)
        epoch_loss = []
        mean_reward_a1 = []
        mean_reward_a2 = []
        correct_solution1 = []
        correct_solution2 = []
        i = 0
        for i, batch in tqdm(
            enumerate(dataloader),
            desc=f"Stage 1 Training {epoch+1}/{num_epochs}",
        ):
            x1 = tokenizer.batch_encode_plus(
                batch["text"],
                padding="max_length",
                max_length=MAX_PROMPT_LEN1,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)

            solutions = batch["solution"]
            FastLanguageModel.for_inference(model)
            # First attempt (trained model) get on-policy action of the train model
            action1_token = model.generate(
                x1["input_ids"],
                attention_mask=x1["attention_mask"],
                max_new_tokens=mnt_attempt1,
                temperature=TEMP,
            )
            x1_len = x1["input_ids"].shape[1]
            # Extract only the newly generated tokens to evaluate the second attempt
            attempt1_answer_tokens = action1_token[:, x1_len:]
            attempt1_answer_mask = get_eos_mask(
                attempt1_answer_tokens, tokenizer
            )  # for later use
            answer1 = tokenizer.batch_decode(
                attempt1_answer_tokens, skip_special_tokens=True
            )
            reward1, correct1 = eval_answers(answer1, solutions)
            mean_reward_a1.append(np.mean(reward1))

            # add self_correction_prompt to the first attempt
            messages = batch["messages"]
            for m, a in zip(messages, answer1):
                m.append({"role": "assistant", "content": a})
                m.append({"role": "user", "content": self_correction_prompt})
            init_x2 = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            x2 = tokenizer.batch_encode_plus(
                init_x2,
                padding="max_length",
                truncation=True,
                max_length=MAX_PROMPT_LEN2,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)
            x2_len = x2["input_ids"].shape[1]
            action2_token = model.generate(
                x2["input_ids"],
                attention_mask=x2["attention_mask"],
                max_new_tokens=mnt_attempt2,
                temperature=TEMP,
            )

            # Extract only the newly generated tokens to evaluate the second attempt
            attempt2_answer_tokens = action2_token[:, x2_len:]
            attempt2_answer_mask = get_eos_mask(
                attempt2_answer_tokens, tokenizer
            )  # for later use
            attempt2_answer = tokenizer.batch_decode(
                attempt2_answer_tokens, skip_special_tokens=True
            )
            reward2, correct2 = eval_answers(attempt2_answer, solutions)
            mean_reward_a2.append(np.mean(reward2))

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action1_token = action1_token.clone()
                # Compute the loss
                # Compute kl term
                with torch.no_grad():
                    with model.disable_adapter():
                        # compute base model log probs
                        _, base_probs = get_log_probs(
                            model, action1_token, x1_len, return_probs=True
                        )

                # Compute probs of the first attempt log probs
                att1_log_probs, att1_probs = get_log_probs(
                    model, action1_token, x1_len, return_probs=True
                )

                # Compute kl divergence
                kl_div = get_kl_div(
                    base_probs, att1_probs.detach(), attempt1_answer_mask
                )

                # Compute log probs of the second attempt
                action2_token = action2_token.clone()
                att2_log_probs = get_log_probs(model, action2_token, x2_len)

                # Compute the loss
                loss = -(
                    (
                        (
                            (att1_log_probs * attempt1_answer_mask).sum(-1)
                            / attempt1_answer_mask.sum(-1)
                            + (att2_log_probs * attempt2_answer_mask).sum(-1)
                            / attempt2_answer_mask.sum(-1)
                        )
                        * torch.tensor(reward2).to(model.device)
                    )
                    - beta2 * kl_div
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                correct_solution1.append(correct1)
                correct_solution2.append(correct2)

            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(epoch_loss):.4f}")
            mean_reward_attempt_1 = np.mean(mean_reward_a1)
            mean_reward_attempt_2 = np.mean(mean_reward_a2)

            diff_at1_at2 = mean_reward_attempt_2 - mean_reward_attempt_1
            total_correct_solution1 = sum(
                sum(inner_list) for inner_list in correct_solution1
            )
            total_correct_solution2 = sum(
                sum(inner_list) for inner_list in correct_solution2
            )
            wandb.log(
                {
                    "mean_reward_attmpt1": mean_reward_attempt_1,
                    "mean_reward_attmpt2": mean_reward_attempt_2,
                    "difference_at1_at2": diff_at1_at2,
                    "correct_solution1": total_correct_solution1,
                    "correct_solution2": total_correct_solution2,
                    "loss": np.mean(epoch_loss),
                    "kl_div": kl_div.mean().item(),
                }
            )

    return model


def train_stage_2(
    model,
    tokenizer,
    train_dataset,
    num_epochs,
    optimizer,
    beta1,
    alpha,
):

    for epoch in range(num_epochs):
        dataloader = train_dataset.iter(batch_size=2)
        epoch_loss = []
        mean_reward_a1 = []
        mean_reward_a2 = []
        correct_solution1 = []
        correct_solution2 = []
        i = 0
        for i, batch in tqdm(
            enumerate(dataloader),
            desc=f"Stage 2 Training {epoch+1}/{num_epochs}",
        ):
            x1 = tokenizer.batch_encode_plus(
                batch["text"],
                padding="max_length",
                max_length=MAX_PROMPT_LEN1,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)

            solutions = batch["solution"]
            FastLanguageModel.for_inference(model)
            # First attempt (trained model) get on-policy action of the train model
            action1_token = model.generate(
                x1["input_ids"],
                attention_mask=x1["attention_mask"],
                max_new_tokens=mnt_attempt1,
                temperature=TEMP,
            )
            x1_len = x1["input_ids"].shape[1]
            # Extract only the newly generated tokens to evaluate the second attempt
            attempt1_answer_tokens = action1_token[:, x1_len:]
            attempt1_answer_mask = get_eos_mask(
                attempt1_answer_tokens, tokenizer
            )  # for later use
            attempt_answer1 = tokenizer.batch_decode(
                attempt1_answer_tokens, skip_special_tokens=True
            )
            reward1, correct1 = eval_answers(attempt_answer1, solutions)
            mean_reward_a1.append(np.mean(reward1))

            # add self_correction_prompt to the first attempt
            messages = batch["messages"]
            for m, a in zip(messages, attempt_answer1):
                m.append({"role": "assistant", "content": a})
                m.append({"role": "user", "content": self_correction_prompt})
            init_x2 = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            x2 = tokenizer.batch_encode_plus(
                init_x2,
                padding="max_length",
                truncation=True,
                max_length=MAX_PROMPT_LEN2,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)
            x2_len = x2["input_ids"].shape[1]
            action2_token = model.generate(
                x2["input_ids"],
                attention_mask=x2["attention_mask"],
                max_new_tokens=mnt_attempt2,
                temperature=TEMP,
            )

            # Extract only the newly generated tokens to evaluate the second attempt
            attempt2_answer_tokens = action2_token[:, x2_len:]
            attempt2_answer_mask = get_eos_mask(
                attempt2_answer_tokens, tokenizer
            )  # for later use
            attempt2_answer = tokenizer.batch_decode(
                attempt2_answer_tokens, skip_special_tokens=True
            )
            reward2, correct2 = eval_answers(attempt2_answer, solutions)
            mean_reward_a2.append(np.mean(reward2))

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action1_token = action1_token.clone()
                # Compute the loss attempt 1 + kl term
                with torch.no_grad():
                    with model.disable_adapter():
                        # compute base model log probs
                        _, base_probs = get_log_probs(
                            model, action1_token, x1_len, return_probs=True
                        )

                # Compute probs of the first attempt log probs
                att1_log_probs, probs = get_log_probs(
                    model, action1_token, x1_len, return_probs=True
                )

                # Compute kl divergence
                kl_div = get_kl_div(base_probs, probs.detach(), attempt1_answer_mask)

                loss_attmpt1 = -(
                    (
                        (att1_log_probs * attempt1_answer_mask).sum(-1)
                        / attempt1_answer_mask.sum(-1)
                    )
                    * torch.tensor(reward1).to(model.device)
                    - beta1 * kl_div
                )

                # Compute log probs of the second attempt
                action2_token = action2_token.clone()
                with torch.no_grad():
                    with model.disable_adapter():
                        # compute base model log probs
                        _, base_probs = get_log_probs(
                            model, action2_token, x2_len, return_probs=True
                        )
                att2_log_probs, probs = get_log_probs(
                    model, action2_token, x2_len, return_probs=True
                )

                # Compute kl divergence
                kl_div = get_kl_div(base_probs, probs.detach(), attempt2_answer_mask)

                # add reward bonus
                reward_boni = reward_bonus(attempt2_answer, attempt_answer1, solutions)
                reward2 = (torch.tensor(reward2) + alpha * reward_boni).to(model.device)

                # Compute the loss
                loss_attmpt2 = -(
                    (
                        (att2_log_probs * attempt2_answer_mask).sum(-1)
                        / attempt2_answer_mask.sum(-1)
                    )
                    * reward2
                    - beta1 * kl_div
                )

                total_loss = (loss_attmpt1 + loss_attmpt2).mean()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss.append(total_loss.item())
                correct_solution1.append(correct1)
                correct_solution2.append(correct2)

            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(epoch_loss):.4f}")
            mean_reward_attempt_1 = np.mean(mean_reward_a1)
            mean_reward_attempt_2 = np.mean(mean_reward_a2)

            diff_at1_at2 = mean_reward_attempt_2 - mean_reward_attempt_1
            total_correct_solution1 = sum(
                sum(inner_list) for inner_list in correct_solution1
            )
            total_correct_solution2 = sum(
                sum(inner_list) for inner_list in correct_solution2
            )
            wandb.log(
                {
                    "mean_reward_attmpt1": mean_reward_attempt_1,
                    "mean_reward_attmpt2": mean_reward_attempt_2,
                    "difference_at1_at2": diff_at1_at2,
                    "correct_solution1": total_correct_solution1,
                    "correct_solution2": total_correct_solution2,
                    "loss": np.mean(epoch_loss),
                    "kl_div": kl_div.mean().item(),
                }
            )

    return model


def evaluate_model(model, tokenizer, dataloader):
    # TODO
    pass


def main():
    wandb.init(project="SCoRe-Math")
    model, tokenizer = load_model_and_tokenizer()
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    train_dataset, test_dataloader = load_and_prepare_data(DATASET_NAME, tokenizer)
    train_dataloader = train_dataset.iter(batch_size=BATCH_SIZE)
    test_dataloader = test_dataloader.iter(batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # total_reward, accuracy = evaluate_model(model, tokenizer, test_dataloader, device=device)

    model = train_stage_1(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        num_epochs=stage_1_epochs,
        optimizer=optimizer,
        beta2=BETA2,
    )

    model = train_stage_2(
        model,
        tokenizer,
        train_dataset,
        num_epochs=stage_2_epochs,
        optimizer=optimizer,
        beta1=BETA1,
        alpha=ALPHA,
    )

    # Save the trained model
    # trained_model.save_pretrained("score_stage_i_model")
    # tokenizer.save_pretrained("score_stage_i_model")
    # print("Training completed. Model saved as 'score_stage_i_model'.")


if __name__ == "__main__":
    main()
