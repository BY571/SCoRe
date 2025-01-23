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
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
MAX_LENGTH_BATCH = 1000  # Tokenization -> input batch seq length
mnt_attempt1 = 1000
mnt_attempt2 = 1000
x2_max_batch_len = 1000
stage_1_epochs = 4  
stage_2_epochs = 1
BETA1 = 0.01
BETA2 = 0.1
ALPHA = 10  # 𝛼 is a positive constant multiplier, ideally larger than 1.0
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
    for a, s in zip(answers, solutions):
        reward1, correct1 = estimate_reward(
            extract_final_answer(a),
            extract_final_answer(s),
        )
        rewards.append(reward1)
        corrects.append(correct1)
    return rewards, corrects


def load_model_and_tokenizer():
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # for parameter https://www.reddit.com/r/LocalLLaMA/comments/1gpwrq1/how_to_use_qwen25coderinstruct_without/
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
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

    return tokenized_dataset, tokenized_test_dataset


def reward_bonus(y2: list, y1: list, y_star: list) -> float:
    """
    Compute the reward bonus for the second attempt.
    Based on: reward_bonus = (r(y2, y_star) - r(y1, y_star))
    """
    # Compute the similarity between the first attempt and the correct answer
    similarity = estimate_reward(y1, y_star)

    # Compute the similarity between the second attempt and the correct answer
    similarity2 = estimate_reward(y2, y_star)

    # Compute the reward bonus
    reward_bonus = similarity2 - similarity

    return reward_bonus


def estimate_reward(y2, y_star) -> torch.Tensor:
    return comparator.compare(y2, y_star, method="bert")


def train_stage_1(
    model,
    tokenizer,
    dataloader,
    num_epochs,
    optimizer,
    beta2,
):  
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        mean_reward_a1 = []
        mean_reward_a2 = []
        correct_solution1 = []
        correct_solution2 = []
        i = 0
        for i, batch in tqdm(
            enumerate(dataloader),
            desc=f"Stage 1 Training...",
        ):
            x1 = tokenizer.batch_encode_plus(
                batch["text"],
                padding="max_length",
                max_length=MAX_LENGTH_BATCH,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)

            solutions = batch["solution"]
            FastLanguageModel.for_inference(model)
            # First attempt (trained model) get on-policy action of the train model
            action1_token = model.generate(
                x1["input_ids"],
                attention_mask=x1["attention_mask"],
                max_new_tokens=1000,
                use_cache=True,
                temperature=0.5,
            )
            x1_len = x1["input_ids"].shape[1]
            only_answer1 = tokenizer.batch_decode(
                action1_token[:, x1_len:], skip_special_tokens=True
            )
            reward1, correct1 = eval_answers(only_answer1, solutions)
            mean_reward_a1.append(np.mean(reward1))

            # add self_correction_prompt to the first attempt
            messages = batch["messages"]
            for m, a in zip(messages, only_answer1):
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
                max_length=x2_max_batch_len,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)
            x2_len = x2["input_ids"].shape[1] # needed for later
            action2_token = model.generate(
                x2["input_ids"],
                attention_mask=x2["attention_mask"],
                max_new_tokens=mnt_attempt2,
                use_cache=True,
                temperature=0.5,
            )

            input_length = x2["input_ids"].shape[1]
            # Extract only the newly generated tokens
            attempt2_answer_tokens = action2_token[:, input_length:]
            attempt2_answer = tokenizer.batch_decode(
                attempt2_answer_tokens, skip_special_tokens=True
            )
            reward2, correct2 = eval_answers(attempt2_answer, solutions)
            mean_reward_a2.append(np.mean(reward2))

            FastLanguageModel.for_training(model)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action1_token = action1_token.clone()
                # Compute the loss
                with torch.no_grad() and model.disable_adapter():
                    base_logits = model(action1_token).logits
                    probs_base_1 = torch.softmax(base_logits[:, x1_len:], dim=-1) # only answer prediction

                    # Compute probs of the first attempt
                    logits_1 = model(action1_token).logits
                    probs_1 = torch.softmax(logits_1[:, x1_len:], dim=-1) # only answer prediction

                    # Compute KL divergence between the first attempt of the base model and the trained model
                    kl_div = F.kl_div(probs_1, probs_base_1, reduction="mean")

                action2_token = action2_token.clone()
                logits_2 = model(action2_token).logits
                logits_2 = logits_2[:, :-1, :]  # shift the logits to the right by one token
                input_ids = action2_token[:, 1:]  # remove the first token
                
                # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
                per_token_logps = []
                for logits_row, input_ids_row in zip(logits_2, input_ids):
                    log_probs = logits_row.log_softmax(dim=-1)
                    token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                    per_token_logps.append(token_log_prob)
                log_probs = torch.stack(per_token_logps)
                # take off prompt token, -1 for shifting logits
                log_probs = log_probs[:, x2_len - 1 :]

                # Compute the loss
                loss = -(log_probs *torch.tensor(reward2).unsqueeze(-1).to(model.device)).mean() + beta2 * torch.mean(kl_div)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                correct_solution1.append(correct1)
                correct_solution2.append(correct2)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(epoch_loss):.4f}"
        )
        mean_reward_attempt_1 = np.mean(mean_reward_a1)
        mean_reward_attempt_2 = np.mean(mean_reward_a2)

        diff_at1_at2 = mean_reward_attempt_2 - mean_reward_attempt_1
        total_correct_solution1 = sum(sum(inner_list) for inner_list in correct_solution1)
        total_correct_solution2 = sum(sum(inner_list) for inner_list in correct_solution2)
        wandb.log(
            {
                "mr_attempt1": mean_reward_attempt_1,
                "mr_attempt2": mean_reward_attempt_2,
                "difference_at1_at2": diff_at1_at2,
                "correct_solution1": total_correct_solution1,
                "correct_solution2": total_correct_solution2,
                "loss": np.mean(epoch_loss),
            }
        )

    return model


def train_stage_2(
    model,
    base_model,
    tokenizer,
    dataloader,
    num_epochs,
    optimizer,
    beta1,
    alpha,
):

    total_loss = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total_batches = len(dataloader)
        mean_reward_a1 = []
        mean_reward_a2 = []
        i = 0
        for i, batch in tqdm(
            enumerate(dataloader),
            total=total_batches,
            desc=f"Stage 2 Training {i+1}/{total_batches}",
        ):
            x1 = batch["input_ids"]
            # reshape (seq_len, batch)-> (batch, seq_len)
            x1 = torch.stack(x1).transpose(0, 1).to(model.device)
            attention_mask = batch["attention_mask"]
            # reshape (seq_len, batch)-> (batch, seq_len)
            attention_mask = (
                torch.stack(attention_mask).transpose(0, 1).to(model.device)
            )

            solutions = batch["solution"]

            # First attempt (trained model) get on-policy action of the train model
            with torch.no_grad():
                action1_token = model.generate(
                    x1,
                    attention_mask=attention_mask,
                    max_new_tokens=mnt_attempt1,
                    use_cache=True,
                    do_sample=True,
                    temperature=1.0,
                )
            action1 = tokenizer.batch_decode(action1_token, skip_special_tokens=True)
            reward1 = estimate_reward(action1, solutions)
            mean_reward_a1.append(reward1)

            x2 = [a1 + self_correction_prompt for a1 in action1]
            x2_encoded = tokenizer(
                x2,
                padding="max_length",
                truncation=True,
                max_length=x2_max_batch_len,
                return_tensors="pt",
            )
            input_ids_2 = x2_encoded.input_ids.to(model.device)
            attention_mask_2 = x2_encoded.attention_mask.to(model.device)
            action2_token = model.generate(
                input_ids_2,
                attention_mask=attention_mask_2,
                max_new_tokens=mnt_attempt2,
                use_cache=True,
                do_sample=True,
                temperature=1.0,
            )

            input_length = input_ids_2.shape[1]
            # Extract only the newly generated tokens
            attempt2_answer_tokens = action2_token[:, input_length:]
            attempt2_answer = tokenizer.batch_decode(
                attempt2_answer_tokens, skip_special_tokens=True
            )
            reward2 = estimate_reward(attempt2_answer, solutions)
            mean_reward_a2.append(reward2)

            # Compute kl + logprobs of the first attempt
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    base_logits = base_model(x1, attention_mask=attention_mask).logits
                    probs_base_1 = torch.softmax(base_logits, dim=-1)

                logits_1 = model(x1, attention_mask=attention_mask).logits
                probs_1 = torch.softmax(logits_1, dim=-1)
                kl_div_1 = F.kl_div(probs_1, probs_base_1, reduction="mean")

                log_probs_1 = torch.log_softmax(logits_1, dim=-1)
                action_log_probs_1 = (
                    torch.gather(log_probs_1, -1, x1.unsqueeze(-1)).sum(1).mean()
                )

                # Compute kl + logprobs + reward bonus of the second attempt
                with torch.no_grad():
                    base_logits = base_model(attempt2_answer_tokens).logits
                    probs_base_2 = torch.softmax(base_logits, dim=-1)

                logits_2 = model(attempt2_answer_tokens).logits
                probs_2 = torch.softmax(logits_2, dim=-1)
                kl_div_2 = F.kl_div(probs_2, probs_base_2, reduction="mean")

                # Compute reward 2
                reward_boni = reward_bonus(attempt2_answer, action1, solutions)
                reward2 = reward2 + alpha * reward_boni

                # Compute policy gradient loss using reward for y2_logits
                log_probs_2 = torch.log_softmax(logits_2, dim=-1)
                action_log_probs_2 = (
                    torch.gather(log_probs_2, -1, attempt2_answer_tokens.unsqueeze(-1))
                    .sum(1)
                    .mean()
                )

                episode_action_log_probs = action_log_probs_1 + action_log_probs_2

                # Compute loss
                loss = -episode_action_log_probs * (
                    (reward2 + beta1 * kl_div_2) + (reward1 + beta1 * kl_div_1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
        )
        mean_reward_attempt_1 = np.mean(mean_reward_a1)
        mean_reward_attempt_2 = np.mean(mean_reward_a2)

        diff_at1_at2 = mean_reward_attempt_2 - mean_reward_attempt_1
        wandb.log(
            {
                "stage2/mr_attempt1": mean_reward_attempt_1,
                "stage2/mr_attempt2": mean_reward_attempt_2,
                "stage2/difference_at1_at2": diff_at1_at2,
            }
        )


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
    train_dataloader = train_dataset.iter(batch_size=2)
    test_dataloader = test_dataloader.iter(batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameter: ", total_params)

    # total_reward, accuracy = evaluate_model(model, tokenizer, test_dataloader, device=device)

    model = train_stage_1(
        model=model,
        tokenizer=tokenizer,
        dataloader=train_dataloader,
        num_epochs=stage_1_epochs,
        optimizer=optimizer,
        beta2=BETA2,
    )

    # model = train_stage_2(
    #     model,
    #     base_model,
    #     tokenizer,
    #     dataloader,
    #     num_epochs=stage_2_epochs,
    #     optimizer=optimizer,
    #     beta1=BETA1,
    #     alpha=ALPHA,
    # )

    # Save the trained model
    # trained_model.save_pretrained("score_stage_i_model")
    # tokenizer.save_pretrained("score_stage_i_model")
    # print("Training completed. Model saved as 'score_stage_i_model'.")


if __name__ == "__main__":
    main()
