import re

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Prompts
prompt = """You are provided with a sequence of numbers. Your task is to identify the logical pattern within the sequence and provide a logical conclusion of what the next number in the sequence should be.
You can think and reason about what the answer will be but provide as a final answer only the numerical value and in the following way:

'Final Answer: The final answer is $answer$.'"""

self_correction_prompt = """There might be an error in the solution above because of lack of understanding of the question. Please correct
the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you
give your final answer, write it in the form 'Final Answer: The final answer is $answer$.' """


# Hyperparameters
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_NAME = "Sebasdi/score_toy_dataset"
BATCH_SIZE = 5
LEARNING_RATE = 5e-6
NUM_EPOCHS = 10
MAX_LENGTH_BATCH = 150  # Tokenization -> input batch seq length
mnt_attempt1 = 75
mnt_attempt2 = 75
x2_max_batch_len = 250
stage_1_epochs = 30
stage_2_epochs = 100
BETA1 = 0.01
BETA2 = 0.1
ALPHA = 10  # 𝛼 is a positive constant multiplier, ideally larger than 1.0


def extract_final_answers(answers):
    return [extract_final_answer(a) for a in answers]


def extract_final_answer(text):
    """
    Extracts the final answer from a string formatted as:
    'Final Answer: The final answer is $answer$.'

    Parameters:
    text (str): The model output containing the final answer.

    Returns:
    int or None: The extracted final answer as an integer, or None if not found.
    """
    # Regular expression to capture the answer after "The final answer is"
    match = re.search(r"The final answer is (\d+)", text)

    # Return the answer as an integer if found, otherwise None
    return int(match.group(1)) if match else None


def load_model_and_tokenizer(
    model_name: str,
    return_tokenizer: bool = True,
    quantize: bool = True,
    lora: bool = True,
):
    if "Phi" not in model_name:
        print(
            "Warning: 'Phi' not found in model_name! This code is optimized for the Phi models!"
        )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = MAX_LENGTH_BATCH
    tokenizer.pad_token = (
        tokenizer.unk_token
    )  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if quantize else None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        # attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    )
    if lora:
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=8,  # 16
            lora_alpha=32,
            target_modules=(
                "all-linear" if "Phi" in model_name else ["q_proj", "v_proj"]
            ),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
    if return_tokenizer:
        return model, tokenizer
    return model


def load_and_prepare_data(dataset_name: str, tokenizer: AutoTokenizer):
    dataset = load_dataset(dataset_name, split="train")

    sequence_prompt = """\n\nHere is your the sequence you need to find the logical continuation for:\n{sequence}"""

    def apply_chat_template(examples, tokenizer):
        # add the math preprompt to all problems
        messages = [
            [{"role": "user", "content": prompt + sequence_prompt.format(sequence=seq)}]
            for seq in examples["sequence"]
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    # Max length is 113 with padding!
    tokenized_dataset = dataset.map(
        apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, batched=True
    )
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["explanation", "sequence"]
    )  # remove explanation
    train_dataloader = DataLoader(
        dataset=tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    return train_dataloader


def reward_bonus(y2: list, y1: list, solutions: list) -> float:
    """
    Compute the reward bonus for the second attempt.
    Based on: reward_bonus = (r(y2, y_star) - r(y1, y_star))
    """
    # Compute the similarity between the first attempt and the correct answer
    similarity = estimate_reward(y1, solutions)

    # Compute the similarity between the second attempt and the correct answer
    similarity2 = estimate_reward(y2, solutions)

    # Compute the reward bonus
    reward_bonus = similarity2 - similarity

    return reward_bonus


def estimate_reward(y2: list, solutions: list) -> float:
    answers = extract_final_answers(y2)
    rewards = [1 if a == s else 0 for (a, s) in zip(answers, solutions)]
    return np.mean(rewards).item()


def train_stage_1(
    model,
    base_model,
    tokenizer,
    dataloader,
    num_epochs,
    optimizer,
    beta2,
):
    total_loss = 0
    total_samples = 0

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
            desc=f"Stage 1 Training {i+1}/{total_batches}",
        ):
            x1 = batch["input_ids"]
            # reshape (seq_len, batch)-> (batch, seq_len)
            x1 = torch.stack(x1).transpose(0, 1).to(model.device)
            attention_mask = batch["attention_mask"]
            # reshape (seq_len, batch)-> (batch, seq_len)
            attention_mask = (
                torch.stack(attention_mask).transpose(0, 1).to(model.device)
            )

            solutions = batch["correct_answer"]

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

            # Compute the loss
            with torch.no_grad():
                base_logits = base_model(x1, attention_mask=attention_mask).logits
                probs_base_1 = torch.softmax(base_logits, dim=-1)

            # Compute logprobs of the first attempt of the trained model
            logits_1 = model(x1, attention_mask=attention_mask).logits
            probs_1 = torch.softmax(logits_1, dim=-1)

            # Compute KL divergence between the first attempt of the base model and the trained model
            kl_div = F.kl_div(probs_1, probs_base_1, reduction="mean")

            logits_2 = model(attempt2_answer_tokens).logits

            # Compute policy gradient loss using reward for y2_logits
            log_probs = torch.log_softmax(logits_2, dim=-1)
            action_log_probs = (
                torch.gather(log_probs, -1, attempt2_answer_tokens.unsqueeze(-1))
                .sum(1)
                .mean()
            )

            # Compute the loss
            loss = -action_log_probs * (reward2 + beta2 * torch.mean(kl_div).to("cpu"))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_loss += loss.item()

            total_samples += len(solutions)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
        )
        mean_reward_attempt_1 = np.mean(mean_reward_a1)
        mean_reward_attempt_2 = np.mean(mean_reward_a2)

        diff_at1_at2 = mean_reward_attempt_2 - mean_reward_attempt_1
        wandb.log(
            {
                "stage1/mr_attempt1": mean_reward_attempt_1,
                "stage1/mr_attempt2": mean_reward_attempt_2,
                "stage1/difference_at1_at2": diff_at1_at2,
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
    total_samples = 0
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

            solutions = batch["correct_answer"]

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

            total_samples += len(solutions)

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


def main():
    wandb.init(project="SCoRe-toy")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, quantize=True, lora=True)
    dataloader = load_and_prepare_data(DATASET_NAME, tokenizer)

    # Load the base model for comparison
    base_model = load_model_and_tokenizer(
        MODEL_NAME, return_tokenizer=False, quantize=False, lora=False
    ).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Freeze the base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameter: ", total_params)

    model = train_stage_1(
        model=model,
        base_model=base_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        num_epochs=stage_1_epochs,
        optimizer=optimizer,
        beta2=BETA2,
    )

    model = train_stage_2(
        model=model,
        base_model=base_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        num_epochs=stage_2_epochs,
        optimizer=optimizer,
        beta1=BETA1,
        alpha=ALPHA,
    )


if __name__ == "__main__":
    main()
