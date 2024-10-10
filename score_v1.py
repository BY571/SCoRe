import os
import re

import numpy as np
import torch
import wandb
from datasets import concatenate_datasets, load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from string_matcher import LLMAnswerComparator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set CUDA device to the first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

math_prompt = """You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by
step. At the end of the Solution, when you give your final answer, write it in the form 'Final Answer: The final
answer is $answer$. I hope it is correct.'"""
self_correction_prompt = """There might be an error in the solution above because of lack of understanding of the question. Please correct
the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you
give your final answer, write it in the form 'Final Answer: The final answer is $answer$. I hope it is correct.' """

# Hyperparameters
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_NAME = "Sebasdi/math_final_answer"
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512
BETA1 = 0.1
BETA2 = 1.0
ALPHA = 2.0  # 𝛼 is a positive constant multiplier, ideally larger than 1.0
comparator = LLMAnswerComparator(threshold=0.9)


def extract_final_answer(solution):
    pattern = r"Final Answer: The final answer is \$(.*?)\$\ ."  # I hope it is correct\
    out = []
    for sol in solution:
        match = re.search(pattern, sol)
        if match:
            out.append(match.group(1))
        else:
            out.append(sol)
    return out


def load_model_and_tokenizer(model_name, return_tokenizer=True, quantize=True, lora=True):
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
    tokenizer.model_max_length = 1024
    tokenizer.pad_token = (
        tokenizer.unk_token
    )  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"
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


def load_and_prepare_data(dataset_name, tokenizer):
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
        # add the math preprompt to all problems
        examples["problem"] = [math_prompt + problem for problem in examples["problem"]]
        return tokenizer(
            examples["problem"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataloader = DataLoader(
        tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader


def compute_kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q), dim=-1)


def reward_bonus(y2, y1, y_star):
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


def estimate_reward(y2, y_star, return_correctness=False):

    results = comparator.batch_compare(y2, oracle_responses=y_star, method="bert")
    # results is a list of tuples (similarity, correct)
    similarities = [result[0] for result in results]
    correctness = [result[1] for result in results]
    if return_correctness:
        return torch.tensor(similarities), torch.tensor(correctness)
    return torch.tensor(similarities)


def train_stage_1(
    model,
    base_model,
    tokenizer,
    dataloader,
    num_epochs,
    optimizer,
    beta2,
    device="cpu",
):
    total_loss = 0  # Initialize total_loss here
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # Initialize epoch_loss for each epoch

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"]
            input_ids = torch.stack(input_ids)
            # reshape (max token, batch)-> (batch, max token)
            input_ids = input_ids.transpose(0, 1).to(device)
            attention_mask = batch["attention_mask"]
            attention_mask = torch.stack(attention_mask)
            # reshape (max token, batch)-> (batch, max token)
            attention_mask = attention_mask.transpose(0, 1).to(device)

            solutions = batch["solution"]

            # First attempt (trained model) get on-policy action of the train model
            action1_token = model.generate(input_ids, max_new_tokens=256, use_cache=False)
            # <- our final action (includes the prompt!)
            action1 = tokenizer.batch_decode(action1_token, skip_special_tokens=True)

            input_ids_2 = [a1 + self_correction_prompt for a1 in action1]
            input_ids_2 = tokenizer(input_ids_2, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt",).input_ids
            action2_token = model.generate(input_ids_2.to(device), max_new_tokens=256, use_cache=False)
            action2 = tokenizer.batch_decode(action2_token, skip_special_tokens=True)

            # Now we have the trajectory a1, a2 and we need to calculate the Reward and Loss

            # Compute reward:
            reward = estimate_reward(
                extract_final_answer(action2), extract_final_answer(solutions)
            )

            # Compute the loss
            # First attempt (base model) get on-policy action of the base model
            with torch.no_grad():
                out_dict_base = base_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    output_logits=True,
                    return_dict_in_generate=True,
                    use_cache=False,
                )
                logits_base_1 = torch.stack(out_dict_base.logits).transpose(0, 1)
                probs_base_1 = torch.softmax(logits_base_1, dim=-1)

            # Compute logprobs of the first attempt of the trained model
            logits_1 = model(action1_token).logits
            probs_1 = torch.softmax(logits_1, dim=-1)

            # Compute KL divergence between the first attempt of the base model and the trained model

            # Get the minimum length
            min_length = min(probs_base_1.size(1), probs_1.size(1))
            kl_div = compute_kl_divergence(probs_base_1[:, :min_length, :], probs_1[:, :min_length, :])

            # Second attempt (trained model)
            outputs_2 = model(action2_token)
            logits_2 = outputs_2.logits
            # probs_2 = torch.softmax(logits_2, dim=-1)

            # Compute policy gradient loss using reward for y2_logits
            log_probs = torch.log_softmax(logits_2, dim=-1)
            action_log_probs = torch.gather(log_probs, -1, action2_token.unsqueeze(-1))

            # TODO: Test do we need action log probs for the first attempt?

            # Compute the loss
            loss = -torch.sum(action_log_probs) * reward.mean() + beta2 * torch.mean(kl_div).to("cpu")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            epoch_loss += loss.item()
            print("Loss: ", loss.item())

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
        )

    return model, total_loss


def train_stage_2(
    model,
    base_model,
    tokenizer,
    dataloader,
    num_epochs,
    optimizer,
    beta1,
    alpha,
    device="cpu",
):

    total_loss = 0  # Initialize total_loss here
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # Initialize epoch_loss for each epoch

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"]
            input_ids = torch.stack(input_ids)
            # reshape (max token, batch)-> (batch, max token)
            input_ids = input_ids.transpose(0, 1).to(device)
            attention_mask = batch["attention_mask"]
            attention_mask = torch.stack(attention_mask)
            # reshape (max token, batch)-> (batch, max token)
            attention_mask = attention_mask.transpose(0, 1).to(device)

            solutions = batch["solution"]

            # First attempt (base model)
            with torch.no_grad():
                out_dict_base = base_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_LENGTH,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
                logits_base_1 = torch.stack(out_dict_base.logits).transpose(0, 1)
                probs_base_1 = torch.softmax(logits_base_1, dim=-1)

            # First attempt (trained model)
            outputs_1 = model(input_ids, attention_mask=attention_mask)
            logits_1 = outputs_1.logits
            probs_1 = torch.softmax(logits_1, dim=-1)

            # Add check for correctnes (interaction with "the environment")
            # 1. transform to words
            attempt1_probs = torch.argmax(probs_1, dim=-1)
            attempt1 = tokenizer.batch_decode(attempt1_probs, skip_special_tokens=True)

            kl_div = compute_kl_divergence(probs_base_1, probs_1)
            # Generate second attempt responses
            y1 = torch.argmax(probs_1, dim=-1)
            y1_decoded = tokenizer.batch_decode(y1, skip_special_tokens=True)
            # Estimate reward
            reward_1 = estimate_reward(y1_decoded, solutions)
            reward_1 = -(torch.mean(reward_1) - beta1 * torch.mean(kl_div))

            correction_inputs = [
                attempt + self_correction_prompt for attempt in attempt1
            ]
            correction_encodings = tokenizer(
                correction_inputs,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            # Combine original input_ids with correction_encodings
            combined_input_ids = torch.cat(
                (input_ids, correction_encodings["input_ids"].to(device)), dim=1
            )
            combined_attention_mask = torch.cat(
                (attention_mask, correction_encodings["attention_mask"].to(device)),
                dim=1,
            )

            # Second attempt (base model)
            with torch.no_grad():
                outputs_base_2 = base_model.generate(
                    combined_input_ids,
                    max_new_tokens=1000,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
                logits_base_2 = torch.stack(out_dict_base.logits).transpose(0, 1)
                probs_base_2 = torch.softmax(logits_base_2, dim=-1)

            # Second attempt (trained model)
            outputs_2 = model(
                combined_input_ids, attention_mask=combined_attention_mask
            )
            logits_2 = outputs_2.logits
            probs_2 = torch.softmax(logits_2, dim=-1)

            # Compute KL divergence between the first attempt of the base model and the trained model
            kl_div = compute_kl_divergence(probs_base_2, probs_2)

            # Generate second attempt responses
            y2 = torch.argmax(probs_2, dim=-1)
            y2_decoded = tokenizer.batch_decode(y2, skip_special_tokens=True)

            # Estimate reward
            reward_2 = estimate_reward(y2_decoded, solutions)
            reward_boni = reward_bonus(y2_decoded, y1_decoded, solutions)
            reward_2 = reward_2 + alpha * reward_boni

            action_log_probs = torch.log(
                torch.gather(probs_2, -1, y2.unsqueeze(-1))
            ).to("cpu")
            reward_2 = -(torch.mean(reward_2) * -BETA1 * torch.mean(kl_div))

            # Compute loss
            loss = (reward_1 + reward_2) * torch.mean(action_log_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
        )

    return model, total_loss


def evaluate_model(model, tokenizer, dataloader, device="cpu"):
    model.eval()
    total_reward = 0
    total_samples = 0
    correct_answers = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            input_ids = torch.stack(input_ids)
            input_ids = input_ids.transpose(0, 1).to(device)
            attention_mask = batch["attention_mask"]
            attention_mask = torch.stack(attention_mask)
            attention_mask = attention_mask.transpose(0, 1).to(device)

            solutions = batch["solution"]

            # First attempt (trained model)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            y_pred = torch.argmax(probs, dim=-1)
            y_pred_decoded = tokenizer.batch_decode(y_pred, skip_special_tokens=True)

            reward, correct = estimate_reward(
                extract_final_answer(y_pred_decoded),
                extract_final_answer(solutions),
                return_correctness=True,
            )

            if correct.item() == 1:
                correct_answers.append(correct)
                total_reward += torch.sum(reward).item()
                total_samples += len(solutions)
                continue
            else:
                # Second attempt (trained model)
                correction_inputs = [
                    attempt + self_correction_prompt for attempt in y_pred_decoded
                ]
                correction_encodings = tokenizer(
                    correction_inputs,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                # Combine original input_ids with correction_encodings
                combined_input_ids = torch.cat(
                    (input_ids, correction_encodings["input_ids"].to(device)), dim=1
                )
                combined_attention_mask = torch.cat(
                    (attention_mask, correction_encodings["attention_mask"].to(device)),
                    dim=1,
                )
                outputs_2 = model(
                    combined_input_ids, attention_mask=combined_attention_mask
                )
                logits = outputs_2.logits
                probs = torch.softmax(logits, dim=-1)
                y_pred = torch.argmax(probs, dim=-1)
                y_pred_decoded = tokenizer.batch_decode(
                    y_pred, skip_special_tokens=True
                )

                reward, correct = estimate_reward(
                    y_pred_decoded, solutions, return_correctness=True
                )
                correct_answers.append(correct)
                total_reward += torch.sum(reward).item()
                total_samples += len(solutions)

    correct_answers = torch.stack(correct_answers).float()
    accuracy = torch.mean(correct_answers)

    return total_reward, accuracy.item()


def main():
    wandb.init(project="SCoRe-v1")
    score_iterations = 4
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    dataloader, test_dataloader = load_and_prepare_data(DATASET_NAME, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model)
    # Load the base model for comparison
    base_model = (
        load_model_and_tokenizer(MODEL_NAME, return_tokenizer=False, quantize=True, lora=False)
        .to(device)
        .eval()
    )
    base_model = torch.compile(base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # TODO add scheduler with warmup steps

    for i in range(score_iterations):
        # total_reward, accuracy = evaluate_model(model, tokenizer, test_dataloader, device=device)

        model, stage_1_loss = train_stage_1(
            model,
            base_model,
            tokenizer,
            dataloader,
            1,
            optimizer,
            BETA2,
            device=device,
        )

        # model, stage_2_loss = train_stage_2(
        #     model,
        #     base_model,
        #     tokenizer,
        #     dataloader,
        #     1,
        #     optimizer,
        #     BETA1,
        #     ALPHA,
        #     device=device,
        # )

        # Combine all metrics into a single log
        wandb.log(
            {
                "stage_1_loss": stage_1_loss / len(dataloader),
                # "stage_2_loss": stage_2_loss / len(dataloader),
                "total_reward": total_reward,
                "accuracy": accuracy,
            },
            i,
        )
    total_reward, accuracy = evaluate_model(
        model, tokenizer, test_dataloader, device=device
    )
    wandb.log({"total_reward": total_reward, "accuracy": accuracy})
    # Save the trained model
    # trained_model.save_pretrained("score_stage_i_model")
    # tokenizer.save_pretrained("score_stage_i_model")
    # print("Training completed. Model saved as 'score_stage_i_model'.")


if __name__ == "__main__":
    main()
