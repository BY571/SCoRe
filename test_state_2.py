import numpy as np
import torch
from datasets import load_dataset
from string_matcher import LLMAnswerComparator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer

math_prompt = """You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by
step. At the end of the Solution, when you give your final answer, write it in the form 'Final Answer: The final
answer is $answer$. I hope it is correct.'"""

self_correction_prompt = """There might be an error in the solution above because of lack of understanding of the question. Please correct
the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you
give your final answer, write it in the form 'Final Answer: The final answer is $answer$. I hope it is correct.'
"""

# Hyperparameters
MODEL_NAME = "gpt2"  # "deepseek-ai/deepseek-coder-1.3b-instruct"
DATASET_NAME = "lighteval/MATH"
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 256
BETA1 = 0.01
ALPHA = 2.0  # 𝛼 is a positive constant multiplier, ideally larger than 1.0
comparator = LLMAnswerComparator(threshold=0.9)


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_and_prepare_data(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name, split="train[:5%]")

    def tokenize_function(examples):
        # add the math preprompt to all problems
        examples["problem"] = [math_prompt + problem for problem in examples["problem"]]
        return tokenizer(
            examples["problem"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)


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


def compute_kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q), dim=-1)


def estimate_reward(y, y_star):

    results = comparator.batch_compare(y, oracle_responses=y_star, method="bert")
    # results is a list of tuples (similarity, correct)
    similarities = [result[0] for result in results]
    return torch.tensor(similarities)


def train_score_stage_i(model, tokenizer, dataloader, num_epochs, learning_rate, beta1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Load the base model for comparison
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

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
                outputs_base_1 = base_model(input_ids, attention_mask=attention_mask)
                logits_base_1 = outputs_base_1.logits
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
            reward_1 = -(torch.mean(reward_1) - BETA1 * torch.mean(kl_div))

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
                outputs_base_2 = base_model(
                    combined_input_ids, attention_mask=combined_attention_mask
                )
                logits_base_2 = outputs_base_2.logits
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
            reward_2 = reward_2 + ALPHA * reward_boni

            reward_2 = -(torch.mean(reward_2) - BETA1 * torch.mean(kl_div))

            # Compute loss
            loss = reward_1 + reward_2
            print("Loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}"
        )

    return model


def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = load_and_prepare_data(DATASET_NAME, tokenizer)

    trained_model = train_score_stage_i(
        model, tokenizer, dataloader, NUM_EPOCHS, LEARNING_RATE, BETA1
    )

    # Save the trained model
    # trained_model.save_pretrained("score_stage_i_model")
    # tokenizer.save_pretrained("score_stage_i_model")
    # print("Training completed. Model saved as 'score_stage_i_model'.")


if __name__ == "__main__":
    main()