import numpy as np
import torch
from datasets import load_dataset
from string_matcher import LLMAnswerComparator
from torch.utils.data import DataLoader, DistributedSampler
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
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"  # "gpt2"
DATASET_NAME = "lighteval/MATH"
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 256
BETA2 = 1.0
comparator = LLMAnswerComparator(threshold=0.9)


def setup_distributed_training():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_and_prepare_data(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name, split="train[:5%]")

    def tokenize_function(examples):
        examples["problem"] = [math_prompt + problem for problem in examples["problem"]]
        return tokenizer(
            examples["problem"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def compute_kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q), dim=-1)


def estimate_reward(y2, y_star):
    results = comparator.batch_compare(y2, oracle_responses=y_star, method="bert")
    similarities = [result[0] for result in results]
    return torch.tensor(similarities)


def train_score_stage_i(model, tokenizer, dataloader, num_epochs, learning_rate, beta2):
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")  # Set device based on rank
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model
    )  # Wrap model for distributed training
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Load the base model for comparison
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"]
            input_ids = torch.stack(input_ids).transpose(0, 1).to(device)
            attention_mask = batch["attention_mask"]
            attention_mask = torch.stack(attention_mask).transpose(0, 1).to(device)

            solutions = batch["solution"]

            # First attempt (base model)
            with torch.no_grad():
                outputs_base_1 = base_model(input_ids.to("cpu"), attention_mask=attention_mask.to("cpu"))
                logits_base_1 = outputs_base_1.logits
                probs_base_1 = torch.softmax(logits_base_1, dim=-1)

            # First attempt (trained model)
            outputs_1 = model(input_ids, attention_mask=attention_mask)
            logits_1 = outputs_1.logits
            probs_1 = torch.softmax(logits_1, dim=-1)

            # Add check for correctness (interaction with "the environment")
            attempt1_probs = torch.argmax(probs_1, dim=-1)
            attempt1 = tokenizer.batch_decode(attempt1_probs, skip_special_tokens=True)

            correction_inputs = [
                attempt + self_correction_prompt for attempt in attempt1
            ]
            correction_encodings = tokenizer(
                correction_inputs,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(
                device
            )  # Move to the correct device

            # Combine original input_ids with correction_encodings
            combined_input_ids = torch.cat(
                (input_ids, correction_encodings["input_ids"]), dim=1
            )
            combined_attention_mask = torch.cat(
                (attention_mask, correction_encodings["attention_mask"]), dim=1
            )

            # Second attempt (trained model)
            outputs_2 = model(
                combined_input_ids, attention_mask=combined_attention_mask
            )
            logits_2 = outputs_2.logits
            probs_2 = torch.softmax(logits_2, dim=-1)

            # Compute KL divergence
            kl_div = compute_kl_divergence(probs_base_1.to(device), probs_1)

            # Generate second attempt responses
            y2 = torch.argmax(probs_2, dim=-1)
            y2_decoded = tokenizer.batch_decode(y2, skip_special_tokens=True)

            # Estimate reward
            reward = estimate_reward(y2_decoded, solutions)

            # Compute loss
            action_log_probs = torch.log(
                torch.gather(probs_2, -1, y2.unsqueeze(-1))
            ).to("cpu")
            loss = -torch.mean(action_log_probs * reward) + beta2 * torch.mean(kl_div)

            print("Loss: ", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}"
        )

    return model


def main():
    setup_distributed_training()  # Initialize distributed training
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = load_and_prepare_data(DATASET_NAME, tokenizer)

    # Use DistributedSampler for the DataLoader
    sampler = DistributedSampler(tokenized_dataset)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    trained_model = train_score_stage_i(
        model, tokenizer, dataloader, NUM_EPOCHS, LEARNING_RATE, BETA2
    )

    # Save the trained model
    # trained_model.save_pretrained("score_stage_i_model")
    # tokenizer.save_pretrained("score_stage_i_model")
    # print("Training completed. Model saved as 'score_stage_i_model'.")


if __name__ == "__main__":
    main()
