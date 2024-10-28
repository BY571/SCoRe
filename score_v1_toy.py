import os
import re
import numpy as np
import torch
import wandb
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from string_matcher import LLMAnswerComparator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F


# Prompts
prompt = """You are provided with a sequence of numbers. Your task is to identify the logical pattern within the sequence and provide a logical conclusion of what the next number in the sequence should be.
You can think and reason about what the answer will be but provide as a final answer only the numerical value and in the following way:

'Final Answer: The final answer is $answer$.'"""

self_correction_prompt = """There might be an error in the solution above because of lack of understanding of the question. Please correct
the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you
give your final answer, write it in the form 'Final Answer: The final answer is $answer$.' """ 


# Hyperparameters
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" #HF1BitLLM/Llama3-8B-1.58-100B-tokens  microsoft/Phi-3-mini-4k-instruct
DATASET_NAME = "Sebasdi/score_toy_dataset"
BATCH_SIZE = 11
LEARNING_RATE = 5e-6
NUM_EPOCHS = 10
MAX_LENGTH_BATCH = 150 # Tokenization -> input batch seq length
mnt_attempt1 = 75
mnt_attempt2 = 75
x2_max_batch_len = 250
BETA1 = 0.01
BETA2 = 0.1
ALPHA = 10  # 𝛼 is a positive constant multiplier, ideally larger than 1.0
comparator = LLMAnswerComparator(threshold=0.9)

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

def load_model_and_tokenizer(model_name: str, return_tokenizer: bool = True, quantize: bool = True, lora: bool = True):
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
        #attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support

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

    # def tokenize_function(examples, tokenizer):
    #     # add the math preprompt to all problems
    #     examples["sequence"] = [prompt + sequence_prompt.format(sequence=seq) for seq in examples["sequence"]]
    #     return tokenizer(
    #         examples["sequence"],
    #         truncation=True,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    def apply_chat_template(examples, tokenizer):
        # add the math preprompt to all problems
        messages = [[{"role": "user",
                     "content": prompt + sequence_prompt.format(sequence=seq)} ]for seq in examples["sequence"]] 

        return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True, truncation=True)

    # Max length is 113 with padding!
    tokenized_dataset = dataset.map(apply_chat_template,
                                    fn_kwargs={"tokenizer": tokenizer},
                                    batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["explanation", "sequence"]) # remove explanation
    train_dataloader = DataLoader(
        tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    return train_dataloader


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * torch.log(p / q), dim=-1)


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


def estimate_reward(y2: list, solutions: list) -> torch.Tensor:
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
    total_loss = 0  # Initialize total_loss here
    total_correct = 0  # Initialize total correct predictions
    total_samples = 0  # Initialize total samples for accuracy calculation

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # Initialize epoch_loss for each epoch
        total_batches = len(dataloader)
        mean_reward_a1 = []
        mean_reward_a2 = []
        i = 0
        for i, batch in tqdm(enumerate(dataloader), total=total_batches, desc=f"Stage 1 Training {i+1}/{total_batches}"):
            x1 = batch["input_ids"]
            # reshape (seq_len, batch)-> (batch,seq_len)
            x1 = torch.stack(x1).transpose(0, 1).to(model.device)
            attention_mask = batch["attention_mask"]
            # reshape (seq_len, batch)-> (batch,seq_len)
            attention_mask = torch.stack(attention_mask).transpose(0, 1).to(model.device)

            solutions = batch["correct_answer"]

            # First attempt (trained model) get on-policy action of the train model
            with torch.no_grad():
                action1_token = model.generate(x1,
                                            attention_mask=attention_mask,
                                            max_new_tokens=mnt_attempt1,
                                            use_cache=True,
                                            do_sample=True,
                                            temperature=1.0)
            action1 = tokenizer.batch_decode(action1_token, skip_special_tokens=True)
            reward1 = estimate_reward(action1, solutions)
            mean_reward_a1.append(reward1)

            x2 = [a1 + self_correction_prompt for a1 in action1]
            x2_encoded = tokenizer(x2,
                                   padding="max_length",
                                   truncation=True,
                                   max_length=x2_max_batch_len,
                                   return_tensors="pt")
            input_ids_2 = x2_encoded.input_ids.to(model.device)
            attention_mask_2 = x2_encoded.attention_mask.to(model.device)
            action2_token = model.generate(input_ids_2,
                                           attention_mask=attention_mask_2,
                                           max_new_tokens=mnt_attempt2,
                                           use_cache=True,
                                           do_sample=True,
                                           temperature=1.0)


            # Compute reward:
            # Calculate accuracy
            input_length = input_ids_2.shape[1]
            # Extract only the newly generated tokens
            attempt2_answer_tokens = action2_token[:, input_length:]
            attempt2_answer = tokenizer.batch_decode(attempt2_answer_tokens, skip_special_tokens=True)
            reward2 = estimate_reward(attempt2_answer, solutions)
            mean_reward_a2.append(reward2)

            # Compute the loss
            with torch.no_grad():
                base_logits = base_model(x1,
                                         attention_mask=attention_mask).logits
                probs_base_1 = torch.softmax(base_logits, dim=-1)

            # Compute logprobs of the first attempt of the trained model
            logits_1 = model(x1,
                             attention_mask=attention_mask).logits
            probs_1 = torch.softmax(logits_1, dim=-1)

            # Compute KL divergence between the first attempt of the base model and the trained model
            kl_div = F.kl_div(probs_1, probs_base_1, reduction='mean')

            logits_2 = model(attempt2_answer_tokens).logits

            # Compute policy gradient loss using reward for y2_logits
            log_probs = torch.log_softmax(logits_2, dim=-1)
            action_log_probs = torch.gather(log_probs, -1, attempt2_answer_tokens.unsqueeze(-1)).sum(1).mean()


            # Compute the loss
            loss = -action_log_probs * (reward2 + beta2 * torch.mean(kl_div).to("cpu"))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_loss += loss.item()

            total_samples += len(solutions)

            print("Loss: ", loss.item())

        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {epoch_accuracy:.4f}")
        mean_reward_attempt_1 = np.mean(mean_reward_a1)
        mean_reward_attempt_2 = np.mean(mean_reward_a2)

        diff_at1_at2 = mean_reward_attempt_2 - mean_reward_attempt_1
        wandb.log({"mr_attempt1": mean_reward_attempt_1,
                   "mr_attempt2": mean_reward_attempt_2,
                   "difference_at1_at2": diff_at1_at2})


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
                    max_new_tokens=max_new_tokens,
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
    wandb.init(project="SCoRe-v1-toy")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, quantize=True, lora=True)
    dataloader = load_and_prepare_data(DATASET_NAME, tokenizer)
    
    # Load the base model for comparison
    base_model = load_model_and_tokenizer(MODEL_NAME, return_tokenizer=False, quantize=True, lora=False).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for param in base_model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameter: ", total_params)

    model= train_stage_1(
        model=model,
        base_model=base_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        num_epochs=3,
        optimizer=optimizer,
        beta2=BETA2)

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

    # total_reward, accuracy = evaluate_model(
    #     model, tokenizer, test_dataloader, device=device
    # )
    #wandb.log({"total_reward": total_reward, "accuracy": accuracy})
    # Save the trained model
    # trained_model.save_pretrained("score_stage_i_model")
    # tokenizer.save_pretrained("score_stage_i_model")
    # print("Training completed. Model saved as 'score_stage_i_model'.")


if __name__ == "__main__":
    main()