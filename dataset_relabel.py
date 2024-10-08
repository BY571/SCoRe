import os
from openai import OpenAI

from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import time  # Added import for time module

with open('api.txt', 'r') as file:
    client = OpenAI(
        # This is the default and can be omitted
        api_key=file.read().strip(),
    )

def extract_final_answer(solution):
    prompt = f"""
    Given the following math solution, what is the final answer? 
    Provide only the answer in its simplest form, without any additional explanation.
    The answer could be a number, an expression, a set, or any other mathematical entity.

    Solution:
    {solution}

    Final answer:
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a math expert tasked with extracting the final numerical answer from a given solution."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20,  # Limiting tokens as we expect a short numerical answer
        n=1,
    )
    
    return response.choices[0].message.content.strip()

def update_solution(solution):
    time.sleep(0.15)  # Wait for 0.126 seconds (approximately 7.936 requests per second)
    final_answer = extract_final_answer(solution)
    return f"{solution}\nThe final answer is ${final_answer}$. I hope it is correct."

def update_dataset_split(dataset_split):
    updated_data = []
    
    for item in tqdm(dataset_split, desc=f"Updating solutions for {dataset_split.split}"):
        original_solution = item['solution']
        
        try:
            # Update the solution using the existing function
            updated_solution = update_solution(original_solution)
            
            # Include all original fields in the updated data
            updated_data.append({
                **item,  # Copy all original fields
                "final_answer_solution": updated_solution,  # Update the solution field with a new name
            })
        except Exception as e:
            print(f"Error processing item: {e}")
            updated_data.append(item)  # Keep original item if there's an error

    return dataset_split.from_list(updated_data)

def update_dataset():
    # Load both train and test splits of the dataset
    dataset = load_dataset("lighteval/MATH")
    
    # Update each split
    updated_train = update_dataset_split(dataset['train'])
    updated_test = update_dataset_split(dataset['test'])
    
    # Combine updated splits into a new dataset
    updated_dataset = DatasetDict({
        'train': updated_train,
        'test': updated_test
    })
    
    # Save the updated dataset
    updated_dataset.save_to_disk("updated_math_dataset")
    
    print("Dataset updated and saved to 'updated_math_dataset' directory")

if __name__ == "__main__":
    update_dataset()