import os
import openai
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

# Load the API key from api.txt
def load_api_key():
    with open('api.txt', 'r') as file:
        return file.read().strip()

# Set your OpenAI API key
openai.api_key = load_api_key()

def extract_final_answer(solution):
    prompt = f"""
    Given the following math solution, what is the final answer? 
    Provide only the answer in its simplest form, without any additional explanation.
    The answer could be a number, an expression, a set, or any other mathematical entity.

    Solution:
    {solution}

    Final answer:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a math expert tasked with extracting the final numerical answer from a given solution."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=30  # Limiting tokens as we expect a short numerical answer
    )
    
    return response.choices[0].message['content'].strip()

def update_solution(solution):
    final_answer = extract_final_answer(solution)
    return f"{solution}\nThe final answer is ${final_answer}$. I hope it is correct."

def update_dataset_split(dataset_split):
    updated_data = []
    
    for item in tqdm(dataset_split, desc=f"Updating solutions for {dataset_split.split}"):
        problem = item['problem']
        original_solution = item['solution']
        
        try:
            updated_solution = update_solution(original_solution)
            
            updated_item = {
                "problem": problem,
                "solution": updated_solution,
                "original_solution": original_solution
            }
            updated_data.append(updated_item)
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