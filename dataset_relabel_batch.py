import os
import openai
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import time

# Set your OpenAI API key
openai.api_key = "sk-QEvs87MTsegPblI1NHQmNoax7xv4TzcuZqt_z35ZTjT3BlbkFJ8CR7PhsLd-ymI2Kj16ECv4WOQq7Ow4p2tdzMPBC60A"

BATCH_SIZE = 10  # Number of prompts per batch

def extract_final_answers(solutions):
    # Prepare batched prompts for the API
    messages = [
        [
            {"role": "system", "content": "You are a math expert tasked with extracting the final numerical answer from a given solution."},
            {"role": "user", "content": f"""
                Given the following math solution, what is the final answer? 
                Provide only the answer in its simplest form, without any additional explanation.
                The answer could be a number, an expression, a set, or any other mathematical entity.

                Solution:
                {solution}

                Final answer:
                """}
        ] 
        for solution in solutions
    ]
    
    responses = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=30,
        n=1
    )
    
    # Extract and return answers from each response
    return [response['message']['content'].strip() for response in responses.choices]

def update_solution_batch(solutions):
    # Process solutions in batches
    final_answers = extract_final_answers(solutions)
    updated_solutions = [
        f"{solution}\nThe final answer is ${final_answer}$. I hope it is correct."
        for solution, final_answer in zip(solutions, final_answers)
    ]
    return updated_solutions

def update_dataset_split(dataset_split):
    updated_data = []
    solutions_batch = []
    items_batch = []
    
    for item in tqdm(dataset_split, desc=f"Updating solutions for {dataset_split.split}"):
        problem = item['problem']
        original_solution = item['solution']
        
        # Collect items into batches
        solutions_batch.append(original_solution)
        items_batch.append(item)
        
        # Process the batch when it reaches the batch size
        if len(solutions_batch) >= BATCH_SIZE:
            try:
                updated_solutions = update_solution_batch(solutions_batch)
                
                for original_item, updated_solution in zip(items_batch, updated_solutions):
                    updated_data.append({
                        "problem": original_item['problem'],
                        "solution": updated_solution,
                        "original_solution": original_item['solution']
                    })
            except Exception as e:
                print(f"Error processing batch: {e}")
                updated_data.extend(items_batch)  # Keep original items if there's an error
            
            solutions_batch = []
            items_batch = []

    # Process any remaining items in the last batch
    if solutions_batch:
        try:
            updated_solutions = update_solution_batch(solutions_batch)
            
            for original_item, updated_solution in zip(items_batch, updated_solutions):
                updated_data.append({
                    "problem": original_item['problem'],
                    "solution": updated_solution,
                    "original_solution": original_item['solution']
                })
        except Exception as e:
            print(f"Error processing final batch: {e}")
            updated_data.extend(items_batch)  # Keep original items if there's an error

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
