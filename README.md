# SCoRe
Minimal implementation of the paper [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/pdf/2409.12917)



## Environment Setup

### 1. Create and Activate Conda Environment

To set up the environment for this project, follow these steps:

1. Create a new conda environment named "llmrl" with Python 3.9:
   ```
   conda create -n score python=3.9
   ```

2. Activate the environment:
   ```
   conda activate score
   ```

### 2. Install Dependencies

Install the required packages using the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Run to test on toy problem

```
python score_toy.py
```

## Run SCoRe on Math Probelm

```
python score_math.py
```
`dataset_relabel.py` was used to add final answer pattern: ```'Final Answer: The final answer is $answer$. I hope it is correct.'``` 