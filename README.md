> **_NOTE:_**  This repository is a work in progress. Changes and updates may occur as the project evolves.


# SCoRe: Self-Correct via Reinforcement Learning
Minimal implementation of the paper [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/pdf/2409.12917)



## Environment Setup

### 1. Create and Activate Conda Environment

To set up the environment for this project, follow the step in [unsloth](https://github.com/unslothai/unsloth)

### 2. Install Dependencies

Install the required packages using the `requirements.txt` file:

```
pip install -r requirements.txt
```


## Run SCoRe on Math Probelm

```
python score_math.py
```
`dataset_relabel.py` was used to add final answer pattern: ```'Final Answer: The final answer is $answer$. I hope it is correct.'``` 


#### TODOs:
- add eval [ ] 
- create SCoRe Trainer class [ ]
- cleanup code [ ]
- run experiments for math [ ]
