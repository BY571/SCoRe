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


# Multi-GPU-Training

```bash
   export CUDA_VISIBLE_DEVICES=0,1,2  # Use GPUs 0 and 1
   torchrun --nproc_per_node=2 your_training_script.py
```


## TODO:

- Test with bigger model (1B deep seek)
- Still nan?

- clean up code
- make env setup?
- combine stage 1-2 

- what does this mean
```
In some of our experiments, we also choose to
amplify the coverage of states used for on-policy RL by incorporating first-attempt solutions obtained
by repeatedly sampling the base model as offline prompts in RL. We find that incorporating this data,
especially in Stage II – where the first-turn policy may have drifted further from that of the base model –
can have substantial benefits especially when attempting to learn from limited data.```

-> Adding preprompts to second stage input from repeated sampled first attempts of the base model!
```

# Additional Info:
Fine tune Phis: [hugging face](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/sample_finetune.py)