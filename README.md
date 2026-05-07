# SCoRe: Self-Correct via Reinforcement Learning

Minimal, task-agnostic implementation of [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) (Kumar et al., 2024) using [Unsloth](https://github.com/unslothai/unsloth) for efficient 4-bit LoRA training.

## Layout

```
train.py              # task-agnostic entry point
reward_function.py    # registry of reward functions and answer extractors
configs/math.yaml     # example config (math task)
```

## Setup

Follow the [Unsloth install guide](https://github.com/unslothai/unsloth#-installation-instructions) for your CUDA version, then:

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py --config configs/math.yaml
```

Logs go to W&B (project from `wandb_project` in the config). LoRA adapters are saved to `outputs/{run_name}/stage{1,2}/`.

## Adapt to a new task

Everything task-specific lives in the YAML and the reward registry. To add a new task:

1. **Push your dataset to the HuggingFace Hub** with `train` and `test` splits, an input column (e.g. `problem`) and a target column (e.g. `solution`).
2. **Add an answer extractor** to `reward_function.py` if the default `math_final_answer` or `identity` extractors don't match your target format:
    ```python
    @register_extractor("my_extractor")
    def my_extractor(text: str) -> str:
        ...
    ```
3. **Add a reward function** if `exact_match` (string equality after extraction) isn't enough:
    ```python
    @register_reward("my_reward")
    def my_reward(predictions: list[str], targets: list[str]) -> list[float]:
        ...
    ```
4. **Copy `configs/math.yaml`**, change `dataset.*`, `prompts.*`, and `reward.*` to match your task.
5. `python train.py --config configs/yours.yaml`.

## Dataset format contract

The dataset must have:

- A `train` and a `test` split.
- A column named by `dataset.input_field` (string): the question / prompt.
- A column named by `dataset.target_field` (string): the ground-truth answer in a form the chosen `answer_extractor` can parse. For example, with the `math_final_answer` extractor, the target should contain `"the final answer is: <answer>"` or a `\boxed{<answer>}` somewhere in the text.

`train.py` only reads these two fields — anything else in the dataset is ignored.

## How the math example dataset was built

`Sebasdi/math_final_answer` is `lighteval/MATH` with each solution suffixed by an extracted final-answer marker (so the extractor has something to parse). The script that produced it (`dataset_relabel.py`) used GPT-3.5 to extract the final answer from each MATH solution; it is preserved in git history if regeneration is ever needed.
