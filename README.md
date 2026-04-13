# LLMEvals

Small evaluation scripts and datasets for testing LLM responses on two tasks:
1) Customer support ticket classification (synthetic dataset).
2) Regulatory compliance Q&A evaluation with multiple scoring signals.

## Contents
- `evals.py`: Generates a small synthetic ticket dataset, runs a task model, and writes predictions.
- `evals_std_knowledge.py`: Runs a standards Q&A evaluation pipeline (task model, similarity, LLM grading, triad-style scores, and claim support analysis).
- `synthetic_eval_dataset.csv`: Synthetic ticket dataset used by `evals.py`.
- `model_results.csv`: Predictions and errors from `evals.py`.
- `std_eval_dataset.csv`: Standards Q&A dataset used by `evals_std_knowledge.py`.
- `std_eval_dataset_copy.csv`: Copy of the standards dataset written by the script.
- `std_model_results.csv`: Results from the standards evaluation run.
- `std_unsupported_claims.csv`: Per-question unsupported claims extracted during grading.

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` set in your environment or a local `.env` file

Suggested packages:
- `openai`
- `pydantic`
- `python-dotenv`

## Quick Start
1) Create and activate a virtual environment (recommended).
2) Install dependencies:

```bash
pip install openai pydantic python-dotenv
```

3) Set the API key:

```bash
export OPENAI_API_KEY="your_key_here"
```

4) Run the scripts:

```bash
python evals.py
python evals_std_knowledge.py
```

## What The Scripts Do

### `evals.py`
- Builds a small synthetic ticket dataset.
- Calls the task model to classify category, priority, and sentiment.
- Writes predictions to `model_results.csv`.

### `evals_std_knowledge.py`
- Loads `std_eval_dataset.csv` and writes a copy.
- Calls the task model to generate answers.
- Computes embedding similarity.
- Grades with an LLM and triad-style scores.
- Extracts and counts supported/unsupported claims.
- Writes results to `std_model_results.csv` and `std_unsupported_claims.csv`.

## Configuration Notes
- Models are defined at the top of each script (e.g., `TASK_MODEL`, `ANALYSIS_MODEL`).
- `MAX_ROWS` in `evals_std_knowledge.py` limits how many rows are evaluated.

## Output Files
- `model_results.csv`: Per-ticket predictions with any parsing errors.
- `std_model_results.csv`: Per-question scores and rationales.
- `std_unsupported_claims.csv`: Unsupported claims extracted from model outputs.

## Prompt Optimization Loop
Use `prompt_optimization.py` to build a prompt-optimization dataset from the eval results and optionally run a full optimize-evaluate loop.

1) Run a one-off dataset build:

```bash
python prompt_optimization.py
```

2) Run the loop (evaluates, writes optimization dataset, pauses for prompt updates):

```bash
python prompt_optimization.py --loop --iterations 3
```

3) Update the task prompt between iterations:
- Edit `task_prompt.txt`, or
- Set `TASK_PROMPT_FILE` to point to another prompt file.

4) Optional thresholds:

```bash
python prompt_optimization.py --llm-threshold 0.6 --faithfulness-threshold 0.75
```

## Prompt Optimizer (Dashboard Steps)
These are the steps used in the OpenAI dashboard to optimize the prompt with the dataset.

1) Upload the dataset:
- Create a dataset from `prompt_optimization_dataset.csv`.

2) Open the prompt:
- Go to Chat prompts, open the prompt (e.g., `StdKnowledgePrompt`).

3) Set the prompt messages:
- System message: contents of `task_prompt.txt`.
- Prompt message:

```text
Standard: {{standard}}
Question: {{query}}
```

4) Generate outputs:
- Click Generate output to populate the `output` column.

5) Annotate/evaluate:
- Use the rating (thumbs) to mark rows as Good/Bad.
- Optional: add notes in the feedback field if needed.

6) Optimize:
- Click Optimize to generate the optimized prompt.

7) Apply and re-run:
- Copy the optimized prompt into `task_prompt.txt`.
- Re-run `evals_std_knowledge.py` and `prompt_optimization.py`.

## Troubleshooting
- If you see `OPENAI_API_KEY not found`, ensure the environment variable or `.env` is set.
- If JSON parsing fails in `evals.py`, check the raw model response printed in the console.
