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

## Troubleshooting
- If you see `OPENAI_API_KEY not found`, ensure the environment variable or `.env` is set.
- If JSON parsing fails in `evals.py`, check the raw model response printed in the console.
