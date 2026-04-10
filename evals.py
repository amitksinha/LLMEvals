from __future__ import annotations

import json
import os
import random
import traceback
from dataclasses import dataclass, asdict
from typing import List, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


print("Script started...")

TASK_MODEL = "gpt-5.4"
ANALYSIS_MODEL = "gpt-4o-mini"
SEED = 7
NUM_EXAMPLES = 5

Category = Literal["billing", "technical", "account", "other"]
Priority = Literal["low", "medium", "high"]
Sentiment = Literal["negative", "neutral", "positive"]


class TicketOutput(BaseModel):
    category: Category
    priority: Priority
    customer_sentiment: Sentiment
    short_rationale: str = Field(min_length=3, max_length=200)


@dataclass
class Example:
    id: str
    ticket_text: str
    gold_category: Category
    gold_priority: Priority
    gold_sentiment: Sentiment


@dataclass
class ResultRow:
    id: str
    ticket_text: str
    gold_category: Category
    gold_priority: Priority
    gold_sentiment: Sentiment
    pred_category: str | None
    pred_priority: str | None
    pred_sentiment: str | None
    short_rationale: str | None
    error: str | None


def make_synthetic_dataset(n: int, seed: int = 7) -> List[Example]:
    random.seed(seed)

    examples = [
        Example(
            id="ex_001",
            ticket_text="I was charged twice for my subscription and need a refund.",
            gold_category="billing",
            gold_priority="high",
            gold_sentiment="negative",
        ),
        Example(
            id="ex_002",
            ticket_text="The mobile app crashes every time I upload a PDF.",
            gold_category="technical",
            gold_priority="high",
            gold_sentiment="negative",
        ),
        Example(
            id="ex_003",
            ticket_text="I cannot sign in even after resetting my password twice.",
            gold_category="account",
            gold_priority="high",
            gold_sentiment="negative",
        ),
        Example(
            id="ex_004",
            ticket_text="Do you offer nonprofit discounts for annual plans?",
            gold_category="other",
            gold_priority="low",
            gold_sentiment="neutral",
        ),
        Example(
            id="ex_005",
            ticket_text="Please update the email on my account because I lost access to the old one.",
            gold_category="account",
            gold_priority="medium",
            gold_sentiment="neutral",
        ),
    ]
    return examples[:n]


PROMPT = """You classify customer support tickets.

Return valid JSON only with these keys:
- category: one of billing, technical, account, other
- priority: one of low, medium, high
- customer_sentiment: one of negative, neutral, positive
- short_rationale: brief explanation
"""


def get_client() -> OpenAI:
    print("Loading environment variables...")
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")

    print("Creating OpenAI client...")
    return OpenAI(api_key=api_key)


def run_task_model(client: OpenAI, prompt: str, ticket_text: str):
    print(f"Calling model for ticket: {ticket_text[:60]}...")

    response = client.responses.create(
        model=TASK_MODEL,
        instructions=prompt,
        input=f"Ticket:\n{ticket_text}",
    )

    print("Raw response object received.")
    raw = response.output_text
    print("Raw output text:")
    print(raw)

    parsed = json.loads(raw)
    obj = TicketOutput.model_validate(parsed)
    return obj


def main():
    print("Inside main()")

    client = get_client()
    examples = make_synthetic_dataset(NUM_EXAMPLES, seed=SEED)
    save_dataset_csv(examples, "synthetic_eval_dataset.csv")

    print(f"Generated {len(examples)} examples.")

    results: List[ResultRow] = []

    for ex in examples:
        print("\n-----------------------------------")
        print("Example ID:", ex.id)
        print("Ticket:", ex.ticket_text)
        print("Gold:", ex.gold_category, ex.gold_priority, ex.gold_sentiment)

        try:
            pred = run_task_model(client, PROMPT, ex.ticket_text)
            print("Parsed prediction:")
            print(pred.model_dump())
            results.append(
                ResultRow(
                    id=ex.id,
                    ticket_text=ex.ticket_text,
                    gold_category=ex.gold_category,
                    gold_priority=ex.gold_priority,
                    gold_sentiment=ex.gold_sentiment,
                    pred_category=pred.category,
                    pred_priority=pred.priority,
                    pred_sentiment=pred.customer_sentiment,
                    short_rationale=pred.short_rationale,
                    error="no_proc_error",
                )
            )
        except json.JSONDecodeError as e:
            print("JSON parse error:")
            print(str(e))
            results.append(
                ResultRow(
                    id=ex.id,
                    ticket_text=ex.ticket_text,
                    gold_category=ex.gold_category,
                    gold_priority=ex.gold_priority,
                    gold_sentiment=ex.gold_sentiment,
                    pred_category=None,
                    pred_priority=None,
                    pred_sentiment=None,
                    short_rationale=None,
                    error=f"json_error: {e}",
                )
            )
        except ValidationError as e:
            print("Pydantic validation error:")
            print(str(e))
            results.append(
                ResultRow(
                    id=ex.id,
                    ticket_text=ex.ticket_text,
                    gold_category=ex.gold_category,
                    gold_priority=ex.gold_priority,
                    gold_sentiment=ex.gold_sentiment,
                    pred_category=None,
                    pred_priority=None,
                    pred_sentiment=None,
                    short_rationale=None,
                    error=f"validation_error: {e}",
                )
            )
        except Exception as e:
            print("Unexpected error during model call:")
            print(str(e))
            traceback.print_exc()
            results.append(
                ResultRow(
                    id=ex.id,
                    ticket_text=ex.ticket_text,
                    gold_category=ex.gold_category,
                    gold_priority=ex.gold_priority,
                    gold_sentiment=ex.gold_sentiment,
                    pred_category=None,
                    pred_priority=None,
                    pred_sentiment=None,
                    short_rationale=None,
                    error=f"unexpected_error: {e}",
                )
            )

    save_results_csv(results, "model_results.csv")

    print("\nDone.")


def save_dataset_csv(examples: List[Example], path: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "ticket_text", "gold_category", "gold_priority", "gold_sentiment"],
        )
        writer.writeheader()
        for ex in examples:
            writer.writerow(asdict(ex))

    print(f"Saved dataset to {path}")


def save_results_csv(results: List[ResultRow], path: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "ticket_text",
                "gold_category",
                "gold_priority",
                "gold_sentiment",
                "pred_category",
                "pred_priority",
                "pred_sentiment",
                "short_rationale",
                "error",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    print(f"Saved results to {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:")
        print(str(e))
        traceback.print_exc()