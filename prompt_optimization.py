from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

RESULTS_CSV_DEFAULT = "std_model_results.csv"
UNSUPPORTED_CLAIMS_CSV_DEFAULT = "std_unsupported_claims.csv"
OUTPUT_CSV_DEFAULT = "prompt_optimization_dataset.csv"


@dataclass
class ResultRow:
    number: str
    standard: str
    question_level: str
    query: str
    gold_response: str
    pred_response: str
    similarity_score: Optional[float]
    llm_score: Optional[float]
    llm_rationale: Optional[str]
    answer_relevance_score: Optional[float]
    answer_relevance_rationale: Optional[str]
    context_relevance_score: Optional[float]
    context_relevance_rationale: Optional[str]
    faithfulness_score: Optional[float]
    faithfulness_rationale: Optional[str]
    error: Optional[str]


@dataclass
class UnsupportedClaimsRow:
    number: str
    unsupported_claims: str
    unsupported_claims_count: Optional[int]


@dataclass
class OptimizationRow:
    number: str
    standard: str
    question_level: str
    query: str
    gold_response: str
    pred_response: str
    llm_score: Optional[float]
    answer_relevance_score: Optional[float]
    context_relevance_score: Optional[float]
    faithfulness_score: Optional[float]
    unsupported_claims: Optional[str]
    label: str
    output_feedback: str


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_results(path: str) -> List[ResultRow]:
    rows: List[ResultRow] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ResultRow(
                    number=(row.get("number") or "").strip(),
                    standard=(row.get("standard") or "").strip(),
                    question_level=(row.get("question_level") or "").strip(),
                    query=(row.get("query") or "").strip(),
                    gold_response=(row.get("gold_response") or "").strip(),
                    pred_response=(row.get("pred_response") or "").strip(),
                    similarity_score=parse_float(row.get("similarity_score") or ""),
                    llm_score=parse_float(row.get("llm_score") or ""),
                    llm_rationale=(row.get("llm_rationale") or "").strip(),
                    answer_relevance_score=parse_float(row.get("answer_relevance_score") or ""),
                    answer_relevance_rationale=(row.get("answer_relevance_rationale") or "").strip(),
                    context_relevance_score=parse_float(row.get("context_relevance_score") or ""),
                    context_relevance_rationale=(row.get("context_relevance_rationale") or "").strip(),
                    faithfulness_score=parse_float(row.get("faithfulness_score") or ""),
                    faithfulness_rationale=(row.get("faithfulness_rationale") or "").strip(),
                    error=(row.get("error") or "").strip(),
                )
            )
    return rows


def load_unsupported_claims(path: str) -> Dict[str, UnsupportedClaimsRow]:
    if not os.path.exists(path):
        return {}
    rows: Dict[str, UnsupportedClaimsRow] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            number = (row.get("number") or "").strip()
            rows[number] = UnsupportedClaimsRow(
                number=number,
                unsupported_claims=(row.get("unsupported_claims") or "").strip(),
                unsupported_claims_count=parse_int(row.get("unsupported_claims_count") or ""),
            )
    return rows


def is_low_score(
    row: ResultRow,
    llm_threshold: float,
    faithfulness_threshold: float,
    answer_relevance_threshold: float,
    context_relevance_threshold: float,
) -> bool:
    if row.llm_score is not None and row.llm_score < llm_threshold:
        return True
    if row.faithfulness_score is not None and row.faithfulness_score < faithfulness_threshold:
        return True
    if row.answer_relevance_score is not None and row.answer_relevance_score < answer_relevance_threshold:
        return True
    if row.context_relevance_score is not None and row.context_relevance_score < context_relevance_threshold:
        return True
    return False


def build_feedback(row: ResultRow, unsupported_claims: Optional[str]) -> str:
    parts: List[str] = []
    if row.llm_rationale:
        parts.append(f"Correctness: {row.llm_rationale}")
    if row.answer_relevance_rationale:
        parts.append(f"Answer relevance: {row.answer_relevance_rationale}")
    if row.context_relevance_rationale:
        parts.append(f"Context relevance: {row.context_relevance_rationale}")
    if row.faithfulness_rationale:
        parts.append(f"Faithfulness: {row.faithfulness_rationale}")
    if unsupported_claims:
        parts.append(f"Unsupported claims: {unsupported_claims}")
    if not parts:
        parts.append("Low score detected; add detailed critique here.")
    return " | ".join(parts)


def build_optimization_rows(
    results: List[ResultRow],
    unsupported: Dict[str, UnsupportedClaimsRow],
    llm_threshold: float,
    faithfulness_threshold: float,
    answer_relevance_threshold: float,
    context_relevance_threshold: float,
) -> List[OptimizationRow]:
    output: List[OptimizationRow] = []
    for row in results:
        if not is_low_score(
            row,
            llm_threshold=llm_threshold,
            faithfulness_threshold=faithfulness_threshold,
            answer_relevance_threshold=answer_relevance_threshold,
            context_relevance_threshold=context_relevance_threshold,
        ):
            continue
        unsupported_row = unsupported.get(row.number)
        unsupported_claims = unsupported_row.unsupported_claims if unsupported_row else ""
        output.append(
            OptimizationRow(
                number=row.number,
                standard=row.standard,
                question_level=row.question_level,
                query=row.query,
                gold_response=row.gold_response,
                pred_response=row.pred_response,
                llm_score=row.llm_score,
                answer_relevance_score=row.answer_relevance_score,
                context_relevance_score=row.context_relevance_score,
                faithfulness_score=row.faithfulness_score,
                unsupported_claims=unsupported_claims or None,
                label="Bad",
                output_feedback=build_feedback(row, unsupported_claims),
            )
        )
    return output


def save_optimization_rows(rows: List[OptimizationRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "number",
                "standard",
                "question_level",
                "query",
                "gold_response",
                "pred_response",
                "llm_score",
                "answer_relevance_score",
                "context_relevance_score",
                "faithfulness_score",
                "unsupported_claims",
                "label",
                "output_feedback",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def resolve_output_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext or '.csv'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a prompt-optimization dataset from eval results.")
    parser.add_argument("--results", default=RESULTS_CSV_DEFAULT)
    parser.add_argument("--unsupported", default=UNSUPPORTED_CLAIMS_CSV_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_CSV_DEFAULT)
    parser.add_argument("--llm-threshold", type=float, default=0.7)
    parser.add_argument("--faithfulness-threshold", type=float, default=0.8)
    parser.add_argument("--answer-relevance-threshold", type=float, default=0.7)
    parser.add_argument("--context-relevance-threshold", type=float, default=0.7)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run evals and build optimization data in a loop.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of loop iterations to run when --loop is set.",
    )
    parser.add_argument(
        "--diff-old",
        help="Older std_model_results.csv to compare.",
    )
    parser.add_argument(
        "--diff-new",
        help="Newer std_model_results.csv to compare.",
    )
    parser.add_argument(
        "--diff-output",
        default="std_model_results_diff.csv",
        help="Output CSV path for the diff table.",
    )
    parser.add_argument(
        "--diff-summary",
        help="Write a human-readable diff summary to this file.",
    )
    return parser.parse_args()


def build_results_diff(old_path: str, new_path: str, output_path: str) -> None:
    fields = [
        "pred_response",
        "llm_score",
        "answer_relevance_score",
        "context_relevance_score",
        "faithfulness_score",
        "llm_rationale",
        "answer_relevance_rationale",
        "context_relevance_rationale",
        "faithfulness_rationale",
    ]

    with open(old_path, "r", newline="", encoding="utf-8") as f:
        old_rows = {row.get("number", "").strip(): row for row in csv.DictReader(f)}

    with open(new_path, "r", newline="", encoding="utf-8") as f:
        new_rows = {row.get("number", "").strip(): row for row in csv.DictReader(f)}

    diff_rows: List[Dict[str, str]] = []
    for number in sorted(set(old_rows) & set(new_rows)):
        old_row = old_rows[number]
        new_row = new_rows[number]
        for field in fields:
            if field not in old_row or field not in new_row:
                continue
            old_val = (old_row.get(field) or "").strip()
            new_val = (new_row.get(field) or "").strip()
            if old_val != new_val:
                diff_rows.append(
                    {
                        "number": number,
                        "field": field,
                        "old_value": old_val,
                        "new_value": new_val,
                    }
                )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["number", "field", "old_value", "new_value"],
        )
        writer.writeheader()
        for row in diff_rows:
            writer.writerow(row)


def write_diff_summary(diff_csv_path: str, summary_path: str) -> None:
    rows_by_number: Dict[str, List[Dict[str, str]]] = {}
    with open(diff_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            number = (row.get("number") or "").strip()
            if not number:
                continue
            rows_by_number.setdefault(number, []).append(row)

    def sort_key(value: str) -> tuple:
        return (0, int(value)) if value.isdigit() else (1, value)

    with open(summary_path, "w", encoding="utf-8") as f:
        for number in sorted(rows_by_number, key=sort_key):
            f.write(f"Row {number}\n")
            for row in rows_by_number[number]:
                field = (row.get("field") or "").strip()
                old_value = (row.get("old_value") or "").strip()
                new_value = (row.get("new_value") or "").strip()
                f.write(f"- {field}:\n")
                f.write(f"  old: {old_value}\n")
                f.write(f"  new: {new_value}\n")
            f.write("\n")


def run_evals() -> None:
    try:
        import evals_std_knowledge
    except Exception as exc:
        raise RuntimeError(
            "Failed to import evals_std_knowledge. Ensure it is in the same folder."
        ) from exc
    evals_std_knowledge.main()


def prompt_for_next_iteration(iteration: int, total: int, prompt_path: str) -> bool:
    if iteration >= total:
        return False
    print(
        f"\nUpdate your prompt in {prompt_path} (or set TASK_PROMPT_FILE), "
        "run the prompt optimizer if desired, then press Enter to continue."
    )
    response = input("Press Enter to continue or type 'q' to quit: ").strip().lower()
    return response != "q"


def main() -> None:
    args = parse_args()
    if args.diff_old and args.diff_new:
        build_results_diff(args.diff_old, args.diff_new, args.diff_output)
        print(f"Wrote diff to {args.diff_output}")
        if args.diff_summary:
            write_diff_summary(args.diff_output, args.diff_summary)
            print(f"Wrote diff summary to {args.diff_summary}")
        return
    iterations = max(args.iterations, 1)
    prompt_path = os.environ.get("TASK_PROMPT_FILE", "task_prompt.txt")

    for iteration in range(1, iterations + 1):
        if args.loop:
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            run_evals()

        results = load_results(args.results)
        unsupported = load_unsupported_claims(args.unsupported)
        optimization_rows = build_optimization_rows(
            results,
            unsupported,
            llm_threshold=args.llm_threshold,
            faithfulness_threshold=args.faithfulness_threshold,
            answer_relevance_threshold=args.answer_relevance_threshold,
            context_relevance_threshold=args.context_relevance_threshold,
        )
        output_path = resolve_output_path(args.output)
        save_optimization_rows(optimization_rows, output_path)
        print(f"Wrote {len(optimization_rows)} rows to {output_path}")

        if not args.loop:
            break

        continue_loop = prompt_for_next_iteration(iteration, iterations, prompt_path)
        if not continue_loop:
            print("Stopping loop early.")
            break


if __name__ == "__main__":
    main()
