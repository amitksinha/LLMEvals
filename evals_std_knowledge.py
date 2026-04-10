from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


TASK_MODEL = "gpt-5.4"
ANALYSIS_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
INPUT_CSV = "std_eval_dataset.csv"
DATASET_COPY_CSV = "std_eval_dataset_copy.csv"
RESULTS_CSV = "std_model_results.csv"
UNSUPPORTED_CLAIMS_CSV = "std_unsupported_claims.csv"
MAX_ROWS = 5


class LLMGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=3, max_length=500)


class TriadGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=3, max_length=500)


class ClaimsList(BaseModel):
    claims: List[str]


class ClaimSupport(BaseModel):
    supported: List[str]
    unsupported: List[str]
    not_enough_info: List[str]


@dataclass
class StdRecord:
    number: str
    standard: str
    question_level: str
    query: str
    response: str


@dataclass
class ResultRow:
    number: str
    standard: str
    question_level: str
    query: str
    gold_response: str
    pred_response: str | None
    similarity_score: float | None
    llm_score: float | None
    llm_rationale: str | None
    answer_relevance_score: float | None
    answer_relevance_rationale: str | None
    context_relevance_score: float | None
    context_relevance_rationale: str | None
    faithfulness_score: float | None
    faithfulness_rationale: str | None
    claim_count: int | None
    supported_claims_count: int | None
    unsupported_claims_count: int | None
    not_enough_info_claims_count: int | None
    unsupported_claims: str | None
    error: str | None


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")
    return OpenAI(api_key=api_key)


def load_std_dataset(path: str) -> List[StdRecord]:
    records: List[StdRecord] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            number = str(row.get("Number", row.get("number", ""))).strip()
            standard = str(row.get("Standard", row.get("standard", ""))).strip()
            question_level = str(row.get("Question Level", row.get("question_level", ""))).strip()
            query = str(row.get("Query", row.get("query", ""))).strip()
            response = str(row.get("Response", row.get("response", ""))).strip()
            if not any([number, standard, question_level, query, response]):
                continue
            records.append(
                StdRecord(
                    number=number,
                    standard=standard,
                    question_level=question_level,
                    query=query,
                    response=response,
                )
            )
    return records


def save_dataset_copy(records: List[StdRecord], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["number", "standard", "question_level", "query", "response"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def run_task_model(client: OpenAI, record: StdRecord) -> str:
    prompt = (
        "You answer regulatory compliance questions."
        " Provide a concise, accurate response."
        " Use plain text only."
    )
    input_text = f"Standard: {record.standard}\nQuestion: {record.query}"

    response = client.responses.create(
        model=TASK_MODEL,
        instructions=prompt,
        input=input_text,
    )
    return response.output_text.strip()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_similarity(client: OpenAI, text_a: str, text_b: str) -> float:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text_a, text_b],
    )
    vec_a = resp.data[0].embedding
    vec_b = resp.data[1].embedding
    return cosine_similarity(vec_a, vec_b)


def grade_with_llm(client: OpenAI, record: StdRecord, pred: str) -> LLMGrade:
    system = (
        "You grade answers to regulatory questions."
        " Compare the predicted answer to the gold answer."
        " Score 0 to 1 based on correctness and coverage."
        " Provide a brief rationale."
    )
    user = {
        "question": record.query,
        "gold_answer": record.response,
        "predicted_answer": pred,
        "scoring": {
            "1.0": "Fully correct and complete",
            "0.7": "Mostly correct with minor omissions",
            "0.4": "Partially correct but missing key points",
            "0.0": "Incorrect or unrelated",
        },
    }

    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, indent=2)},
        ],
        text_format=LLMGrade,
    )
    return response.output_parsed


def grade_answer_relevance(client: OpenAI, question: str, answer: str) -> TriadGrade:
    system = (
        "You grade answer relevance."
        " Score how well the answer addresses the question."
        " Use 0 to 1 with a short rationale."
    )
    user = {
        "question": question,
        "answer": answer,
        "scoring": {
            "1.0": "Directly answers the question fully",
            "0.7": "Mostly relevant with minor gaps",
            "0.4": "Partially relevant or tangential",
            "0.0": "Irrelevant",
        },
    }
    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, indent=2)},
        ],
        text_format=TriadGrade,
    )
    return response.output_parsed


def grade_context_relevance(client: OpenAI, question: str, context: str) -> TriadGrade:
    system = (
        "You grade context relevance."
        " Score how useful the context is for answering the question."
        " Use 0 to 1 with a short rationale."
    )
    user = {
        "question": question,
        "context": context,
        "scoring": {
            "1.0": "Context contains the key facts needed",
            "0.7": "Mostly relevant but missing some key facts",
            "0.4": "Some relevant info but mostly off-topic",
            "0.0": "Not relevant",
        },
    }
    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, indent=2)},
        ],
        text_format=TriadGrade,
    )
    return response.output_parsed


def grade_faithfulness(client: OpenAI, answer: str, context: str) -> TriadGrade:
    claims = extract_claims(client, answer)
    support = check_claim_support(client, claims, context)

    total_claims = len(claims)
    if total_claims == 0:
        return TriadGrade(score=1.0, rationale="No claims extracted from answer.")

    supported_count = len(support.supported)
    score = supported_count / total_claims
    rationale = (
        f"Supported {supported_count}/{total_claims} claims; "
        f"unsupported: {len(support.unsupported)}; "
        f"not enough info: {len(support.not_enough_info)}."
    )
    return TriadGrade(score=score, rationale=rationale)


def extract_claims(client: OpenAI, answer: str) -> List[str]:
    system = (
        "Extract atomic factual claims from the answer."
        " Each claim should be a short, standalone sentence."
        " Do not include opinions or recommendations."
    )
    user = {"answer": answer}
    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, indent=2)},
        ],
        text_format=ClaimsList,
    )
    claims = [c.strip() for c in response.output_parsed.claims if c and c.strip()]
    return claims


def check_claim_support(client: OpenAI, claims: List[str], context: str) -> ClaimSupport:
    system = (
        "Classify each claim as supported, unsupported, or not enough info,"
        " using only the provided context."
    )
    user = {
        "context": context,
        "claims": claims,
        "labels": {
            "supported": "Claim is directly supported by context",
            "unsupported": "Claim is contradicted or not supported",
            "not_enough_info": "Context does not contain enough to verify",
        },
    }
    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, indent=2)},
        ],
        text_format=ClaimSupport,
    )
    return response.output_parsed


def save_results(results: List[ResultRow], path: str) -> None:
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
                "similarity_score",
                "llm_score",
                "llm_rationale",
                "answer_relevance_score",
                "answer_relevance_rationale",
                "context_relevance_score",
                "context_relevance_rationale",
                "faithfulness_score",
                "faithfulness_rationale",
                "claim_count",
                "supported_claims_count",
                "unsupported_claims_count",
                "not_enough_info_claims_count",
                "unsupported_claims",
                "error",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def save_unsupported_claims(results: List[ResultRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "number",
                "query",
                "faithfulness_score",
                "unsupported_claims_count",
                "unsupported_claims",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "number": row.number,
                    "query": row.query,
                    "faithfulness_score": row.faithfulness_score,
                    "unsupported_claims_count": row.unsupported_claims_count,
                    "unsupported_claims": row.unsupported_claims,
                }
            )


def classify_error(err: Exception) -> str:
    msg = str(err).lower()
    if "rate limit" in msg or "rate_limit" in msg or "429" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "service unavailable" in msg or "503" in msg:
        return "service_unavailable"
    if "connection" in msg or "network" in msg:
        return "network"
    return "other"


def main() -> None:
    client = get_client()
    records = load_std_dataset(INPUT_CSV)
    if MAX_ROWS is not None:
        records = records[:MAX_ROWS]
    save_dataset_copy(records, DATASET_COPY_CSV)

    results: List[ResultRow] = []
    total = len(records)

    for idx, record in enumerate(records, start=1):
        print(f"\n[{idx}/{total}] {record.number} | {record.standard} | {record.question_level}")
        print(f"Question: {record.query[:120]}{'...' if len(record.query) > 120 else ''}")
        pred_response: str | None = None
        similarity_score: float | None = None
        llm_score: float | None = None
        llm_rationale: str | None = None
        answer_relevance_score: float | None = None
        answer_relevance_rationale: str | None = None
        context_relevance_score: float | None = None
        context_relevance_rationale: str | None = None
        faithfulness_score: float | None = None
        faithfulness_rationale: str | None = None
        claim_count: int | None = None
        supported_claims_count: int | None = None
        unsupported_claims_count: int | None = None
        not_enough_info_claims_count: int | None = None
        unsupported_claims: str | None = None
        errors: List[str] = []

        try:
            print("Running task model...")
            pred_response = run_task_model(client, record)
            print("Task model done.")
        except Exception as e:
            error_type = classify_error(e)
            errors.append(f"task_error[{error_type}]: {e}")
            print(f"Task model error ({error_type}): {e}")

        if pred_response:
            try:
                # print("Computing similarity...")
                similarity_score = compute_similarity(
                    client,
                    record.response,
                    pred_response,
                )
                print(f"Similarity score: {similarity_score:.4f}")
            except Exception as e:
                error_type = classify_error(e)
                errors.append(f"similarity_error[{error_type}]: {e}")
                print(f"Similarity error ({error_type}): {e}")

            try:
                # print("Running LLM grader...")
                grade = grade_with_llm(client, record, pred_response)
                llm_score = grade.score
                llm_rationale = grade.rationale
                print(f"LLM score: {llm_score:.3f}")
            except (ValidationError, Exception) as e:
                error_type = classify_error(e)
                errors.append(f"llm_grade_error[{error_type}]: {e}")
                print(f"LLM grade error ({error_type}): {e}")

            try:
                # print("Running answer relevance...")
                grade = grade_answer_relevance(client, record.query, pred_response)
                answer_relevance_score = grade.score
                answer_relevance_rationale = grade.rationale
                print(f"Answer relevance: {answer_relevance_score:.3f}")
            except (ValidationError, Exception) as e:
                error_type = classify_error(e)
                errors.append(f"answer_relevance_error[{error_type}]: {e}")
                print(f"Answer relevance error ({error_type}): {e}")

            try:
                # print("Running context relevance...")
                grade = grade_context_relevance(client, record.query, record.response)
                context_relevance_score = grade.score
                context_relevance_rationale = grade.rationale
                print(f"Context relevance: {context_relevance_score:.3f}")
            except (ValidationError, Exception) as e:
                error_type = classify_error(e)
                errors.append(f"context_relevance_error[{error_type}]: {e}")
                print(f"Context relevance error ({error_type}): {e}")

            try:
                # print("Running faithfulness...")
                claims = extract_claims(client, pred_response)
                support = check_claim_support(client, claims, record.response)
                claim_count = len(claims)
                supported_claims_count = len(support.supported)
                unsupported_claims_count = len(support.unsupported)
                not_enough_info_claims_count = len(support.not_enough_info)
                if claim_count > 0:
                    faithfulness_score = supported_claims_count / claim_count
                    faithfulness_rationale = (
                        f"Supported {supported_claims_count}/{claim_count} claims; "
                        f"unsupported: {unsupported_claims_count}; "
                        f"not enough info: {not_enough_info_claims_count}."
                    )
                else:
                    faithfulness_score = 1.0
                    faithfulness_rationale = "No claims extracted from answer."
                unsupported_claims = " | ".join(support.unsupported)
                print(f"Faithfulness: {faithfulness_score:.3f}")
            except (ValidationError, Exception) as e:
                error_type = classify_error(e)
                errors.append(f"faithfulness_error[{error_type}]: {e}")
                print(f"Faithfulness error ({error_type}): {e}")

        error_value = "no_proc_error" if not errors else "; ".join(errors)

        results.append(
            ResultRow(
                number=record.number,
                standard=record.standard,
                question_level=record.question_level,
                query=record.query,
                gold_response=record.response,
                pred_response=pred_response,
                similarity_score=similarity_score,
                llm_score=llm_score,
                llm_rationale=llm_rationale,
                answer_relevance_score=answer_relevance_score,
                answer_relevance_rationale=answer_relevance_rationale,
                context_relevance_score=context_relevance_score,
                context_relevance_rationale=context_relevance_rationale,
                faithfulness_score=faithfulness_score,
                faithfulness_rationale=faithfulness_rationale,
                claim_count=claim_count,
                supported_claims_count=supported_claims_count,
                unsupported_claims_count=unsupported_claims_count,
                not_enough_info_claims_count=not_enough_info_claims_count,
                unsupported_claims=unsupported_claims,
                error=error_value,
            )
        )

    save_results(results, RESULTS_CSV)
    save_unsupported_claims(results, UNSUPPORTED_CLAIMS_CSV)
    print(f"\nSaved results to {RESULTS_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:")
        print(str(e))
        raise
