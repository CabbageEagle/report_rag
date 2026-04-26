from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MULTI_COMPANY_PATTERNS = [
    r"^Which of the companies had ",
]

SINGLE_COMPANY_PATTERNS = [
    r"^For (?P<company>.+?), what ",
    r"^Did (?P<company>.+?) (?:announce|mention|outline|report|detail) ",
    r"^According to the annual report, what is the .+? for (?P<company>.+?) \(",
    r"^What is the total number of employees let go by (?P<company>.+?) according to",
    r"^Which leadership positions changed at (?P<company>.+?) in the reporting period",
    r"^What are the names of new products launched by (?P<company>.+?) as mentioned",
    r"^What is the name of the last product launched by (?P<company>.+?) as mentioned",
    r"^What was the largest single spending of (?P<company>.+?) on executive compensation",
    r"^What was the value of .+? of (?P<company>.+?) at the end",
    r"^For (?P<company>.+?), what is the value of ",
    r"^For (?P<company>.+?), what was the value of ",
]


def is_multi_company_question(question: str) -> bool:
    normalized = question.strip()
    return any(re.match(pattern, normalized) for pattern in MULTI_COMPANY_PATTERNS)


def extract_company(question: str) -> str | None:
    normalized = " ".join(question.split())
    for pattern in SINGLE_COMPANY_PATTERNS:
        match = re.match(pattern, normalized)
        if match:
            return match.group("company").strip()

    fallback_match = re.search(
        r" for (?P<company>.+?) according to the annual report",
        normalized,
        flags=re.IGNORECASE,
    )
    if fallback_match:
        return fallback_match.group("company").strip()

    return None


def normalize_question_text(text: str) -> str:
    return " ".join(text.split()).strip()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def build_answer_index(answers_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    answers = answers_json.get("answers")
    if not isinstance(answers, list):
        raise ValueError("Expected answers JSON to contain a top-level 'answers' list.")

    index: dict[str, dict[str, Any]] = {}
    for answer in answers:
        question = normalize_question_text(answer["question_text"])
        index[question] = answer
    return index


def convert(
    questions_path: str | Path,
    answers_path: str | Path,
    output_path: str | Path,
    skip_no_references: bool = False,
    skip_multi_company: bool = True,
) -> dict[str, Any]:
    questions = load_json(questions_path)
    answers = load_json(answers_path)
    answer_index = build_answer_index(answers)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_multi = 0
    skipped_no_answer = 0
    skipped_no_refs = 0
    skipped_no_company = 0

    with out_path.open("w", encoding="utf-8") as writer:
        for index, question_obj in enumerate(questions):
            question = normalize_question_text(question_obj["text"])
            kind = question_obj.get("kind")

            if skip_multi_company and is_multi_company_question(question):
                skipped_multi += 1
                continue

            answer = answer_index.get(question)
            if answer is None:
                skipped_no_answer += 1
                continue

            company = extract_company(question)
            if company is None:
                skipped_no_company += 1
                continue

            references = answer.get("references", [])
            if skip_no_references and not references:
                skipped_no_refs += 1
                continue

            report_ids = sorted(
                {
                    reference["pdf_sha1"]
                    for reference in references
                    if isinstance(reference, dict) and "pdf_sha1" in reference
                }
            )
            gold_pages = sorted(
                {
                    int(reference["page_index"])
                    for reference in references
                    if isinstance(reference, dict) and "page_index" in reference
                }
            )

            row = {
                "id": f"erc2_{index:05d}",
                "company": company,
                "question": question,
                "question_type": kind,
                "gold_answer": answer.get("value"),
                "gold_pages": gold_pages,
                "report_ids": report_ids,
                "is_silver": True,
                "source": Path(answers_path).name,
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "written": written,
        "skipped_multi_company": skipped_multi,
        "skipped_no_answer": skipped_no_answer,
        "skipped_no_company": skipped_no_company,
        "skipped_no_references": skipped_no_refs,
        "output": str(out_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a silver eval_pages JSONL from ERC2 questions and answer references."
    )
    parser.add_argument(
        "--questions",
        default="data/erc2_set/questions.json",
        help="Path to ERC2 questions.json",
    )
    parser.add_argument(
        "--answers",
        default="data/erc2_set/answers_1st_place_o3-mini.json",
        help="Path to answers JSON with references",
    )
    parser.add_argument(
        "--output",
        default="data/eval/eval_pages.silver.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--skip_no_references",
        action="store_true",
        help="Skip rows whose answer has no references",
    )
    parser.add_argument(
        "--include_multi_company",
        action="store_true",
        help="Include multi-company comparison questions",
    )
    args = parser.parse_args()

    summary = convert(
        questions_path=args.questions,
        answers_path=args.answers,
        output_path=args.output,
        skip_no_references=args.skip_no_references,
        skip_multi_company=not args.include_multi_company,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
