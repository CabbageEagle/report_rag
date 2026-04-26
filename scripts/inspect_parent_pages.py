from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reranking import JinaReranker
from src.retrieval import BM25Retriever, HybridRetriever, VectorRetriever


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_file_exists(path: Path, flag_name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{flag_name} not found: '{path}'. "
            f"Pass an explicit {flag_name} path or adjust --dataset-root."
        )
    return path


def resolve_question(questions_path: Path, question: str | None, question_index: int | None) -> dict[str, Any]:
    if question is not None:
        return {"text": question}

    if question_index is None:
        raise ValueError("Provide either --question or --question-index.")

    ensure_file_exists(questions_path, "--questions-path")
    questions = load_json(questions_path)
    if question_index < 0 or question_index >= len(questions):
        raise IndexError(f"question_index {question_index} is out of range 0..{len(questions) - 1}")
    return questions[question_index]


def infer_company(subset_path: Path, question_text: str) -> str:
    with subset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        company_names = sorted(
            {row["company_name"] for row in reader if row.get("company_name")},
            key=len,
            reverse=True,
        )

    companies: list[str] = []
    for company in company_names:
        escaped_company = re.escape(company)
        pattern = rf"{escaped_company}(?:\W|$)"
        if re.search(pattern, question_text, re.IGNORECASE):
            companies.append(company)
            question_text = re.sub(pattern, "", question_text, flags=re.IGNORECASE)

    if not companies:
        raise ValueError("Could not infer company from question. Pass --company explicitly.")
    if len(companies) > 1:
        raise ValueError(
            f"Question matches multiple companies {companies}. Pass --company explicitly."
        )
    return companies[0]


def build_retriever(dataset_root: Path, use_hybrid: bool):
    vector_db_dir = dataset_root / "databases" / "vector_dbs"
    documents_dir = dataset_root / "databases" / "chunked_reports"

    if use_hybrid:
        return HybridRetriever(
            vector_db_dir=vector_db_dir,
            bm25_db_dir=dataset_root / "databases" / "bm25_dbs",
            documents_dir=documents_dir,
        )

    return VectorRetriever(
        vector_db_dir=vector_db_dir,
        documents_dir=documents_dir,
    )


def build_bm25_reranker(dataset_root: Path) -> tuple[BM25Retriever, JinaReranker]:
    documents_dir = dataset_root / "databases" / "chunked_reports"
    bm25_db_dir = dataset_root / "databases" / "bm25_dbs"
    return BM25Retriever(bm25_db_dir=bm25_db_dir, documents_dir=documents_dir), JinaReranker()


def retrieve_parent_pages(
    dataset_root: Path,
    company: str,
    question_text: str,
    top_n: int,
    use_hybrid: bool,
    dense_top_k: int,
    bm25_top_k: int,
    fusion_top_k: int,
    rrf_k: int,
    llm_reranking_sample_size: int,
    documents_batch_size: int,
    llm_weight: float,
) -> list[dict[str, Any]]:
    if use_hybrid:
        retriever = build_retriever(dataset_root, use_hybrid)

        return retriever.retrieve_by_company_name(
            company_name=company,
            query=question_text,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            fusion_top_k=fusion_top_k,
            rrf_k=rrf_k,
            top_n=top_n,
            return_parent_pages=True,
        )

    bm25_retriever, reranker = build_bm25_reranker(dataset_root)
    candidate_top_n = max(top_n, bm25_top_k, llm_reranking_sample_size)
    bm25_results = bm25_retriever.retrieve_by_company_name(
        company_name=company,
        query=question_text,
        top_n=candidate_top_n,
        return_parent_pages=True,
    )
    if not bm25_results:
        return bm25_results

    rerank_candidates = bm25_results[:llm_reranking_sample_size]
    jina_response = reranker.rerank(
        query=question_text,
        documents=[candidate["text"] for candidate in rerank_candidates],
        top_n=len(rerank_candidates),
    )

    reranked_results: list[dict[str, Any]] = []
    for item in jina_response.get("results", []):
        idx = item.get("index")
        if idx is None or idx >= len(rerank_candidates):
            continue
        candidate = rerank_candidates[idx].copy()
        relevance_score = float(item.get("relevance_score", 0.0))
        candidate["relevance_score"] = relevance_score
        candidate["combined_score"] = round(
            llm_weight * relevance_score + (1 - llm_weight) * candidate["retrieval_score"],
            4,
        )
        reranked_results.append(candidate)

    reranked_results.sort(key=lambda result: result["combined_score"], reverse=True)
    return reranked_results[:top_n]


def print_results(company: str, question_text: str, results: list[dict[str, Any]], preview_chars: int):
    print(f"Company: {company}")
    print(f"Question: {question_text}")
    print(f"Parent pages: {len(results)}")
    print()

    for idx, result in enumerate(results, start=1):
        page = result.get("page")
        retrieval_score = result.get("retrieval_score")
        relevance_score = result.get("relevance_score")
        combined_score = result.get("combined_score")
        text = (result.get("text") or "").strip()
        preview = text[:preview_chars]
        if len(text) > preview_chars:
            preview += "..."

        print(
            f"[{idx}] page={page} "
            f"retrieval_score={retrieval_score} "
            f"relevance_score={relevance_score} "
            f"combined_score={combined_score}"
        )
        print(preview)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect retrieved parent pages for a company question."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/test_set",
        help="Dataset root containing databases/, questions.json, and subset.csv",
    )
    parser.add_argument(
        "--questions-path",
        help="Optional explicit path to questions.json",
    )
    parser.add_argument(
        "--subset-path",
        help="Optional explicit path to subset.csv",
    )
    parser.add_argument(
        "--question",
        help="Question text to inspect",
    )
    parser.add_argument(
        "--question-index",
        type=int,
        help="Question index in questions.json",
    )
    parser.add_argument(
        "--company",
        help="Company name. If omitted, try to infer from subset.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of parent pages to retrieve",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use the old hybrid vector+bm25 path instead of BM25+rerank",
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=12,
        help="Dense retrieval candidate count for hybrid retrieval",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=12,
        help="BM25 candidate count for hybrid retrieval",
    )
    parser.add_argument(
        "--fusion-top-k",
        type=int,
        default=20,
        help="Fused candidate count for hybrid retrieval",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k for hybrid retrieval",
    )
    parser.add_argument(
        "--llm-reranking-sample-size",
        type=int,
        default=30,
        help="Number of BM25 parent pages to send into reranking",
    )
    parser.add_argument(
        "--documents-batch-size",
        type=int,
        default=2,
        help="Unused in BM25+rerank mode; kept for hybrid parity",
    )
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.7,
        help="Weight for reranking score when combining with retrieval score",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=800,
        help="Characters of each page to print",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    questions_path = Path(args.questions_path) if args.questions_path else dataset_root / "questions.json"
    subset_path = Path(args.subset_path) if args.subset_path else dataset_root / "subset.csv"

    question_obj = resolve_question(
        questions_path=questions_path,
        question=args.question,
        question_index=args.question_index,
    )
    question_text = question_obj["text"]
    ensure_file_exists(subset_path, "--subset-path")
    company = args.company or infer_company(subset_path, question_text)

    results = retrieve_parent_pages(
        dataset_root=dataset_root,
        company=company,
        question_text=question_text,
        top_n=args.top_n,
        use_hybrid=args.hybrid,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        fusion_top_k=args.fusion_top_k,
        rrf_k=args.rrf_k,
        llm_reranking_sample_size=args.llm_reranking_sample_size,
        documents_batch_size=args.documents_batch_size,
        llm_weight=args.llm_weight,
    )

    payload = {
        "dataset_root": str(dataset_root),
        "company": company,
        "question": question_text,
        "top_n": args.top_n,
        "retriever": "hybrid" if args.hybrid else "bm25+rerank",
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    print_results(
        company=company,
        question_text=question_text,
        results=results,
        preview_chars=args.preview_chars,
    )

    if args.output:
        print(f"Saved JSON to {args.output}")


if __name__ == "__main__":
    main()
