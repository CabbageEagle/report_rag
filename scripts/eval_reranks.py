from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

# 你项目里的类。下面 import 路径按你当前仓库调整
from src.retrieval import HybridRetriever
from src.reranking import LLMReranker

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# -------------------------
# 数据结构
# -------------------------

@dataclass
class EvalSample:
    qid: str
    company: str
    question: str
    gold_pages: List[int]
    question_type: str | None = None


@dataclass
class CandidatePage:
    doc_id: str
    page: int
    text: str
    retrieval_score: float


@dataclass
class RankedPage:
    doc_id: str
    page: int
    text: str
    retrieval_score: float
    rerank_score: float
    rank: int


# -------------------------
# 数据加载
# -------------------------

def load_eval_samples(path: str | Path) -> List[EvalSample]:
    samples: List[EvalSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            samples.append(
                EvalSample(
                    qid=row["id"],
                    company=row["company"],
                    question=row["question"],
                    gold_pages=row["gold_pages"],
                    question_type=row.get("question_type"),
                )
            )
    return samples


# -------------------------
# 第一阶段候选池
# 固定：dual recall + chunk RRF + page mapping(top 20) + 不做 rerank
# -------------------------

class CandidatePoolBuilder:
    def __init__(
        self,
        vector_db_dir: str | Path,
        bm25_db_dir: str | Path,
        documents_dir: str | Path,
        dense_top_k: int = 40,
        bm25_top_k: int = 40,
        hybrid_fusion_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.retriever = HybridRetriever(
            vector_db_dir=vector_db_dir,
            bm25_db_dir=bm25_db_dir,
            documents_dir=documents_dir,
        )
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.hybrid_fusion_top_k = hybrid_fusion_top_k
        self.rrf_k = rrf_k

    def get_parent_page_candidates(self, sample: EvalSample) -> List[CandidatePage]:
        """
        这里要求你当前的 HybridRetriever 支持：
        - use_reranker=False
        - return_parent_pages=True
        - dense_top_k / bm25_top_k / hybrid_fusion_top_k / rrf_k
        如果你参数名略有不同，改这里就行。
        """
        results = self.retriever.retrieve_by_company_name(
            company_name=sample.company,
            query=sample.question,
            return_parent_pages=True,
            use_reranker=False,
            dense_top_k=self.dense_top_k,
            bm25_top_k=self.bm25_top_k,
            hybrid_fusion_top_k=self.hybrid_fusion_top_k,
            rrf_k=self.rrf_k,
            top_n=self.hybrid_fusion_top_k,
        )

        pages: List[CandidatePage] = []
        for r in results:
            pages.append(
                CandidatePage(
                    doc_id=r.get("doc_id", ""),
                    page=int(r["page"]),
                    text=r["text"],
                    retrieval_score=float(r.get("retrieval_score", r.get("distance", 0.0))),
                )
            )
        return pages


# -------------------------
# Reranker 接口
# -------------------------

class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[CandidatePage]) -> List[RankedPage]:
        ...


class IdentityReranker:
    """不做 rerank，直接按 retrieval_score 排。"""

    def rerank(self, query: str, candidates: List[CandidatePage]) -> List[RankedPage]:
        ranked = sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
        return [
            RankedPage(
                doc_id=c.doc_id,
                page=c.page,
                text=c.text,
                retrieval_score=c.retrieval_score,
                rerank_score=c.retrieval_score,
                rank=i + 1,
            )
            for i, c in enumerate(ranked)
        ]


class LLMRerankerAdapter:
    """
    适配你当前项目里的 LLMReranker。
    如果你的类签名和下面不同，只改这个 adapter。
    """

    def __init__(self) -> None:
        self.model = LLMReranker()

    def rerank(self, query: str, candidates: List[CandidatePage]) -> List[RankedPage]:
        docs = [
            {
                "doc_id": c.doc_id,
                "page": c.page,
                "text": c.text,
                "retrieval_score": c.retrieval_score,
            }
            for c in candidates
        ]

        # 这里按你改造后的 reranking.py 对齐
        reranked_docs = self.model.rerank_documents(query=query, documents=docs)

        ranked: List[RankedPage] = []
        for i, d in enumerate(reranked_docs):
            ranked.append(
                RankedPage(
                    doc_id=d.get("doc_id", ""),
                    page=int(d["page"]),
                    text=d["text"],
                    retrieval_score=float(d.get("retrieval_score", 0.0)),
                    rerank_score=float(d.get("combined_score", d.get("rerank_score", d.get("retrieval_score", 0.0)))),
                    rank=i + 1,
                )
            )
        return ranked


class CrossEncoderReranker:
    """
    baseline reranker 和 fine-tuned reranker 都用这个类，
    只改 model_name_or_path。
    """

    def __init__(self, model_name_or_path: str, max_length: int = 1024) -> None:
        if CrossEncoder is None:
            raise ImportError("Please install sentence-transformers")
        self.model = CrossEncoder(model_name_or_path, max_length=max_length)

    def rerank(self, query: str, candidates: List[CandidatePage]) -> List[RankedPage]:
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = []
        for c, s in zip(candidates, scores):
            ranked.append(
                RankedPage(
                    doc_id=c.doc_id,
                    page=c.page,
                    text=c.text,
                    retrieval_score=c.retrieval_score,
                    rerank_score=float(s),
                    rank=0,
                )
            )

        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked


# -------------------------
# 指标
# -------------------------

def hit_at_k(ranked_pages: List[int], gold_pages: set[int], k: int) -> float:
    return 1.0 if any(p in gold_pages for p in ranked_pages[:k]) else 0.0


def first_hit_rank(ranked_pages: List[int], gold_pages: set[int]) -> int | None:
    for i, p in enumerate(ranked_pages, start=1):
        if p in gold_pages:
            return i
    return None


def mrr_at_k(ranked_pages: List[int], gold_pages: set[int], k: int) -> float:
    r = first_hit_rank(ranked_pages[:k], gold_pages)
    return 0.0 if r is None else 1.0 / r


def dcg_at_k(ranked_pages: List[int], gold_pages: set[int], k: int) -> float:
    score = 0.0
    for i, p in enumerate(ranked_pages[:k], start=1):
        rel = 1.0 if p in gold_pages else 0.0
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(ranked_pages: List[int], gold_pages: set[int], k: int) -> float:
    ideal_hits = min(len(gold_pages), k)
    if ideal_hits == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    actual = dcg_at_k(ranked_pages, gold_pages, k)
    return actual / ideal if ideal > 0 else 0.0


# -------------------------
# 单模型评测
# -------------------------

def evaluate_reranker(
    name: str,
    reranker: Reranker,
    samples: List[EvalSample],
    pool_builder: CandidatePoolBuilder,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = out_dir / f"{name}.per_query.jsonl"

    hit1, hit3, hit5 = [], [], []
    mrr10, ndcg10 = [], []
    latencies = []

    with per_query_path.open("w", encoding="utf-8") as wf:
        for sample in samples:
            candidates = pool_builder.get_parent_page_candidates(sample)

            t0 = time.perf_counter()
            ranked = reranker.rerank(sample.question, candidates)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            ranked_pages = [r.page for r in ranked]
            gold_pages = set(sample.gold_pages)

            row = {
                "id": sample.qid,
                "company": sample.company,
                "question": sample.question,
                "question_type": sample.question_type,
                "gold_pages": sample.gold_pages,
                "candidate_pages": [c.page for c in candidates],
                "ranked_pages": ranked_pages,
                "first_hit_rank": first_hit_rank(ranked_pages, gold_pages),
                "hit@1": hit_at_k(ranked_pages, gold_pages, 1),
                "hit@3": hit_at_k(ranked_pages, gold_pages, 3),
                "hit@5": hit_at_k(ranked_pages, gold_pages, 5),
                "mrr@10": mrr_at_k(ranked_pages, gold_pages, 10),
                "ndcg@10": ndcg_at_k(ranked_pages, gold_pages, 10),
                "latency_ms": latency_ms,
            }
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

            hit1.append(row["hit@1"])
            hit3.append(row["hit@3"])
            hit5.append(row["hit@5"])
            mrr10.append(row["mrr@10"])
            ndcg10.append(row["ndcg@10"])

    summary = {
        "system": name,
        "num_queries": len(samples),
        "hit@1": statistics.mean(hit1) if hit1 else 0.0,
        "hit@3": statistics.mean(hit3) if hit3 else 0.0,
        "hit@5": statistics.mean(hit5) if hit5 else 0.0,
        "mrr@10": statistics.mean(mrr10) if mrr10 else 0.0,
        "ndcg@10": statistics.mean(ndcg10) if ndcg10 else 0.0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
    }

    with (out_dir / f"{name}.summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = int(math.ceil((p / 100.0) * len(xs))) - 1
    idx = min(max(idx, 0), len(xs) - 1)
    return xs[idx]


# -------------------------
# 汇总
# -------------------------

def write_global_summary(summaries: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--vector_db_dir", required=True)
    parser.add_argument("--bm25_db_dir", required=True)
    parser.add_argument("--documents_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--dense_top_k", type=int, default=40)
    parser.add_argument("--bm25_top_k", type=int, default=40)
    parser.add_argument("--hybrid_fusion_top_k", type=int, default=20)
    parser.add_argument("--rrf_k", type=int, default=60)

    parser.add_argument("--baseline_ce_model", required=True)
    parser.add_argument("--finetuned_ce_model", required=True)

    args = parser.parse_args()

    samples = load_eval_samples(args.eval_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_builder = CandidatePoolBuilder(
        vector_db_dir=args.vector_db_dir,
        bm25_db_dir=args.bm25_db_dir,
        documents_dir=args.documents_dir,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        hybrid_fusion_top_k=args.hybrid_fusion_top_k,
        rrf_k=args.rrf_k,
    )

    systems: List[tuple[str, Reranker]] = [
        ("llm_rerank", LLMRerankerAdapter()),
        ("baseline_ce", CrossEncoderReranker(args.baseline_ce_model)),
        ("finetuned_ce", CrossEncoderReranker(args.finetuned_ce_model)),
    ]

    summaries = []
    for name, reranker in systems:
        summary = evaluate_reranker(
            name=name,
            reranker=reranker,
            samples=samples,
            pool_builder=pool_builder,
            out_dir=out_dir,
        )
        summaries.append(summary)

    write_global_summary(summaries, out_dir / "all_systems.summary.json")


if __name__ == "__main__":
    main()