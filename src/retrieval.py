import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from src.index_metadata import INDEX_VERSION, get_index_metadata_path
from src.reranking import LLMReranker
from src.retrieval_types import (
    BranchChunkRetrievalResult,
    FusedChunkRetrievalResult,
    PageRetrievalResult,
    PublicRetrievalResult,
)

_log = logging.getLogger(__name__)


def _with_compat_distance(result: Dict) -> PublicRetrievalResult:
    result_with_alias = result.copy()
    if "retrieval_score" in result_with_alias:
        result_with_alias["distance"] = result_with_alias["retrieval_score"]
    return cast(PublicRetrievalResult, result_with_alias)


def _with_compat_distance_list(results: List[Dict]) -> List[PublicRetrievalResult]:
    return [_with_compat_distance(result) for result in results]


def _load_document_by_company_name(documents_dir: Path, company_name: str) -> Tuple[Path, Dict]:
    for path in documents_dir.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
            if doc["metainfo"]["company_name"] == company_name:
                return path, doc
    raise ValueError(f"No report found with '{company_name}' company name.")


def _map_chunks_to_parent_pages(
    chunk_results: List[Dict], pages: List[Dict]
) -> List[PageRetrievalResult]:
    page_lookup = {page["page"]: page for page in pages}
    page_results: List[PageRetrievalResult] = []
    seen_pages = set()

    for result in chunk_results:
        page = result["page"]
        page_key = (result["doc_id"], page)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)

        parent_page = page_lookup[page]
        page_results.append(
            {
                "doc_id": result["doc_id"],
                "page": page,
                "text": parent_page["text"],
                "retrieval_score": result["retrieval_score"],
            }
        )

    return page_results


class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir

    def _load_document_and_index(self, company_name: str) -> Tuple[Dict, object]:
        _, document = _load_document_by_company_name(self.documents_dir, company_name)
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        self._warn_on_legacy_or_mismatched_metadata(bm25_path)
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)
        return document, bm25_index

    def _retrieve_chunk_results(
        self, company_name: str, query: str, top_n: int = 3
    ) -> List[BranchChunkRetrievalResult]:
        document, bm25_index = self._load_document_and_index(company_name)
        chunks = document["content"]["chunks"]
        doc_id = document["metainfo"]["sha1_name"]

        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)

        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        retrieval_results: List[BranchChunkRetrievalResult] = []
        for chunk_idx in top_indices:
            raw_score = round(float(scores[chunk_idx]), 4)
            chunk = chunks[chunk_idx]
            retrieval_results.append(
                {
                    "doc_id": doc_id,
                    "chunk_uid": f"{doc_id}::{chunk_idx}",
                    "chunk_idx": chunk_idx,
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "raw_score": raw_score,
                    "retrieval_score": raw_score,
                    "retriever": "bm25",
                }
            )

        return retrieval_results

    def retrieve_by_company_name(
        self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False
    ) -> List[PublicRetrievalResult]:
        chunk_results = self._retrieve_chunk_results(company_name=company_name, query=query, top_n=top_n)
        if not return_parent_pages:
            return _with_compat_distance_list(chunk_results)

        _, document = _load_document_by_company_name(self.documents_dir, company_name)
        page_results = _map_chunks_to_parent_pages(chunk_results, document["content"]["pages"])
        return _with_compat_distance_list(page_results[:top_n])

    def _warn_on_legacy_or_mismatched_metadata(self, index_path: Path):
        metadata_path = get_index_metadata_path(index_path)
        if not metadata_path.exists():
            _log.warning(
                "BM25 metadata file is missing for %s; treating it as a legacy index.",
                index_path.name,
            )
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as err:
            _log.warning("Could not read BM25 metadata for %s: %s", index_path.name, err)
            return

        if metadata.get("index_version") != INDEX_VERSION:
            _log.warning(
                "BM25 index %s has version %s, expected %s.",
                index_path.name,
                metadata.get("index_version"),
                INDEX_VERSION,
            )


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=None, max_retries=2)
        return llm

    @staticmethod
    def set_up_llm():
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=None, max_retries=2)
        return llm

    def _load_dbs(self):
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob("*.json"))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob("*.faiss")}

        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, "r", encoding="utf-8") as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue

            self._warn_on_legacy_or_mismatched_metadata(vector_db_files[stem])
            all_dbs.append(
                {
                    "name": stem,
                    "vector_db": vector_db,
                    "document": document,
                }
            )
        return all_dbs

    def _warn_on_legacy_or_mismatched_metadata(self, index_path: Path):
        metadata_path = get_index_metadata_path(index_path)
        if not metadata_path.exists():
            _log.warning(
                "Vector index metadata file is missing for %s; treating it as a legacy index.",
                index_path.name,
            )
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as err:
            _log.warning("Could not read vector index metadata for %s: %s", index_path.name, err)
            return

        if metadata.get("index_version") != INDEX_VERSION:
            _log.warning(
                "Vector index %s has version %s, expected %s.",
                index_path.name,
                metadata.get("index_version"),
                INDEX_VERSION,
            )

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def _get_target_report(self, company_name: str) -> Dict:
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                return report
        raise ValueError(f"No report found with '{company_name}' company name.")

    def _retrieve_chunk_results(
        self, company_name: str, query: str, top_n: int = 3
    ) -> List[BranchChunkRetrievalResult]:
        target_report = self._get_target_report(company_name)
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        doc_id = document["metainfo"]["sha1_name"]

        actual_top_n = min(top_n, len(chunks))
        embedding = self.llm.embeddings.create(input=query, model="text-embedding-3-large")
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)

        retrieval_results: List[BranchChunkRetrievalResult] = []
        for raw_score, chunk_idx in zip(distances[0], indices[0]):
            retrieval_score = round(float(raw_score), 4)
            chunk = chunks[int(chunk_idx)]
            retrieval_results.append(
                {
                    "doc_id": doc_id,
                    "chunk_uid": f"{doc_id}::{int(chunk_idx)}",
                    "chunk_idx": int(chunk_idx),
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "raw_score": retrieval_score,
                    "retrieval_score": retrieval_score,
                    "retriever": "vector",
                }
            )

        return retrieval_results

    def retrieve_by_company_name(
        self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False
    ) -> List[PublicRetrievalResult]:
        chunk_results = self._retrieve_chunk_results(company_name=company_name, query=query, top_n=top_n)
        if not return_parent_pages:
            return _with_compat_distance_list(chunk_results)

        target_report = self._get_target_report(company_name)
        page_results = _map_chunks_to_parent_pages(chunk_results, target_report["document"]["content"]["pages"])
        return _with_compat_distance_list(page_results[:top_n])

    def retrieve_all(self, company_name: str) -> List[PublicRetrievalResult]:
        target_report = self._get_target_report(company_name)
        pages = target_report["document"]["content"]["pages"]
        doc_id = target_report["document"]["metainfo"]["sha1_name"]

        all_pages: List[PageRetrievalResult] = []
        for page in sorted(pages, key=lambda p: p["page"]):
            all_pages.append(
                {
                    "doc_id": doc_id,
                    "page": page["page"],
                    "text": page["text"],
                    "retrieval_score": 0.5,
                }
            )
        return _with_compat_distance_list(all_pages)


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, bm25_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.bm25_retriever = BM25Retriever(bm25_db_dir, documents_dir)
        self.documents_dir = documents_dir
        self.reranker = LLMReranker()

    @staticmethod
    def _fuse_with_rrf(
        vector_results: List[BranchChunkRetrievalResult],
        bm25_results: List[BranchChunkRetrievalResult],
        rrf_k: int,
    ) -> List[FusedChunkRetrievalResult]:
        fused_by_uid: Dict[str, FusedChunkRetrievalResult] = {}

        def merge_branch(results: List[BranchChunkRetrievalResult], branch_name: str):
            for rank, result in enumerate(results, start=1):
                chunk_uid = result["chunk_uid"]
                rrf_score = 1 / (rrf_k + rank)
                if chunk_uid not in fused_by_uid:
                    fused_by_uid[chunk_uid] = {
                        "doc_id": result["doc_id"],
                        "chunk_uid": chunk_uid,
                        "chunk_idx": result["chunk_idx"],
                        "page": result["page"],
                        "text": result["text"],
                        "retrieval_score": 0.0,
                        "retrievers": [],
                    }

                fused = fused_by_uid[chunk_uid]
                fused["retrieval_score"] = round(fused["retrieval_score"] + rrf_score, 6)
                if branch_name not in fused["retrievers"]:
                    fused["retrievers"].append(branch_name)
                fused[f"{branch_name}_raw_score"] = result["raw_score"]

        merge_branch(vector_results, "vector")
        merge_branch(bm25_results, "bm25")

        fused_results = list(fused_by_uid.values())
        fused_results.sort(key=lambda item: item["retrieval_score"], reverse=True)
        return fused_results

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        dense_top_k: int = 12,
        bm25_top_k: int = 12,
        fusion_top_k: int = 20,
        rrf_k: int = 60,
        top_n: int = 6,
        return_parent_pages: bool = False,
        apply_llm_reranking: bool = False,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        llm_weight: float = 0.7,
    ) -> List[PublicRetrievalResult]:
        vector_results = self.vector_retriever._retrieve_chunk_results(
            company_name=company_name,
            query=query,
            top_n=dense_top_k,
        )
        bm25_results = self.bm25_retriever._retrieve_chunk_results(
            company_name=company_name,
            query=query,
            top_n=bm25_top_k,
        )

        fused_results = self._fuse_with_rrf(vector_results, bm25_results, rrf_k=rrf_k)
        fused_results = fused_results[:fusion_top_k]

        if return_parent_pages:
            _, document = _load_document_by_company_name(self.documents_dir, company_name)
            retrieval_results = _map_chunks_to_parent_pages(fused_results, document["content"]["pages"])
        else:
            retrieval_results = fused_results

        if apply_llm_reranking and retrieval_results:
            rerank_candidates = retrieval_results[:llm_reranking_sample_size]
            retrieval_results = self.reranker.rerank_documents(
                query=query,
                documents=rerank_candidates,
                documents_batch_size=documents_batch_size,
                llm_weight=llm_weight,
            )

        return _with_compat_distance_list(retrieval_results[:top_n])
