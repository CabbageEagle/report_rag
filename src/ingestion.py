import json
import os
import pickle
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from src.index_metadata import build_index_sidecar_metadata, get_index_metadata_path


def _get_indexable_text(chunk: dict) -> str:
    return chunk.get("contextualized_text") or chunk["text"]


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def _write_sidecar_metadata(self, output_file: Path, report_data: dict):
        metadata = build_index_sidecar_metadata(
            source_report_sha1=report_data["metainfo"]["sha1_name"],
            chunk_count=len(report_data["content"]["chunks"]),
        )
        with open(get_index_metadata_path(output_file), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices."""
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            text_chunks = [_get_indexable_text(chunk) for chunk in report_data["content"]["chunks"]]
            bm25_index = self.create_bm25_index(text_chunks)

            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(bm25_index, f)
            self._write_sidecar_metadata(output_file, report_data)

        print(f"Processed {len(all_report_paths)} reports")


class VectorDBIngestor:
    def __init__(self):
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=None, max_retries=2)
        return llm

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(
        self, text: Union[str, List[str]], model: str = "text-embedding-3-large"
    ) -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        if isinstance(text, list):
            text_chunks = [text[i : i + 1024] for i in range(0, len(text), 1024)]
        else:
            text_chunks = [text]

        embeddings = []
        for chunk in text_chunks:
            response = self.llm.embeddings.create(input=chunk, model=model)
            embeddings.extend([embedding.embedding for embedding in response.data])

        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        return index

    def _process_report(self, report: dict):
        text_chunks = [_get_indexable_text(chunk) for chunk in report["content"]["chunks"]]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def _write_sidecar_metadata(self, output_file: Path, report_data: dict):
        metadata = build_index_sidecar_metadata(
            source_report_sha1=report_data["metainfo"]["sha1_name"],
            chunk_count=len(report_data["content"]["chunks"]),
        )
        with open(get_index_metadata_path(output_file), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))
            self._write_sidecar_metadata(faiss_file_path, report_data)

        print(f"Processed {len(all_report_paths)} reports")
