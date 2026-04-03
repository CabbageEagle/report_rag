import json
import re
import tiktoken
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

import src.prompts as prompts
from src.api_requests import BaseOpenaiProcessor
from src.index_metadata import build_chunk_index_metadata


class TextSplitter:
    def __init__(
        self,
        contextualizer: Optional[Callable[[Dict[str, object]], str]] = None,
        llm_processor: Optional[BaseOpenaiProcessor] = None,
        model: str = "gpt-4o-mini-2024-07-18",
    ):
        self._custom_contextualizer = contextualizer
        self._llm_processor = llm_processor
        self._model = model

    def _get_llm_processor(self) -> BaseOpenaiProcessor:
        if self._llm_processor is None:
            self._llm_processor = BaseOpenaiProcessor()
        return self._llm_processor

    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number."""
        tables_by_page = {}
        for table in tables:
            if "serialized" not in table:
                continue

            page = table["page"]
            if page not in tables_by_page:
                tables_by_page[page] = []

            table_text = "\n".join(
                block["information_block"]
                for block in table["serialized"]["information_blocks"]
            )

            primary_metrics = table["serialized"].get("subject_core_entities_list", [])
            table_markdown = table.get("markdown", "")

            tables_by_page[page].append(
                {
                    "page": page,
                    "text": table_text,
                    "table_id": table["table_id"],
                    "length_tokens": self.count_tokens(table_text),
                    "table_topic": self._extract_table_topic(table_markdown, table_text),
                    "year_range": self._extract_year_range(table_markdown or table_text),
                    "unit": self._extract_unit(table_markdown or table_text),
                    "primary_metrics": primary_metrics[:8],
                }
            )

        return tables_by_page

    def _split_report(
        self,
        file_content: Dict[str, any],
        serialized_tables_report_path: Optional[Path] = None,
    ) -> Dict[str, any]:
        """Split report into contextualized chunks."""
        chunks = []
        chunk_id = 0

        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, "r", encoding="utf-8") as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get("tables", []))

        report_year = self._extract_report_year(file_content)
        company_name = file_content.get("metainfo", {}).get("company_name", "")

        for page in file_content["content"]["pages"]:
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk["id"] = chunk_id
                chunk["type"] = "content"
                chunk_id += 1
                chunks.append(chunk)

            if tables_by_page and page["page"] in tables_by_page:
                for table in tables_by_page[page["page"]]:
                    table["id"] = chunk_id
                    table["type"] = "serialized_table"
                    chunk_id += 1
                    chunks.append(table)

        contextualized_chunks = []
        page_lookup = {page["page"]: page for page in file_content["content"]["pages"]}
        for chunk in chunks:
            page = page_lookup[chunk["page"]]
            nearest_heading = self._extract_nearest_heading(
                page["text"], chunk.get("_start_char")
            )

            prompt_context = {
                "company_name": company_name,
                "report_year": report_year,
                "page": chunk["page"],
                "chunk_type": chunk["type"],
                "nearest_heading": nearest_heading,
                "page_text": page["text"],
                "chunk_text": chunk["text"],
                "table_topic": chunk.get("table_topic", ""),
                "year_range": chunk.get("year_range", ""),
                "unit": chunk.get("unit", ""),
                "primary_metrics": ", ".join(chunk.get("primary_metrics", [])),
            }
            contextual_description = self._generate_contextual_description(prompt_context)

            contextualized_chunk = {
                key: value for key, value in chunk.items() if key != "_start_char"
            }
            contextualized_chunk["nearest_heading"] = nearest_heading
            contextualized_chunk["contextual_description"] = contextual_description
            contextualized_chunk["contextualized_text"] = (
                f"{contextual_description}\n\n{contextualized_chunk['text']}"
            )
            contextualized_chunks.append(contextualized_chunk)

        file_content["content"]["chunks"] = contextualized_chunks
        file_content["content"]["index_metadata"] = build_chunk_index_metadata()
        return file_content

    def count_tokens(self, string: str, encoding_name: str = "o200k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        return len(tokens)

    def _split_page(
        self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50
    ) -> List[Dict[str, any]]:
        """Split page text into chunks. The original text includes markdown tables."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(page["text"])
        chunks_with_meta = []
        search_start = 0

        for chunk in chunks:
            start_char = page["text"].find(chunk, search_start)
            if start_char == -1:
                start_char = page["text"].find(chunk)
            if start_char == -1:
                start_char = 0

            search_start = max(start_char + 1, start_char + (len(chunk) // 2))
            chunks_with_meta.append(
                {
                    "page": page["page"],
                    "length_tokens": self.count_tokens(chunk),
                    "text": chunk,
                    "_start_char": start_char,
                }
            )
        return chunks_with_meta

    def _generate_contextual_description(self, prompt_context: Dict[str, object]) -> str:
        if self._custom_contextualizer is not None:
            description = self._custom_contextualizer(prompt_context)
        else:
            answer_dict = self._get_llm_processor().send_message(
                model=self._model,
                temperature=0,
                system_content=prompts.ContextualizedChunkPrompt.system_prompt_with_schema,
                human_content=prompts.ContextualizedChunkPrompt.user_prompt.format(**prompt_context),
                is_structured=True,
                response_format=prompts.ContextualizedChunkPrompt.AnswerSchema,
            )
            description = answer_dict["contextual_description"]

        description = str(description).strip()
        if not description:
            raise ValueError("Contextual description cannot be empty.")
        return description

    def _extract_report_year(self, file_content: Dict[str, any]) -> str:
        metainfo = file_content.get("metainfo", {})
        for key in ("report_year", "year", "fiscal_year"):
            value = metainfo.get(key)
            if value:
                return str(value)

        page_texts = "\n".join(
            page.get("text", "") for page in file_content.get("content", {}).get("pages", [])[:3]
        )
        patterns = [
            r"annual report[^0-9]{0,40}((?:19|20)\d{2})",
            r"for the fiscal year ended[^0-9]{0,40}((?:19|20)\d{2})",
            r"for the year ended[^0-9]{0,40}((?:19|20)\d{2})",
        ]
        for pattern in patterns:
            match = re.search(pattern, page_texts, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)

        all_years = re.findall(r"\b(?:19|20)\d{2}\b", page_texts)
        return all_years[0] if all_years else ""

    def _extract_nearest_heading(self, page_text: str, chunk_start: Optional[int]) -> str:
        headings = []
        position = 0
        for line in page_text.splitlines():
            stripped = line.strip()
            if self._looks_like_heading(stripped):
                headings.append((position, stripped.lstrip("#").strip()))
            position += len(line) + 1

        if not headings:
            return ""

        if chunk_start is None:
            return headings[-1][1]

        nearest = ""
        for heading_pos, heading_text in headings:
            if heading_pos <= chunk_start:
                nearest = heading_text
            else:
                break
        return nearest

    def _looks_like_heading(self, line: str) -> bool:
        if not line or len(line) > 120:
            return False
        if line.startswith("#"):
            return True
        if line.isupper() and len(line.split()) <= 12:
            return True
        return False

    def _extract_table_topic(self, table_markdown: str, table_text: str) -> str:
        for source in (table_markdown, table_text):
            for line in source.splitlines():
                stripped = line.strip().strip("|")
                if stripped and not stripped.startswith("---"):
                    return stripped[:160]
        return ""

    def _extract_year_range(self, text: str) -> str:
        years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text)))
        if not years:
            return ""
        if len(years) == 1:
            return years[0]
        return f"{years[0]}-{years[-1]}"

    def _extract_unit(self, text: str) -> str:
        patterns = [
            r"\bin (thousands|millions|billions)\b",
            r"\bUSD\b|\bEUR\b|\bGBP\b|\bCHF\b|\bJPY\b|\bCNY\b",
            r"\$ ?in (thousands|millions|billions)",
            r"%",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return ""

    def split_all_reports(
        self,
        all_report_dir: Path,
        output_dir: Path,
        serialized_tables_dir: Optional[Path] = None,
    ):
        all_report_paths = list(all_report_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_reports = []
        processed_reports = 0

        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")

            try:
                with open(report_path, "r", encoding="utf-8") as file:
                    report_data = json.load(file)

                updated_report = self._split_report(report_data, serialized_tables_path)
                with open(output_dir / report_path.name, "w", encoding="utf-8") as file:
                    json.dump(updated_report, file, indent=2, ensure_ascii=False)
                processed_reports += 1
            except Exception as err:
                failed_reports.append(report_path.name)
                stale_output = output_dir / report_path.name
                if stale_output.exists():
                    stale_output.unlink()
                print(f"Failed to split {report_path.name}: {err}")

        print(f"Split {processed_reports} files")
        if failed_reports:
            print(f"Failed reports ({len(failed_reports)}): {', '.join(failed_reports)}")
