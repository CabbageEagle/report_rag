from pathlib import Path


CONTEXTUALIZATION_VERSION = "contextualized-chunks-v1"
INDEX_VERSION = "retrieval-index-v1"
CONTEXTUALIZED_TEXT_FORMAT = "contextual_description\\n\\ntext"
CONTEXTUALIZATION_PROMPT_VERSION = "contextualized-chunk-prompt-v1"


def build_chunk_index_metadata() -> dict:
    return {
        "contextualization_version": CONTEXTUALIZATION_VERSION,
        "uses_contextualized_text": True,
        "contextualized_text_format": CONTEXTUALIZED_TEXT_FORMAT,
        "prompt_version": CONTEXTUALIZATION_PROMPT_VERSION,
    }


def build_index_sidecar_metadata(source_report_sha1: str, chunk_count: int) -> dict:
    return {
        "index_version": INDEX_VERSION,
        "contextualization_version": CONTEXTUALIZATION_VERSION,
        "source_report_sha1": source_report_sha1,
        "chunk_count": chunk_count,
        "uses_contextualized_text": True,
        "contextualized_text_format": CONTEXTUALIZED_TEXT_FORMAT,
        "prompt_version": CONTEXTUALIZATION_PROMPT_VERSION,
    }


def get_index_metadata_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.name}.metadata.json")
