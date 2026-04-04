from typing import Literal, TypedDict, TypeAlias


RetrieverName: TypeAlias = Literal["vector", "bm25"]


class RetrievalResultCommon(TypedDict):
    """
    Fields shared by every retrieval result regardless of granularity.
    """

    doc_id: str
    page: int
    text: str
    retrieval_score: float


class ChunkRetrievalResult(RetrievalResultCommon):
    """
    Chunk-only fields. Present for both single-branch and fused chunk results.
    """

    chunk_uid: str
    chunk_idx: int


class BranchChunkRetrievalResult(ChunkRetrievalResult):
    """
    Fields that only exist before fusion, on a single retrieval branch.
    """

    raw_score: float
    retriever: RetrieverName


class FusedChunkRetrievalResult(ChunkRetrievalResult, total=False):
    """
    Fields that only exist after chunk-level RRF fusion.
    """

    retrievers: list[RetrieverName]
    vector_raw_score: float
    bm25_raw_score: float


class PageRetrievalResult(RetrievalResultCommon):
    """
    Page-level results intentionally do not carry chunk identifiers.
    """


class PublicChunkRetrievalResult(BranchChunkRetrievalResult, total=False):
    """
    Public chunk result with optional compatibility and reranking fields.
    """

    distance: float
    relevance_score: float
    combined_score: float


class PublicFusedChunkRetrievalResult(FusedChunkRetrievalResult, total=False):
    """
    Public fused chunk result with optional compatibility and reranking fields.
    """

    distance: float
    relevance_score: float
    combined_score: float


class PublicPageRetrievalResult(PageRetrievalResult, total=False):
    """
    Public page result with optional compatibility and reranking fields.
    """

    distance: float
    relevance_score: float
    combined_score: float


InternalChunkRetrievalResult: TypeAlias = BranchChunkRetrievalResult | FusedChunkRetrievalResult
InternalRetrievalResult: TypeAlias = InternalChunkRetrievalResult | PageRetrievalResult
PublicRetrievalResult: TypeAlias = (
    PublicChunkRetrievalResult | PublicFusedChunkRetrievalResult | PublicPageRetrievalResult
)
