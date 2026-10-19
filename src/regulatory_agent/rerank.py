"""Reusable cross-encoder reranking for retrieved regulatory evidence."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol, Sequence

from langchain_core.documents import Document

from regulatory_agent.retrieval import Evidence


BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class EvidenceReranker(Protocol):
    """Minimal interface required by the retrieval tools."""

    def score(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> Sequence[float]:
        """Return one relevance score per document; larger is more relevant."""


class BGEReranker:
    """Sentence Transformers cross-encoder adapter for a BGE reranker model."""

    def __init__(
        self,
        model_name: str = BGE_RERANKER_MODEL,
        *,
        device: str | None = None,
        normalize_scores: bool = True,
    ) -> None:
        # Constructing the vector store or importing this module must not load the
        # relatively expensive cross-encoder model.
        from sentence_transformers import CrossEncoder
        from torch import nn

        model_kwargs = {"device": device} if device is not None else {}
        activation = nn.Sigmoid() if normalize_scores else nn.Identity()
        self._model = CrossEncoder(
            model_name,
            activation_fn=activation,
            **model_kwargs,
        )
        # One reranker instance is shared by concurrently dispatched specialist
        # tools. Protect model inference without serializing their retrieval I/O.
        self._predict_lock = Lock()
        self.score_representation = (
            "sigmoid_transformed_logit" if normalize_scores else "raw_logit"
        )

    def score(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> list[float]:
        pairs = [(question, document.page_content) for document in documents]
        if not pairs:
            return []
        with self._predict_lock:
            scores = self._model.predict(pairs)
        return [float(score) for score in scores]


@dataclass(frozen=True)
class RerankedEvidence:
    """A scored vector hit with global and optional selection ranks.

    ``rerank_rank`` is always the global rank among every scored candidate chunk,
    before parent deduplication. ``selection_rank`` is populated only when the
    result has been selected as a unique parent section.
    """

    evidence: Evidence
    rerank_score: float
    rerank_rank: int
    selection_rank: int | None = None


def rerank_evidence(
    question: str,
    evidence: Sequence[Evidence],
    *,
    reranker: EvidenceReranker,
    top_n: int = 5,
    unique_parents: bool = True,
) -> list[RerankedEvidence]:
    """Rerank vector hits and select the best evidence for section expansion.

    Vector distance remains on ``Evidence.score``. Reranker scores are stored
    separately because cosine distance is ordered ascending while cross-encoder
    relevance is ordered descending.

    ``rerank_rank`` always records position in the complete score-sorted candidate
    list. With ``unique_parents=True``, only the best chunk from each parent is
    returned and ``selection_rank`` records its contiguous position in that
    selected-parent list. With ``unique_parents=False``, ``selection_rank`` is
    ``None``.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not evidence:
        return []

    scores = list(
        reranker.score(
            question,
            [item.document for item in evidence],
        )
    )
    if len(scores) != len(evidence):
        raise ValueError(
            "reranker must return exactly one score for every evidence document"
        )

    # Python's stable sort preserves vector-search order when scores are tied.
    scored = sorted(
        zip(evidence, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    selected: list[tuple[Evidence, float, int]] = []
    seen_parents: set[str] = set()
    for global_rank, (item, score) in enumerate(scored, start=1):
        if unique_parents and item.parent_record_id in seen_parents:
            continue
        seen_parents.add(item.parent_record_id)
        selected.append((item, float(score), global_rank))
        if len(selected) == top_n:
            break

    return [
        RerankedEvidence(
            evidence=item,
            rerank_score=score,
            rerank_rank=global_rank,
            selection_rank=selection_rank if unique_parents else None,
        )
        for selection_rank, (item, score, global_rank) in enumerate(
            selected, start=1
        )
    ]
