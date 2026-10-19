"""Source-specific retrieval tools for CFR regulations and FDA guidance."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Literal, Sequence

from langchain_core.tools import BaseTool, tool
from langchain_postgres import PGVector

from regulatory_agent.rerank import (
    EvidenceReranker,
    RerankedEvidence,
    rerank_evidence,
)
from regulatory_agent.retrieval import (
    Evidence,
    expand_parent_sections,
    render_evidence_context,
    retrieve_initial_evidence,
)


DocumentType = Literal["regulation", "guidance"]


@dataclass(frozen=True)
class RegulatorySearchResult:
    """Structured result retained underneath the text-oriented Agent tools."""

    document_type: DocumentType
    candidates: list[Evidence]
    reranked: list[RerankedEvidence]
    expanded_evidence: list[Evidence]
    warnings: list[str]
    diagnostics: "SearchDiagnostics"


@dataclass(frozen=True)
class SearchDiagnostics:
    """Concise pipeline measurements without verbose document payloads."""

    candidate_count: int
    reranked_count: int
    selected_parent_sections: list[str]
    expanded_chunk_count: int
    deduplicated_chunk_count: int
    approximate_context_tokens: int
    excluded_due_to_relevance: list[str]
    excluded_due_to_budget: list[str]
    evidence_complete: bool
    rerank_score_representation: str
    retrieval_latency_ms: float
    reranking_latency_ms: float
    expansion_latency_ms: float
    total_latency_ms: float
    selected_parent_rerank_ranks: list[int] = field(default_factory=list)
    selected_parent_selection_ranks: list[int] = field(default_factory=list)


def retrieve_rerank_expand(
    *,
    store: PGVector,
    reranker: EvidenceReranker,
    question: str,
    document_type: DocumentType,
    candidate_k: int = 15,
    top_n: int = 5,
    min_rerank_score: float | None = None,
    context_char_budget: int = 12_000,
    max_expanded_chunks: int = 12,
    paragraph_aware: bool = True,
) -> RegulatorySearchResult:
    """Run the shared retrieval pipeline for exactly one regulatory source type."""
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if top_n > candidate_k:
        raise ValueError("top_n must not exceed candidate_k")
    if context_char_budget <= 0:
        raise ValueError("context_char_budget must be positive")
    if max_expanded_chunks <= 0:
        raise ValueError("max_expanded_chunks must be positive")

    started = perf_counter()
    candidates = retrieve_initial_evidence(
        store,
        question,
        k=candidate_k,
        metadata_filter={"document_type": document_type},
    )
    retrieved_at = perf_counter()
    all_reranked = rerank_evidence(
        question,
        candidates,
        reranker=reranker,
        top_n=max(1, len(candidates)),
        unique_parents=False,
    )
    reranked_at = perf_counter()

    reranked: list[RerankedEvidence] = []
    excluded_due_to_relevance: list[str] = []
    seen_parents: set[str] = set()
    for item in all_reranked:
        parent_id = item.evidence.parent_record_id
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        if (
            min_rerank_score is not None
            and item.rerank_score < min_rerank_score
        ):
            excluded_due_to_relevance.append(parent_id)
            continue
        if len(reranked) < top_n:
            reranked.append(
                replace(item, selection_rank=len(reranked) + 1)
            )
        else:
            excluded_due_to_relevance.append(parent_id)

    expansion = expand_parent_sections(
        store,
        [item.evidence for item in reranked],
        max_expanded_chunks=max_expanded_chunks,
        max_context_chars=context_char_budget,
        paragraph_aware=paragraph_aware,
    )
    expanded_at = perf_counter()
    selected_by_parent = {
        item.evidence.parent_record_id: item for item in reranked
    }
    expanded_selections = [
        selected_by_parent[parent_id]
        for parent_id in expansion.selected_parent_ids
    ]
    diagnostics = SearchDiagnostics(
        candidate_count=len(candidates),
        reranked_count=len(all_reranked),
        selected_parent_sections=expansion.selected_parent_ids,
        expanded_chunk_count=expansion.expanded_chunk_count,
        deduplicated_chunk_count=expansion.deduplicated_chunk_count,
        approximate_context_tokens=(expansion.context_character_count + 3) // 4,
        excluded_due_to_relevance=excluded_due_to_relevance,
        excluded_due_to_budget=expansion.excluded_parent_ids,
        evidence_complete=all(
            item.scope_complete for item in expansion.evidence
        ) and not expansion.excluded_parent_ids,
        rerank_score_representation=str(
            getattr(reranker, "score_representation", "implementation_defined")
        ),
        retrieval_latency_ms=(retrieved_at - started) * 1000,
        reranking_latency_ms=(reranked_at - retrieved_at) * 1000,
        expansion_latency_ms=(expanded_at - reranked_at) * 1000,
        total_latency_ms=(expanded_at - started) * 1000,
        selected_parent_rerank_ranks=[
            item.rerank_rank for item in expanded_selections
        ],
        selected_parent_selection_ranks=[
            item.selection_rank
            for item in expanded_selections
            if item.selection_rank is not None
        ],
    )
    return RegulatorySearchResult(
        document_type=document_type,
        candidates=candidates,
        reranked=reranked,
        expanded_evidence=expansion.evidence,
        warnings=expansion.warnings,
        diagnostics=diagnostics,
    )


def render_tool_result(result: RegulatorySearchResult) -> str:
    """Render a search result as grounded context suitable for an Agent tool."""
    source_label = (
        "binding CFR regulation"
        if result.document_type == "regulation"
        else "nonbinding FDA guidance"
    )
    if not result.reranked:
        return f"No relevant {source_label} evidence was found."

    selection = [
        "Reranked section selections:",
        *[
            (
                f"- rerank_rank={item.rerank_rank}; "
                f"selection_rank={item.selection_rank}; "
                f"rerank_score={item.rerank_score:.6f}; "
                f"vector_distance={item.evidence.score:.6f}; "
                f"citation={item.evidence.document.metadata.get('citation', '')}; "
                f"parent_record_id={item.evidence.parent_record_id}"
            )
            for item in result.reranked
        ],
    ]
    warnings = (
        ["Section completeness warnings:", *[f"- {item}" for item in result.warnings]]
        if result.warnings
        else ["Section completeness warnings: None"]
    )
    diagnostics = result.diagnostics
    diagnostic_lines = [
        "Search diagnostics:",
        f"- candidate_count={diagnostics.candidate_count}",
        f"- reranked_count={diagnostics.reranked_count}",
        (
            "- selected_parent_sections="
            f"{', '.join(diagnostics.selected_parent_sections) or 'None'}"
        ),
        (
            "- selected_parent_rerank_ranks="
            f"{diagnostics.selected_parent_rerank_ranks}"
        ),
        (
            "- selected_parent_selection_ranks="
            f"{diagnostics.selected_parent_selection_ranks}"
        ),
        f"- expanded_chunk_count={diagnostics.expanded_chunk_count}",
        f"- deduplicated_chunk_count={diagnostics.deduplicated_chunk_count}",
        (
            "- approximate_context_tokens="
            f"{diagnostics.approximate_context_tokens}"
        ),
        (
            "- excluded_due_to_relevance="
            f"{', '.join(diagnostics.excluded_due_to_relevance) or 'None'}"
        ),
        (
            "- excluded_due_to_budget="
            f"{', '.join(diagnostics.excluded_due_to_budget) or 'None'}"
        ),
        f"- evidence_complete={diagnostics.evidence_complete}",
        (
            "- rerank_score_representation="
            f"{diagnostics.rerank_score_representation}"
        ),
        (
            "- latency_ms="
            f"retrieval:{diagnostics.retrieval_latency_ms:.2f},"
            f"reranking:{diagnostics.reranking_latency_ms:.2f},"
            f"expansion:{diagnostics.expansion_latency_ms:.2f},"
            f"total:{diagnostics.total_latency_ms:.2f}"
        ),
    ]
    return "\n".join(
        [
            f"Source scope: {source_label}",
            f"Vector candidates considered: {len(result.candidates)}",
            *selection,
            *diagnostic_lines,
            *warnings,
            "",
            render_evidence_context(result.expanded_evidence),
        ]
    )


def create_fda_guidance_tool(
    *,
    store: PGVector,
    reranker: EvidenceReranker,
    candidate_k: int = 15,
    top_n: int = 5,
    min_rerank_score: float | None = None,
    context_char_budget: int = 12_000,
    max_expanded_chunks: int = 12,
    paragraph_aware: bool = True,
) -> BaseTool:
    """Create a tool restricted to nonbinding FDA guidance evidence."""

    @tool("search_fda_guidance")
    def search_fda_guidance(question: str) -> str:
        """Search nonbinding FDA guidance for evidence relevant to a question.

        Use this tool for FDA recommendations, interpretations, best practices,
        and implementation guidance. Its output is not a binding legal rule.
        """
        result = retrieve_rerank_expand(
            store=store,
            reranker=reranker,
            question=question,
            document_type="guidance",
            candidate_k=candidate_k,
            top_n=top_n,
            min_rerank_score=min_rerank_score,
            context_char_budget=context_char_budget,
            max_expanded_chunks=max_expanded_chunks,
            paragraph_aware=paragraph_aware,
        )
        return render_tool_result(result)

    return search_fda_guidance


def create_cfr_regulation_tool(
    *,
    store: PGVector,
    reranker: EvidenceReranker,
    candidate_k: int = 15,
    top_n: int = 5,
    min_rerank_score: float | None = None,
    context_char_budget: int = 12_000,
    max_expanded_chunks: int = 12,
    paragraph_aware: bool = True,
) -> BaseTool:
    """Create a tool restricted to binding CFR regulation evidence."""

    @tool("search_cfr_regulations")
    def search_cfr_regulations(question: str) -> str:
        """Search binding CFR regulations for legal requirements relevant to a question.

        Use this tool when the answer requires regulatory duties, prohibitions,
        exceptions, definitions, or other binding requirements.
        """
        result = retrieve_rerank_expand(
            store=store,
            reranker=reranker,
            question=question,
            document_type="regulation",
            candidate_k=candidate_k,
            top_n=top_n,
            min_rerank_score=min_rerank_score,
            context_char_budget=context_char_budget,
            max_expanded_chunks=max_expanded_chunks,
            paragraph_aware=paragraph_aware,
        )
        return render_tool_result(result)

    return search_cfr_regulations


def create_regulatory_search_tools(
    *,
    store: PGVector,
    reranker: EvidenceReranker,
    candidate_k: int = 15,
    top_n: int = 5,
    min_rerank_score: float | None = None,
    context_char_budget: int = 12_000,
    max_expanded_chunks: int = 12,
    paragraph_aware: bool = True,
) -> Sequence[BaseTool]:
    """Return both source-specific tools for later use by a routing Agent."""
    return [
        create_cfr_regulation_tool(
            store=store,
            reranker=reranker,
            candidate_k=candidate_k,
            top_n=top_n,
            min_rerank_score=min_rerank_score,
            context_char_budget=context_char_budget,
            max_expanded_chunks=max_expanded_chunks,
            paragraph_aware=paragraph_aware,
        ),
        create_fda_guidance_tool(
            store=store,
            reranker=reranker,
            candidate_k=candidate_k,
            top_n=top_n,
            min_rerank_score=min_rerank_score,
            context_char_budget=context_char_budget,
            max_expanded_chunks=max_expanded_chunks,
            paragraph_aware=paragraph_aware,
        ),
    ]
