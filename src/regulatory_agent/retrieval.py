"""Reusable retrieval and exact parent-section expansion utilities."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_postgres import PGVector

from regulatory_agent.vector_db import (
    COLLECTION_NAME,
    DEFAULT_SEARCH_K,
    retrieve_with_scores,
    vector_document_id,
)


@dataclass(frozen=True)
class Evidence:
    document: Document
    score: float | None
    retrieval_rank: int | None
    parent_record_id: str
    section_complete: bool = True
    evidence_scope: str = "parent_section"
    scope_complete: bool = True


@dataclass(frozen=True)
class ExpansionResult:
    evidence: list[Evidence]
    warnings: list[str]
    selected_parent_ids: list[str] = field(default_factory=list)
    excluded_parent_ids: list[str] = field(default_factory=list)
    expanded_chunk_count: int = 0
    deduplicated_chunk_count: int = 0
    context_character_count: int = 0


def _record_id(document: Document) -> str:
    return str(document.metadata.get("record_id") or document.metadata["chunk_id"])


def retrieve_initial_evidence(
    store: PGVector,
    question: str,
    *,
    k: int = DEFAULT_SEARCH_K,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Evidence]:
    """Retrieve scored chunks and remove duplicate vector hits."""
    results = retrieve_with_scores(
        store,
        question,
        k=k,
        metadata_filter=metadata_filter,
    )
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for rank, (document, score) in enumerate(results, start=1):
        record_id = _record_id(document)
        if record_id in seen:
            continue
        seen.add(record_id)
        parent_id = str(document.metadata.get("parent_record_id") or record_id)
        evidence.append(
            Evidence(
                document=document,
                score=float(score),
                retrieval_rank=rank,
                parent_record_id=parent_id,
            )
        )
    return evidence


def sibling_vector_ids(document: Document) -> list[str]:
    """Derive every exact pgvector ID for a hit's deterministic parent section."""
    metadata = document.metadata
    chunk_count = int(metadata.get("chunk_count", 1))
    if chunk_count <= 1:
        return [vector_document_id(document)]

    chunk_id = str(metadata["chunk_id"])
    chunk_index = int(metadata["chunk_index"])
    suffix = f":{chunk_index:03d}"
    if not chunk_id.endswith(suffix):
        raise ValueError(f"Unexpected deterministic chunk ID: {chunk_id}")
    prefix = chunk_id[: -len(suffix)]
    return [
        f"{COLLECTION_NAME}:{prefix}:{index:03d}"
        for index in range(chunk_count)
    ]


def _ordered_unique_documents(documents: Sequence[Document]) -> list[Document]:
    by_record_id: dict[str, Document] = {}
    for document in documents:
        by_record_id.setdefault(_record_id(document), document)
    return sorted(
        by_record_id.values(),
        key=lambda document: (
            int(document.metadata.get("chunk_index", 0)),
            _record_id(document),
        ),
    )


def _section_is_complete(
    documents: Sequence[Document],
    expected_ids: Sequence[str],
) -> bool:
    """Verify exact IDs, indexes, and provenance coverage for a parent section."""
    if not documents:
        return False
    actual_ids = {vector_document_id(document) for document in documents}
    if actual_ids != set(expected_ids):
        return False

    ordered = _ordered_unique_documents(documents)
    expected_indexes = list(range(len(expected_ids)))
    actual_indexes = [
        int(document.metadata.get("chunk_index", 0)) for document in ordered
    ]
    if actual_indexes != expected_indexes:
        return False

    previous_end: int | None = None
    for document in ordered:
        start = document.metadata.get("body_start")
        end = document.metadata.get("body_end")
        if start is None or end is None or int(end) < int(start):
            return False
        if previous_end is not None and int(start) > previous_end:
            return False
        previous_end = max(previous_end or 0, int(end))
    return True


def _nearby_vector_ids(
    expected_ids: Sequence[str],
    *,
    center_index: int,
    limit: int,
) -> list[str]:
    """Choose a contiguous, relevance-centered subset of sibling IDs."""
    if limit >= len(expected_ids):
        return list(expected_ids)
    start = max(0, center_index - (limit - 1) // 2)
    end = min(len(expected_ids), start + limit)
    start = max(0, end - limit)
    return list(expected_ids[start:end])


def expand_parent_sections(
    store: PGVector,
    retrieved: Sequence[Evidence],
    *,
    max_expanded_chunks: int | None = None,
    max_context_chars: int | None = None,
    paragraph_aware: bool = False,
) -> ExpansionResult:
    """Fetch all exact sibling chunks for only the parents present in hits.

    ``PGVector.get_by_ids`` performs an exact collection-scoped lookup, avoiding
    the incomplete result sets that similarity search plus metadata filters can
    produce. Parents retain first-hit order; siblings retain ``chunk_index`` order.
    """
    if max_expanded_chunks is not None and max_expanded_chunks <= 0:
        raise ValueError("max_expanded_chunks must be positive")
    if max_context_chars is not None and max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")

    by_parent: dict[str, list[Evidence]] = {}
    for item in retrieved:
        by_parent.setdefault(item.parent_record_id, []).append(item)

    expanded: list[Evidence] = []
    warnings: list[str] = []
    selected_parent_ids: list[str] = []
    excluded_parent_ids: list[str] = []
    expanded_chunk_count = 0
    deduplicated_chunk_count = 0
    context_character_count = 0
    for parent_id, hits in by_parent.items():
        representative = hits[0].document
        expected_ids = sibling_vector_ids(representative)
        expected_count = len(expected_ids)

        remaining_chunks = (
            None
            if max_expanded_chunks is None
            else max_expanded_chunks - expanded_chunk_count
        )
        if remaining_chunks is not None and remaining_chunks <= 0:
            excluded_parent_ids.append(parent_id)
            warnings.append(
                f"Section {parent_id} was excluded because the expansion chunk "
                "budget was exhausted."
            )
            continue

        requested_ids = expected_ids
        if remaining_chunks is not None and expected_count > remaining_chunks:
            if selected_parent_ids:
                excluded_parent_ids.append(parent_id)
                warnings.append(
                    f"Section {parent_id} was excluded because its {expected_count} "
                    "chunks exceed the remaining expansion budget."
                )
                continue
            requested_ids = _nearby_vector_ids(
                expected_ids,
                center_index=int(representative.metadata.get("chunk_index", 0)),
                limit=remaining_chunks,
            )

        if expected_count == 1:
            siblings = [representative]
        else:
            siblings = list(store.get_by_ids(requested_ids))
        expanded_chunk_count += len(siblings)

        # Preserve a retrieved hit even if the exact lookup is unexpectedly
        # incomplete, then surface the gap to answer generation.
        siblings.extend(item.document for item in hits)
        ordered = _ordered_unique_documents(siblings)
        deduplicated_chunk_count += len(ordered)
        complete = _section_is_complete(ordered, expected_ids)
        if not complete:
            warnings.append(
                f"Section {parent_id} is incomplete: expected {expected_count} "
                f"chunks, found {len(ordered)} exact chunks within the expansion "
                "budget."
            )

        hit_by_record_id = {_record_id(item.document): item for item in hits}
        parent_evidence: list[Evidence] = []
        for document in ordered:
            hit = hit_by_record_id.get(_record_id(document))
            parent_evidence.append(
                Evidence(
                    document=document,
                    score=hit.score if hit else None,
                    retrieval_rank=hit.retrieval_rank if hit else None,
                    parent_record_id=parent_id,
                    section_complete=complete,
                    scope_complete=complete,
                )
            )

        if paragraph_aware and complete:
            scoped = _paragraph_scoped_evidence(parent_evidence, hits[0])
            if scoped is not None:
                parent_evidence = [scoped]

        remaining_chars = (
            None
            if max_context_chars is None
            else max_context_chars - context_character_count
        )
        if remaining_chars is not None:
            required_chars = len(_deduplicated_section_text(parent_evidence))
            if required_chars > remaining_chars and selected_parent_ids:
                excluded_parent_ids.append(parent_id)
                warnings.append(
                    f"Section {parent_id} was excluded because it exceeds the "
                    "remaining context budget."
                )
                continue
            parent_evidence = _fit_evidence_to_character_budget(
                parent_evidence,
                hit=hits[0],
                max_chars=remaining_chars,
            )
            if not parent_evidence:
                excluded_parent_ids.append(parent_id)
                warnings.append(
                    f"Section {parent_id} was excluded because no evidence fit "
                    "the remaining context budget."
                )
                continue
            if not all(item.scope_complete for item in parent_evidence):
                warnings.append(
                    f"Section {parent_id} was truncated by the context budget."
                )

        parent_chars = len(_deduplicated_section_text(parent_evidence))
        context_character_count += parent_chars
        selected_parent_ids.append(parent_id)
        expanded.extend(parent_evidence)

    return ExpansionResult(
        evidence=expanded,
        warnings=list(dict.fromkeys(warnings)),
        selected_parent_ids=selected_parent_ids,
        excluded_parent_ids=list(dict.fromkeys(excluded_parent_ids)),
        expanded_chunk_count=expanded_chunk_count,
        deduplicated_chunk_count=deduplicated_chunk_count,
        context_character_count=context_character_count,
    )


def _deduplicated_section_text(items: Sequence[Evidence]) -> str:
    """Remove known character overlap while preserving all new source text."""
    if not items:
        return ""
    ordered = sorted(
        items,
        key=lambda item: (
            int(item.document.metadata.get("chunk_index", 0)),
            _record_id(item.document),
        ),
    )
    pieces: list[str] = []
    covered_end: int | None = None
    for position, item in enumerate(ordered):
        document = item.document
        metadata = document.metadata
        start = metadata.get("body_start")
        end = metadata.get("body_end")
        context_chars = int(metadata.get("context_char_count", 0))
        if position == 0 or start is None or end is None or covered_end is None:
            pieces.append(document.page_content)
            covered_end = int(end) if end is not None else None
            continue

        body = document.page_content[context_chars:]
        overlap = max(0, covered_end - int(start))
        new_text = body[min(overlap, len(body)) :]
        if new_text:
            pieces.append(new_text)
        covered_end = max(covered_end, int(end))
    return "\n".join(piece for piece in pieces if piece)


def _deduplicated_body(items: Sequence[Evidence]) -> str:
    """Reconstruct only source body text from ordered provenance offsets."""
    if not items:
        return ""
    ordered = sorted(
        items,
        key=lambda item: (
            int(item.document.metadata.get("chunk_index", 0)),
            _record_id(item.document),
        ),
    )
    pieces: list[str] = []
    covered_end: int | None = None
    for item in ordered:
        document = item.document
        metadata = document.metadata
        context_chars = int(metadata.get("context_char_count", 0))
        body = document.page_content[context_chars:]
        start = metadata.get("body_start")
        end = metadata.get("body_end")
        if start is None or end is None or covered_end is None:
            pieces.append(body)
            covered_end = int(end) if end is not None else None
            continue
        overlap = max(0, covered_end - int(start))
        if overlap < len(body):
            pieces.append(body[overlap:])
        covered_end = max(covered_end, int(end))
    return "".join(pieces)


_TOP_LEVEL_CFR_PARAGRAPH_RE = re.compile(r"(?:^|\n\n)(\([a-z]\))(?=\s)")


def _paragraph_scoped_evidence(
    section: Sequence[Evidence],
    hit: Evidence,
) -> Evidence | None:
    """Extract a verified complete top-level CFR paragraph from a full section.

    Existing metadata identifies the hit's paragraph ancestry but not exact
    paragraph offsets. Therefore this optimization is used only after the full
    parent was reconstructed and matching top-level boundaries were found in the
    source text. Otherwise callers retain the full section.
    """
    document = hit.document
    if document.metadata.get("document_type") != "regulation":
        return None
    paragraph_context = str(document.metadata.get("paragraph_context", ""))
    root_match = re.match(r"^(\([a-z]\))", paragraph_context)
    if root_match is None:
        return None
    root = root_match.group(1)

    body = _deduplicated_body(section)
    boundaries = [
        (match.group(1), match.start(1))
        for match in _TOP_LEVEL_CFR_PARAGRAPH_RE.finditer(body)
    ]
    matching_positions = [
        index for index, (identifier, _) in enumerate(boundaries) if identifier == root
    ]
    if len(matching_positions) != 1:
        return None
    boundary_index = matching_positions[0]
    start = boundaries[boundary_index][1]
    end = (
        boundaries[boundary_index + 1][1]
        if boundary_index + 1 < len(boundaries)
        else len(body)
    )
    scoped_body = body[start:end].rstrip()
    if not scoped_body.startswith(root) or not scoped_body:
        return None

    original_context_chars = int(document.metadata.get("context_char_count", 0))
    context = document.page_content[:original_context_chars]
    scoped_id = f"{hit.parent_record_id}:paragraph-scope:{root[1:-1]}"
    metadata = dict(document.metadata)
    metadata.update(
        {
            "record_id": scoped_id,
            "chunk_id": scoped_id,
            "source_record_id": hit.parent_record_id,
            "synthetic_evidence": True,
            "chunk_index": 0,
            "chunk_count": 1,
            "body_start": start,
            "body_end": end,
            "overlap_chars": 0,
            "context_char_count": len(context),
            "paragraph_context": root,
            "paragraph_start": root,
            "paragraph_end": root,
            "evidence_scope": f"paragraph:{root}",
            "source_section_chunk_count": len(section),
        }
    )
    return Evidence(
        document=Document(page_content=context + scoped_body, metadata=metadata),
        score=hit.score,
        retrieval_rank=hit.retrieval_rank,
        parent_record_id=hit.parent_record_id,
        section_complete=False,
        evidence_scope=f"paragraph:{root}",
        scope_complete=True,
    )


def _fit_evidence_to_character_budget(
    evidence: Sequence[Evidence],
    *,
    hit: Evidence,
    max_chars: int,
) -> list[Evidence]:
    """Keep a relevance-centered chunk neighborhood within a character budget."""
    if max_chars <= 0:
        return []
    if len(_deduplicated_section_text(evidence)) <= max_chars:
        return list(evidence)

    # A paragraph-scoped synthetic document cannot be safely sliced further.
    # Fall back to its original retrieved chunk if that chunk fits.
    if len(evidence) == 1 and evidence[0].evidence_scope.startswith("paragraph:"):
        fallback = replace(
            hit,
            section_complete=False,
            evidence_scope="budgeted_chunks",
            scope_complete=False,
        )
        return (
            [fallback]
            if len(_deduplicated_section_text([fallback])) <= max_chars
            else []
        )

    ordered = sorted(
        evidence,
        key=lambda item: int(item.document.metadata.get("chunk_index", 0)),
    )
    hit_index = int(hit.document.metadata.get("chunk_index", 0))
    selected: list[Evidence] = []
    for item in sorted(
        ordered,
        key=lambda item: (
            abs(int(item.document.metadata.get("chunk_index", 0)) - hit_index),
            int(item.document.metadata.get("chunk_index", 0)),
        ),
    ):
        proposed = sorted(
            [*selected, item],
            key=lambda value: int(value.document.metadata.get("chunk_index", 0)),
        )
        if len(_deduplicated_section_text(proposed)) <= max_chars:
            selected = proposed

    return [
        replace(
            item,
            section_complete=False,
            evidence_scope="budgeted_chunks",
            scope_complete=False,
        )
        for item in selected
    ]


def citation_label(document: Document) -> str:
    metadata = document.metadata
    if metadata.get("document_type") == "regulation":
        section_id = str(metadata.get("section_id", ""))
        return f"21 CFR § {section_id}" if section_id else str(
            metadata.get("citation", "")
        )
    return str(metadata.get("citation", ""))


def supporting_citations(evidence: Sequence[Evidence]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        label = citation_label(item.document)
        if label and label not in seen:
            seen.add(label)
            citations.append(label)
    return citations


def render_evidence_context(
    evidence: Sequence[Evidence],
    *,
    evidence_ids_by_parent: Mapping[str, str] | None = None,
) -> str:
    """Render ordered, overlap-deduplicated sections for an LLM prompt.

    ``evidence_ids_by_parent`` exposes request-local IDs such as ``E1`` without
    conflating them with persistent database record IDs. The plain-text
    delimiters intentionally are not citation placeholders; callers should
    provide an explicit citation-reference contract separately.
    """
    by_parent: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_parent.setdefault(item.parent_record_id, []).append(item)

    sections: list[str] = []
    for index, (parent_id, items) in enumerate(by_parent.items(), start=1):
        first = items[0].document
        metadata = first.metadata
        source_kind = (
            "Binding CFR regulation"
            if metadata.get("document_type") == "regulation"
            else "Nonbinding FDA guidance"
        )
        paragraph_contexts = list(
            dict.fromkeys(
                str(item.document.metadata.get("paragraph_context", ""))
                for item in items
                if item.document.metadata.get("paragraph_context")
            )
        )
        pages = list(
            dict.fromkeys(
                str(item.document.metadata.get("page_start"))
                for item in items
                if item.document.metadata.get("page_start") is not None
            )
        )
        evidence_id = (
            evidence_ids_by_parent.get(parent_id, "")
            if evidence_ids_by_parent is not None
            else ""
        )
        record_ids = list(
            dict.fromkeys(_record_id(item.document) for item in items)
        )
        sections.append(
            "\n".join(
                [
                    f"<evidence_section index=\"{index}\">",
                    f"evidence_id: {evidence_id}",
                    f"source_type: {source_kind}",
                    f"citation: {citation_label(first)}",
                    f"source_record_id: {parent_id}",
                    f"evidence_record_ids: {', '.join(record_ids)}",
                    f"section_id: {metadata.get('section_id', '')}",
                    f"section_complete: {all(item.section_complete for item in items)}",
                    f"evidence_scope: {items[0].evidence_scope}",
                    f"scope_complete: {all(item.scope_complete for item in items)}",
                    f"paragraph_contexts: {', '.join(paragraph_contexts)}",
                    f"pages: {', '.join(pages)}",
                    f"source_url: {metadata.get('source_url', '')}",
                    "--- BEGIN SUPPLIED EVIDENCE TEXT ---",
                    _deduplicated_section_text(items),
                    "--- END SUPPLIED EVIDENCE TEXT ---",
                    "</evidence_section>",
                ]
            )
        )
    return "\n\n".join(sections)
