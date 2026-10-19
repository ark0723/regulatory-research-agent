"""Alternative retrieval preparation strategies for regulatory Documents.

Baseline A keeps each indexable source Document intact. Experiment B splits
only long section Documents, preferring source paragraph and sentence
boundaries. Linked FDA footnotes always remain intact.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from regulatory_agent.documents import (
    CFR_PATH,
    build_cfr_documents,
    build_guidance_documents,
)
from regulatory_agent.paths import DOCS_DIR


DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
REPORT_PATH = DOCS_DIR / "experiments/chunking_analysis.md"

PARAGRAPH_ID_RE = re.compile(
    r"^\s*((?:\([A-Za-z0-9ivxlcdmIVXLCDM]+\))+)", re.UNICODE
)
TOKEN_RE = re.compile(r"\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)")
SENTENCE_END_RE = re.compile(r"[.!?](?:[”\"']|\))?\s+")


@dataclass(frozen=True)
class ParagraphSpan:
    start: int
    end: int
    identifier: str
    context: str


def _copy_document(document: Document) -> Document:
    return Document(
        page_content=document.page_content,
        metadata=dict(document.metadata),
    )


def _context_and_body(page_content: str) -> tuple[str, str]:
    context, separator, body = page_content.partition("\n\n")
    if not separator:
        return "", page_content
    return context, body


def _preparation_metadata(
    document: Document,
    *,
    strategy: str,
) -> dict[str, Any]:
    metadata = dict(document.metadata)
    record_id = metadata["record_id"]
    _, body = _context_and_body(document.page_content)
    context_chars = len(document.page_content) - len(body)
    metadata.update(
        {
            "parent_record_id": record_id,
            "chunk_id": record_id,
            "chunk_index": 0,
            "chunk_count": 1,
            "chunking_strategy": strategy,
            "body_start": 0,
            "body_end": len(body),
            "overlap_chars": 0,
            "context_char_count": context_chars,
            "chunk_size_exception": len(document.page_content) > DEFAULT_CHUNK_SIZE,
        }
    )
    return metadata


def _indexable_source_documents() -> list[Document]:
    guidance_sections, guidance_footnotes = build_guidance_documents()
    linked_footnotes = [
        document
        for document in guidance_footnotes
        if document.metadata.get("footnote_attribution") == "associated"
    ]
    return build_cfr_documents() + guidance_sections + linked_footnotes


def build_baseline_documents(
    documents: Iterable[Document] | None = None,
) -> list[Document]:
    """Build Baseline A without splitting or mutating source Documents."""
    source_documents = (
        list(documents) if documents is not None else _indexable_source_documents()
    )
    baseline: list[Document] = []
    for source in source_documents:
        document = _copy_document(source)
        document.metadata = _preparation_metadata(document, strategy="section")
        baseline.append(document)
    _validate_unique_chunk_ids(baseline)
    return baseline


def _paragraph_identifier(text: str) -> str:
    match = PARAGRAPH_ID_RE.match(text)
    return match.group(1) if match else ""


def _is_roman(token: str) -> bool:
    return bool(re.fullmatch(r"[ivxlcdm]+", token))


def _paragraph_contexts(paragraphs: list[dict[str, Any]]) -> list[str]:
    """Infer identifier ancestry without inventing identifiers."""
    hierarchy: list[str] = []
    contexts: list[str] = []
    for paragraph in paragraphs:
        identifier = _paragraph_identifier(paragraph.get("text", ""))
        tokens = TOKEN_RE.findall(identifier)
        if not tokens:
            contexts.append("".join(f"({token})" for token in hierarchy))
            continue

        if len(tokens) > 1:
            hierarchy = tokens.copy()
        else:
            token = tokens[0]
            if token.isdigit():
                depth = 2
            elif len(token) == 1 and token.isupper():
                depth = 4
            elif _is_roman(token) and len(hierarchy) >= 2 and hierarchy[1].isdigit():
                depth = 3
            else:
                depth = 1
            hierarchy = hierarchy[: depth - 1]
            while len(hierarchy) < depth - 1:
                # Missing parents remain absent rather than being fabricated.
                hierarchy.append("")
            hierarchy.append(token)
        contexts.append("".join(f"({token})" for token in hierarchy if token))
    return contexts


def _cfr_paragraph_spans(
    body: str, paragraphs: list[dict[str, Any]]
) -> list[ParagraphSpan]:
    contexts = _paragraph_contexts(paragraphs)
    spans: list[ParagraphSpan] = []
    cursor = 0
    for index, paragraph in enumerate(paragraphs):
        text = paragraph.get("text", "")
        start = body.find(text, cursor)
        if start < 0:
            raise ValueError(
                f"CFR paragraph {index} is not present in its section body"
            )
        end = start + len(text)
        spans.append(
            ParagraphSpan(
                start=start,
                end=end,
                identifier=_paragraph_identifier(text),
                context=contexts[index],
            )
        )
        cursor = end
    return spans


def _paragraph_boundaries(body: str) -> set[int]:
    boundaries = {0, len(body)}
    for match in re.finditer(r"\n\n+", body):
        boundaries.add(match.end())
    return boundaries


def _sentence_boundaries(body: str) -> set[int]:
    return {match.end() for match in SENTENCE_END_RE.finditer(body)}


def _choose_end(
    start: int,
    max_end: int,
    body_length: int,
    paragraph_boundaries: set[int],
    sentence_boundaries: set[int],
) -> tuple[int, str]:
    if max_end >= body_length:
        return body_length, "document_end"
    # Do not select a very early paragraph boundary merely because it has
    # higher priority; that can produce chunks made almost entirely of overlap.
    preferred_minimum = start + max(1, (max_end - start) // 2)
    paragraph_ends = [
        boundary
        for boundary in paragraph_boundaries
        if preferred_minimum <= boundary <= max_end
    ]
    if paragraph_ends:
        return max(paragraph_ends), "paragraph"
    sentence_ends = [
        boundary
        for boundary in sentence_boundaries
        if preferred_minimum <= boundary <= max_end
    ]
    if sentence_ends:
        return max(sentence_ends), "sentence"
    return max_end, "hard_limit"


def _choose_overlap_start(
    current_start: int,
    end: int,
    overlap: int,
    boundaries: set[int],
) -> int:
    desired = max(current_start + 1, end - overlap)
    candidates = [
        boundary for boundary in boundaries if current_start < boundary < end
    ]
    if not candidates:
        return desired
    return min(candidates, key=lambda boundary: (abs(boundary - desired), boundary))


def _split_ranges(
    body: str,
    *,
    body_budget: int,
    chunk_overlap: int,
) -> list[tuple[int, int, str]]:
    paragraph_boundaries = _paragraph_boundaries(body)
    sentence_boundaries = _sentence_boundaries(body)
    all_boundaries = paragraph_boundaries | sentence_boundaries
    ranges: list[tuple[int, int, str]] = []
    start = 0

    while start < len(body):
        end, boundary_type = _choose_end(
            start,
            start + body_budget,
            len(body),
            paragraph_boundaries,
            sentence_boundaries,
        )
        if end <= start:
            raise ValueError("Chunking failed to make forward progress")
        ranges.append((start, end, boundary_type))
        if end == len(body):
            break
        next_start = _choose_overlap_start(
            start, end, chunk_overlap, all_boundaries
        )
        if next_start <= start:
            next_start = min(end, start + body_budget)
        start = next_start
    return ranges


def _span_for_offset(
    spans: list[ParagraphSpan], offset: int
) -> ParagraphSpan | None:
    for span in spans:
        if span.start <= offset < span.end:
            return span
        if offset < span.start:
            return span
    return spans[-1] if spans else None


def _span_for_end(
    spans: list[ParagraphSpan], end: int
) -> ParagraphSpan | None:
    candidates = [span for span in spans if span.start < end]
    return candidates[-1] if candidates else None


def _chunk_section(
    document: Document,
    *,
    chunk_size: int,
    chunk_overlap: int,
    cfr_paragraphs: list[dict[str, Any]] | None,
) -> list[Document]:
    context, body = _context_and_body(document.page_content)
    if not context or not body:
        raise ValueError(
            f"Document {document.metadata['record_id']} has no context/body separator"
        )

    is_cfr = document.metadata["document_type"] == "regulation"
    paragraph_spans = (
        _cfr_paragraph_spans(body, cfr_paragraphs or []) if is_cfr else []
    )
    # CFR chunks may add a short paragraph-context line. Reserving space keeps
    # the total Document length within the configured target in normal cases.
    context_reserve = 100 if is_cfr else 0
    body_budget = chunk_size - len(context) - 2 - context_reserve
    if body_budget < 200:
        raise ValueError(
            f"chunk_size={chunk_size} leaves too little room after context"
        )
    ranges = _split_ranges(
        body, body_budget=body_budget, chunk_overlap=chunk_overlap
    )
    if len(ranges) == 1:
        document_copy = _copy_document(document)
        metadata = _preparation_metadata(
            document_copy, strategy="structure_aware"
        )
        metadata["chunk_size_exception"] = len(document_copy.page_content) > chunk_size
        document_copy.metadata = metadata
        return [document_copy]

    parent_id = document.metadata["record_id"]
    chunks: list[Document] = []
    previous_end = 0
    for index, (start, end, boundary_type) in enumerate(ranges):
        start_span = _span_for_offset(paragraph_spans, start) if is_cfr else None
        end_span = _span_for_end(paragraph_spans, end) if is_cfr else None
        paragraph_context = start_span.context if start_span else ""
        chunk_context = context
        if is_cfr and paragraph_context:
            chunk_context += f"\nParagraph context: {paragraph_context}"
        context_prefix = chunk_context + "\n\n"
        chunk_body = body[start:end]
        page_content = context_prefix + chunk_body
        chunk_id = (
            f"{parent_id}:structure-aware:{chunk_size}-{chunk_overlap}:"
            f"{index:03d}"
        )
        metadata = dict(document.metadata)
        metadata.update(
            {
                "record_id": chunk_id,
                "parent_record_id": parent_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "chunk_count": len(ranges),
                "chunking_strategy": "structure_aware",
                "body_start": start,
                "body_end": end,
                "overlap_chars": max(0, previous_end - start),
                "context_char_count": len(context_prefix),
                "chunk_boundary": boundary_type,
                "chunk_size_exception": len(page_content) > chunk_size,
            }
        )
        if is_cfr:
            metadata.update(
                {
                    "paragraph_context": paragraph_context,
                    "paragraph_start": start_span.identifier if start_span else "",
                    "paragraph_end": end_span.identifier if end_span else "",
                }
            )
        chunks.append(Document(page_content=page_content, metadata=metadata))
        previous_end = end
    return chunks


def _cfr_paragraphs_by_section(
    records_path: Path = CFR_PATH,
) -> dict[str, list[dict[str, Any]]]:
    records = json.loads(Path(records_path).read_text(encoding="utf-8"))
    return {record["section"]: record["paragraphs"] for record in records}


def build_chunked_documents(
    documents: Iterable[Document] | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    cfr_records_path: Path = CFR_PATH,
) -> list[Document]:
    """Build Experiment B with section-aware, deterministic chunking."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    source_documents = (
        list(documents) if documents is not None else _indexable_source_documents()
    )
    cfr_paragraphs = _cfr_paragraphs_by_section(cfr_records_path)
    prepared: list[Document] = []
    for source in source_documents:
        if source.metadata["content_type"] == "footnote":
            document = _copy_document(source)
            document.metadata = _preparation_metadata(
                document, strategy="structure_aware"
            )
            document.metadata["chunk_size_exception"] = (
                len(document.page_content) > chunk_size
            )
            prepared.append(document)
            continue

        if len(source.page_content) <= chunk_size:
            document = _copy_document(source)
            document.metadata = _preparation_metadata(
                document, strategy="structure_aware"
            )
            document.metadata["chunk_size_exception"] = False
            prepared.append(document)
            continue

        paragraphs = (
            cfr_paragraphs.get(source.metadata["section_id"])
            if source.metadata["document_type"] == "regulation"
            else None
        )
        prepared.extend(
            _chunk_section(
                source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                cfr_paragraphs=paragraphs,
            )
        )

    _validate_unique_chunk_ids(prepared)
    return prepared


def _validate_unique_chunk_ids(documents: list[Document]) -> None:
    chunk_ids = [document.metadata["chunk_id"] for document in documents]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id values detected")


def reconstruct_body(parent: Document, chunks: Iterable[Document]) -> str:
    """Reconstruct and verify a parent body using chunk provenance offsets."""
    _, original_body = _context_and_body(parent.page_content)
    reconstructed: list[str | None] = [None] * len(original_body)
    ordered = sorted(chunks, key=lambda item: item.metadata["chunk_index"])
    for chunk in ordered:
        start = chunk.metadata["body_start"]
        end = chunk.metadata["body_end"]
        context_chars = chunk.metadata["context_char_count"]
        chunk_body = chunk.page_content[context_chars:]
        if len(chunk_body) != end - start:
            raise ValueError("Chunk body length does not match provenance offsets")
        for offset, character in enumerate(chunk_body, start=start):
            previous = reconstructed[offset]
            if previous is not None and previous != character:
                raise ValueError("Overlapping chunks contain conflicting text")
            reconstructed[offset] = character
    if any(character is None for character in reconstructed):
        raise ValueError("Chunk ranges do not cover the complete parent body")
    result = "".join(character or "" for character in reconstructed)
    if result != original_body:
        raise ValueError("Reconstructed body differs from the source body")
    return result


def _length_stats(documents: list[Document]) -> dict[str, float | int]:
    lengths = sorted(len(document.page_content) for document in documents)
    percentile_index = max(0, int(len(lengths) * 0.95) - 1)
    return {
        "min": min(lengths),
        "median": statistics.median(lengths),
        "p95": lengths[percentile_index],
        "max": max(lengths),
    }


def _counts_by_type(documents: list[Document]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        key = (
            f"{document.metadata['document_type']}/"
            f"{document.metadata['content_type']}"
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_chunking_report(
    baseline: list[Document],
    chunked: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    output_path: Path = REPORT_PATH,
) -> None:
    baseline_by_id = {
        document.metadata["record_id"]: document for document in baseline
    }
    chunks_by_parent: dict[str, list[Document]] = {}
    for document in chunked:
        chunks_by_parent.setdefault(
            document.metadata["parent_record_id"], []
        ).append(document)

    split_parents = {
        parent_id: chunks
        for parent_id, chunks in chunks_by_parent.items()
        if len(chunks) > 1
    }
    section_parent_ids = {
        document.metadata["record_id"]
        for document in baseline
        if document.metadata["content_type"] == "section"
    }
    split_section_count = len(set(split_parents) & section_parent_ids)
    unchanged_section_count = len(section_parent_ids) - split_section_count
    unchanged_footnote_count = sum(
        document.metadata["content_type"] == "footnote" for document in baseline
    )
    exceptions = [
        document
        for document in chunked
        if document.metadata.get("chunk_size_exception")
    ]
    total_overlap = sum(
        int(document.metadata.get("overlap_chars", 0)) for document in chunked
    )
    boundary_counts = {
        boundary: sum(
            document.metadata.get("chunk_boundary") == boundary
            for document in chunked
        )
        for boundary in ("paragraph", "sentence", "hard_limit", "document_end")
    }
    hard_limit_details = [
        f"{document.metadata['parent_record_id']} chunk "
        f"{document.metadata['chunk_index']}"
        for document in chunked
        if document.metadata.get("chunk_boundary") == "hard_limit"
    ]

    for parent_id, chunks in chunks_by_parent.items():
        reconstruct_body(baseline_by_id[parent_id], chunks)

    def count_rows(values: dict[str, int]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in sorted(values.items()))

    def stats_rows(values: dict[str, float | int]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in values.items())

    examples: list[str] = []
    for document_type, section_id in (
        ("regulation", "50.23"),
        ("regulation", "50.3"),
        ("regulation", "50.24"),
        ("guidance", "V.5"),
        ("guidance", "V.1"),
        ("guidance", "III.A.2"),
    ):
        parent = next(
            document
            for document in baseline
            if document.metadata["document_type"] == document_type
            and document.metadata["section_id"] == section_id
        )
        chunks = chunks_by_parent[parent.metadata["record_id"]]
        lengths = ", ".join(str(len(chunk.page_content)) for chunk in chunks)
        boundary_summary = ", ".join(
            str(chunk.metadata.get("chunk_boundary", "unchanged"))
            for chunk in chunks
        )
        examples.append(
            f"- {document_type} `{section_id}`: {len(chunks)} chunks; "
            f"lengths [{lengths}]; boundaries [{boundary_summary}]"
        )

    report = f"""# Section-aware chunking analysis

## Experiment configuration

- Baseline A: section-level Documents without splitting
- Experiment B: structure-aware splitting for long sections
- Target chunk size: {chunk_size} characters, including contextual header
- Requested overlap: {chunk_overlap} characters; actual overlap is boundary-aligned
- Unlinked FDA footnote 1: excluded from both retrieval strategies and retained in source review data

## Document counts

- Baseline A: {len(baseline)}
- Experiment B: {len(chunked)}
- Section Documents split: {split_section_count}
- Section Documents unchanged: {unchanged_section_count}
- Linked footnotes unchanged: {unchanged_footnote_count}

### Baseline by type

{count_rows(_counts_by_type(baseline))}

### Experiment B by type

{count_rows(_counts_by_type(chunked))}

## Chunk length statistics

### Baseline A

{stats_rows(_length_stats(baseline))}

### Experiment B

{stats_rows(_length_stats(chunked))}

- Chunks exceeding target: {len(exceptions)}
- Total intentional overlap characters: {total_overlap}
- Split boundaries: {boundary_counts}
- Unavoidable hard-limit splits: {hard_limit_details or 'None'}

## Representative sections

{chr(10).join(examples)}

## Integrity and provenance

- CFR normalized records contain usable paragraph identifiers such as `(a)`, `(1)`, `(i)`, and combined identifiers such as `(d)(1)`.
- CFR chunk ends prefer normalized paragraph boundaries. Oversized paragraphs fall back to sentence boundaries and only then to a hard character limit.
- FDA Guidance chunks prefer paragraph boundaries, then sentence boundaries, and never cross a Section boundary.
- Linked footnotes remain individual Documents and are never split.
- All original metadata and citations are retained. Split chunks receive deterministic IDs and keep their original ID in `parent_record_id`.
- Full parent bodies were reconstructed from chunk offsets, with overlaps checked for character equality. No source body text was lost, rewritten, or introduced in this experiment.

## Known limitations

- Chunk page ranges retain the parent Section page range because the normalized source does not provide exact page offsets for each paragraph.
- Boundary-aligned overlap may be smaller or larger than {chunk_overlap} characters.
- CFR paragraph ancestry is inferred only from identifiers present in the normalized XML. Missing identifiers are not invented.
- Contextual title, Section path, and CFR paragraph-context headers repeat across chunks; these headers are excluded from body reconstruction checks.
- Chunk size is character-based rather than tokenizer-specific.
- Two Guidance paragraphs (`III.A.1` chunk 0 and `V.14` chunk 0) contain no usable paragraph or sentence boundary near the target; those chunks use the documented hard-limit fallback.

## Recommendation

Keep Baseline A and Experiment B as separate, versioned retrieval inputs. Compare them later with the same queries, embedding model, filters, and retrieval metrics. This preparation analysis demonstrates structural and text integrity, not retrieval-quality improvement.
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    baseline = build_baseline_documents()
    chunked = build_chunked_documents()
    write_chunking_report(baseline, chunked)
    print(f"Baseline Documents: {len(baseline)}")
    print(f"Chunked Documents: {len(chunked)}")
    print(f"Chunked length stats: {_length_stats(chunked)}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
