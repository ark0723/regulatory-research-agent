"""Deterministic citation catalog, resolution, and output validation."""

from __future__ import annotations

import re
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from regulatory_agent.retrieval import Evidence, citation_label


class CitationSource(BaseModel):
    """One request-local citation target backed by supplied evidence."""

    evidence_id: str
    source_record_id: str
    evidence_record_ids: list[str]
    citation: str
    allowed_citations: list[str]
    document_type: str
    source_url: str = ""
    evidence_scope: str
    scope_complete: bool
    synthetic: bool = False


class CitationValidationIssue(BaseModel):
    """A deterministic citation failure safe to expose in graph state."""

    code: str
    message: str
    reference: str | None = None


class GeneratedAnswer(BaseModel):
    """Structured answer draft using request-local inline evidence markers."""

    answer: str = Field(
        description=(
            "Answer text with inline evidence markers such as [[E1]] after "
            "each evidence-grounded claim or list"
        )
    )
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs used in the answer, in first-appearance order",
    )


class CitationValidationResult(BaseModel):
    """Result of deterministic validation and citation rendering."""

    valid: bool
    rendered_answer: str = ""
    citations: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    issues: list[CitationValidationIssue] = Field(default_factory=list)


_VALID_MARKER_RE = re.compile(r"\[\[(E[1-9]\d*)\]\]")
_ANY_MARKER_RE = re.compile(r"\[\[([^\]\n]*)\]\]")
_PLACEHOLDER_RE = re.compile(r"【[^】]+】")
_CFR_REFERENCE_RE = re.compile(
    r"\b21\s*CFR\s*(?:§|section)?\s*"
    r"(\d+\.\d+(?:\([A-Za-z0-9ivxlcdmIVXLCDM]+\))*)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s)\]>]+")


def _ordered_parent_groups(
    evidence: Sequence[Evidence],
) -> list[tuple[str, list[Evidence]]]:
    groups: dict[str, list[Evidence]] = {}
    for item in evidence:
        groups.setdefault(item.parent_record_id, []).append(item)
    return list(groups.items())


def build_citation_catalog(evidence: Sequence[Evidence]) -> dict[str, CitationSource]:
    """Build stable request-local citation IDs in supplied-evidence order.

    Persistent ``source_record_id`` values identify the original parent source.
    ``evidence_record_ids`` identify the concrete supplied chunks and may include
    synthetic paragraph-scope IDs. The two identities are never interchanged.
    """

    catalog: dict[str, CitationSource] = {}
    for index, (parent_id, items) in enumerate(_ordered_parent_groups(evidence), 1):
        evidence_id = f"E{index}"
        first = items[0]
        metadata = first.document.metadata
        section_citation = citation_label(first.document)
        scopes = list(dict.fromkeys(item.evidence_scope for item in items))
        scope_complete = bool(items) and all(item.scope_complete for item in items)
        synthetic = any(
            bool(item.document.metadata.get("synthetic_evidence"))
            or item.evidence_scope.startswith("paragraph:")
            for item in items
        )

        allowed_citations = [section_citation] if section_citation else []
        resolved_citation = section_citation
        if (
            metadata.get("document_type") == "regulation"
            and scope_complete
            and len(scopes) == 1
            and scopes[0].startswith("paragraph:")
        ):
            paragraph = scopes[0].removeprefix("paragraph:")
            if re.fullmatch(r"\([a-z]\)", paragraph) and section_citation:
                precise = f"{section_citation}{paragraph}"
                allowed_citations.append(precise)
                resolved_citation = precise

        record_ids = list(
            dict.fromkeys(
                str(
                    item.document.metadata.get("record_id")
                    or item.document.metadata.get("chunk_id")
                    or ""
                )
                for item in items
            )
        )
        catalog[evidence_id] = CitationSource(
            evidence_id=evidence_id,
            source_record_id=str(
                metadata.get("source_record_id") or parent_id
            ),
            evidence_record_ids=[record_id for record_id in record_ids if record_id],
            citation=resolved_citation,
            allowed_citations=list(dict.fromkeys(allowed_citations)),
            document_type=str(metadata.get("document_type", "")),
            source_url=str(metadata.get("source_url", "")),
            evidence_scope=scopes[0] if len(scopes) == 1 else "mixed",
            scope_complete=scope_complete,
            synthetic=synthetic,
        )
    return catalog


def evidence_ids_by_parent(
    catalog: dict[str, CitationSource],
) -> dict[str, str]:
    """Return the parent/source-record to request-local evidence-ID mapping."""

    return {
        source.source_record_id: evidence_id
        for evidence_id, source in catalog.items()
    }


def render_citation_catalog(catalog: dict[str, CitationSource]) -> str:
    """Render only the fields an LLM needs to make valid references."""

    lines = []
    for source in catalog.values():
        lines.append(
            "- "
            f"{source.evidence_id}: display_citation={source.citation}; "
            f"allowed_citations={source.allowed_citations}; "
            f"scope={source.evidence_scope}; scope_complete={source.scope_complete}"
        )
    return "\n".join(lines) or "None"


def _normalize_citation(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def validate_and_render_answer(
    draft: GeneratedAnswer,
    catalog: dict[str, CitationSource],
) -> CitationValidationResult:
    """Resolve evidence markers or return explicit deterministic failures.

    This verifies citation syntax and provenance only. It does not decide whether
    arbitrary natural-language claims are semantically entailed by the evidence.
    """

    answer = draft.answer.strip()
    issues: list[CitationValidationIssue] = []
    if not answer:
        issues.append(
            CitationValidationIssue(
                code="empty_generated_answer",
                message="The generated answer is empty.",
            )
        )
    placeholders = _PLACEHOLDER_RE.findall(answer)
    for placeholder in placeholders:
        issues.append(
            CitationValidationIssue(
                code="unresolved_placeholder",
                message="Unresolved citation placeholders are not allowed.",
                reference=placeholder,
            )
        )

    raw_markers = _ANY_MARKER_RE.findall(answer)
    valid_markers = _VALID_MARKER_RE.findall(answer)
    for marker in raw_markers:
        if not re.fullmatch(r"E[1-9]\d*", marker):
            issues.append(
                CitationValidationIssue(
                    code="malformed_citation_reference",
                    message="Citation markers must use the form [[E1]].",
                    reference=marker,
                )
            )
    stripped_markers = _ANY_MARKER_RE.sub("", answer)
    if "[[" in stripped_markers or "]]" in stripped_markers:
        issues.append(
            CitationValidationIssue(
                code="malformed_citation_reference",
                message="An unterminated citation marker was found.",
            )
        )

    ordered_ids = list(dict.fromkeys(valid_markers))
    for evidence_id in ordered_ids:
        if evidence_id not in catalog:
            issues.append(
                CitationValidationIssue(
                    code="unknown_evidence_reference",
                    message="The answer references evidence that was not supplied.",
                    reference=evidence_id,
                )
            )

    if draft.cited_evidence_ids != ordered_ids:
        issues.append(
            CitationValidationIssue(
                code="citation_reference_mismatch",
                message=(
                    "cited_evidence_ids must match inline markers in "
                    "first-appearance order."
                ),
                reference=str(draft.cited_evidence_ids),
            )
        )
    if answer and not ordered_ids:
        issues.append(
            CitationValidationIssue(
                code="missing_citation_reference",
                message="A grounded answer must cite at least one supplied evidence ID.",
            )
        )

    known_ids = [evidence_id for evidence_id in ordered_ids if evidence_id in catalog]
    allowed_by_id = {
        evidence_id: {
            _normalize_citation(citation)
            for citation in catalog[evidence_id].allowed_citations
        }
        for evidence_id in known_ids
    }
    for match in _CFR_REFERENCE_RE.finditer(answer):
        reference = f"21 CFR § {match.group(1)}"
        normalized = _normalize_citation(reference)
        if not any(normalized in allowed for allowed in allowed_by_id.values()):
            code: Literal[
                "unsupported_paragraph_specificity", "unmapped_cfr_citation"
            ] = "unmapped_cfr_citation"
            section_reference = re.sub(r"(?:\([^)]+\))+$", "", match.group(1))
            if any(
                _normalize_citation(f"21 CFR § {section_reference}") in allowed
                for allowed in allowed_by_id.values()
            ):
                code = "unsupported_paragraph_specificity"
            issues.append(
                CitationValidationIssue(
                    code=code,
                    message=(
                        "The CFR reference is not supported at that specificity "
                        "by the cited evidence."
                    ),
                    reference=reference,
                )
            )

    allowed_urls = {
        catalog[evidence_id].source_url
        for evidence_id in known_ids
        if catalog[evidence_id].source_url
    }
    for url in _URL_RE.findall(answer):
        if url not in allowed_urls:
            issues.append(
                CitationValidationIssue(
                    code="unmapped_source_url",
                    message="The source URL is not present in cited evidence metadata.",
                    reference=url,
                )
            )

    if issues:
        return CitationValidationResult(
            valid=False,
            cited_evidence_ids=ordered_ids,
            issues=issues,
        )

    rendered = _VALID_MARKER_RE.sub(
        lambda match: f"[{catalog[match.group(1)].citation}]",
        answer,
    )
    citations = list(
        dict.fromkeys(catalog[evidence_id].citation for evidence_id in ordered_ids)
    )
    return CitationValidationResult(
        valid=True,
        rendered_answer=rendered,
        citations=citations,
        cited_evidence_ids=ordered_ids,
    )
