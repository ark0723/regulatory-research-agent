"""Build LangChain Documents with shared CFR / FDA Guidance metadata.

The common metadata fields are deliberately scalar values so they can be used
directly in pgvector JSONB metadata filters. Source-specific fields are added
only where they are meaningful.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from regulatory_agent.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

GUIDANCE_PATH = PROCESSED_DATA_DIR / "FDA_informed_consent_2023_sections.json"
GUIDANCE_FOOTNOTE_REVIEW_PATH = (
    PROCESSED_DATA_DIR / "FDA_informed_consent_2023_footnotes_review.json"
)
CFR_PATH = PROCESSED_DATA_DIR / "21_cfr_part50.json"

METADATA_SCHEMA_VERSION = "1.0"
AUTHORITY = "FDA"
JURISDICTION = "US"

GUIDANCE_TITLE = (
    "FDA Informed Consent Guidance for IRBs, Clinical Investigators, "
    "and Sponsors (2023)"
)
CFR_TITLE = "21 CFR Part 50"

# Every section and footnote Document has these keys. Optional numeric values
# use None instead of sentinel numbers so range filters cannot match them by
# accident.
COMMON_METADATA_KEYS = frozenset(
    {
        "metadata_schema_version",
        "record_id",
        "document_id",
        "document_type",
        "content_type",
        "authority",
        "jurisdiction",
        "document_title",
        "citation",
        "section_id",
        "section_title",
        "section_level",
        "section_path",
        "parent_section_id",
        "page_start",
        "page_end",
        "source_format",
        "source_file",
        "source_url",
    }
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def _parent_section_id(section_id: str) -> str:
    return section_id.rsplit(".", 1)[0] if "." in section_id else ""


def _common_metadata(
    *,
    record_id: str,
    document_id: str,
    document_type: str,
    content_type: str,
    document_title: str,
    citation: str,
    section_id: str,
    section_title: str,
    section_level: int | None,
    section_path: str,
    parent_section_id: str,
    page_start: int | None,
    page_end: int | None,
    source_format: str,
    source_file: str,
    source_url: str,
) -> dict[str, str | int | None]:
    """Create the metadata contract shared by every retrieval document."""
    return {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "record_id": record_id,
        "document_id": document_id,
        "document_type": document_type,
        "content_type": content_type,
        "authority": AUTHORITY,
        "jurisdiction": JURISDICTION,
        "document_title": document_title,
        "citation": citation,
        "section_id": section_id,
        "section_title": section_title,
        "section_level": section_level,
        "section_path": section_path,
        "parent_section_id": parent_section_id,
        "page_start": page_start,
        "page_end": page_end,
        "source_format": source_format,
        "source_file": source_file,
        "source_url": source_url,
    }


def build_guidance_documents(
    records_path: Path = GUIDANCE_PATH,
    review_path: Path | None = GUIDANCE_FOOTNOTE_REVIEW_PATH,
) -> tuple[list[Document], list[Document]]:
    """Return ``(section_documents, footnote_documents)`` for the guidance.

    Empty parent sections are skipped for embedding. Associated footnotes and
    unresolved footnotes from the parser review sidecar become independent
    Documents; no footnote payload is copied into section metadata.
    """
    records_path = Path(records_path)
    records = _load_records(records_path)
    section_documents: list[Document] = []
    footnote_documents: list[Document] = []

    for record in records:
        document_id = record["document_id"]
        section_id = record["section_id"]
        section_title = record["section_title"]
        path_items = record["section_path"]
        section_path = " > ".join(path_items)
        page_start = record.get("pdf_page_start", record.get("page_start"))
        page_end = record.get("pdf_page_end", record.get("page_end"))
        source_file = record.get("source_file", records_path.name)
        source_format = record.get("source_format", "pdf")
        source_url = record.get("source_url", "")

        content = record.get("content", "").strip()
        if content:
            metadata = _common_metadata(
                record_id=f"{document_id}:section:{section_id}",
                document_id=document_id,
                document_type=record.get("document_type", "guidance"),
                content_type="section",
                document_title=GUIDANCE_TITLE,
                citation=f"FDA Informed Consent Guidance (2023), Section {section_id}",
                section_id=section_id,
                section_title=section_title,
                section_level=record.get("section_level"),
                section_path=section_path,
                parent_section_id=_parent_section_id(section_id),
                page_start=page_start,
                page_end=page_end,
                source_format=source_format,
                source_file=source_file,
                source_url=source_url,
            )
            metadata.update(
                {
                    "section_heading": record.get("section_heading", ""),
                    "printed_page_start": record.get("printed_page_start"),
                    "printed_page_end": record.get("printed_page_end"),
                }
            )
            section_documents.append(
                Document(
                    page_content=(
                        f"{GUIDANCE_TITLE}\n"
                        f"Section: {section_path}\n\n"
                        f"{content}"
                    ),
                    metadata=metadata,
                )
            )

        for footnote in record.get("footnotes", []):
            number = str(footnote["number"])
            text = footnote.get("text", "").strip()
            if not text:
                continue
            footnote_page = footnote.get("pdf_page", page_start)
            metadata = _common_metadata(
                record_id=f"{document_id}:footnote:{number}",
                document_id=document_id,
                document_type=record.get("document_type", "guidance"),
                content_type="footnote",
                document_title=GUIDANCE_TITLE,
                citation=(
                    f"FDA Informed Consent Guidance (2023), "
                    f"Section {section_id}, Footnote {number}"
                ),
                section_id=section_id,
                section_title=section_title,
                section_level=record.get("section_level"),
                section_path=section_path,
                parent_section_id=_parent_section_id(section_id),
                page_start=footnote_page,
                page_end=footnote_page,
                source_format=source_format,
                source_file=source_file,
                source_url=source_url,
            )
            metadata.update(
                {
                    "footnote_number": number,
                    "footnote_attribution": "associated",
                }
            )
            footnote_documents.append(
                Document(
                    page_content=(
                        f"{GUIDANCE_TITLE}\n"
                        f"Section: {section_path}\n"
                        f"Footnote {number}: {text}"
                    ),
                    metadata=metadata,
                )
            )

    if review_path is not None:
        review_path = Path(review_path)
        if review_path.exists():
            review = json.loads(review_path.read_text(encoding="utf-8"))
            source_file = review.get("source_file", records_path.name)
            source_url = records[0].get("source_url", "") if records else ""
            for footnote in review.get("unresolved_footnotes", []):
                number = str(footnote["number"])
                text = footnote.get("text", "").strip()
                if not text:
                    continue
                footnote_page = footnote.get("pdf_page")
                metadata = _common_metadata(
                    record_id=f"{review['document_id']}:footnote:{number}",
                    document_id=review["document_id"],
                    document_type="guidance",
                    content_type="footnote",
                    document_title=GUIDANCE_TITLE,
                    citation=f"FDA Informed Consent Guidance (2023), Footnote {number}",
                    section_id="",
                    section_title="",
                    section_level=None,
                    section_path="",
                    parent_section_id="",
                    page_start=footnote_page,
                    page_end=footnote_page,
                    source_format="pdf",
                    source_file=source_file,
                    source_url=source_url,
                )
                metadata.update(
                    {
                        "footnote_number": number,
                        "footnote_attribution": "unresolved",
                        "footnote_review_reason": footnote.get("reason", ""),
                    }
                )
                footnote_documents.append(
                    Document(
                        page_content=(
                            f"{GUIDANCE_TITLE}\n"
                            f"Footnote {number}: {text}"
                        ),
                        metadata=metadata,
                    )
                )

    _ensure_unique_record_ids(section_documents + footnote_documents)
    return section_documents, footnote_documents


def build_cfr_documents(records_path: Path = CFR_PATH) -> list[Document]:
    """Create CFR section Documents using the same metadata contract."""
    records_path = Path(records_path)
    records = _load_records(records_path)
    documents: list[Document] = []

    for record in records:
        content = record.get("content", "").strip()
        if not content:
            continue

        document_id = record["document_id"]
        section_id = record["section"]
        source_section_title = record["section_title"]
        section_title = re.sub(
            rf"^§\s*{re.escape(section_id)}\s*", "", source_section_title
        ).strip()
        subpart = record.get("subpart", "")
        section_label = record.get("section_label", f"§ {section_id}")
        path_items = [
            f"Title {record['title']}",
            f"Part {record['part']}",
            record.get("subpart_title", f"Subpart {subpart}"),
            f"{section_label} {section_title}",
        ]
        section_path = " > ".join(item for item in path_items if item)
        source_file = record.get(
            "source_file", (RAW_DATA_DIR / "21_cfr_part50.xml").name
        )

        metadata = _common_metadata(
            record_id=f"{document_id}:section:{section_id}",
            document_id=document_id,
            document_type=record.get("document_type", "regulation"),
            content_type="section",
            document_title=CFR_TITLE,
            citation=f"21 CFR {section_id}",
            section_id=section_id,
            section_title=section_title,
            section_level=4,
            section_path=section_path,
            parent_section_id=f"subpart:{subpart}",
            page_start=None,
            page_end=None,
            source_format=record.get("source_format", "ecfr_xml"),
            source_file=source_file,
            source_url=record.get("source_url", ""),
        )
        metadata.update(
            {
                "cfr_title": record["title"],
                "cfr_part": record["part"],
                "cfr_subpart": subpart,
                "cfr_section": section_id,
                "ecfr_node": record.get("node", ""),
            }
        )
        documents.append(
            Document(
                page_content=(
                    f"{CFR_TITLE}\n"
                    f"Section: {section_label} — {section_title}\n"
                    f"Hierarchy: {section_path}\n\n"
                    f"{content}"
                ),
                metadata=metadata,
            )
        )

    _ensure_unique_record_ids(documents)
    return documents


def _ensure_unique_record_ids(documents: list[Document]) -> None:
    record_ids = [document.metadata["record_id"] for document in documents]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate record_id values detected")


def build_all_documents() -> list[Document]:
    """Build one collection ready for chunking, embedding, or pgvector."""
    guidance_sections, guidance_footnotes = build_guidance_documents()
    documents = build_cfr_documents() + guidance_sections + guidance_footnotes
    _ensure_unique_record_ids(documents)
    return documents


def main() -> None:
    guidance_sections, guidance_footnotes = build_guidance_documents()
    cfr_documents = build_cfr_documents()
    print(f"CFR section Documents: {len(cfr_documents)}")
    print(f"FDA Guidance section Documents: {len(guidance_sections)}")
    print(f"FDA Guidance footnote Documents: {len(guidance_footnotes)}")
    print(f"Total Documents: {len(cfr_documents + guidance_sections + guidance_footnotes)}")
    print("\nSample metadata:")
    print(json.dumps(guidance_sections[0].metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
