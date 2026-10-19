"""Normalize the FDA informed-consent guidance PDF into section-level JSON.

Bookmark coordinates define section boundaries. Span-level layout identifies
page furniture, superscript references, and footnotes.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pymupdf

from regulatory_agent.paths import DOCS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

PDF_PATH = RAW_DATA_DIR / "FDA_informed_consent_2023.pdf"
OUTPUT_PATH = PROCESSED_DATA_DIR / "FDA_informed_consent_2023_sections.json"
REVIEW_PATH = PROCESSED_DATA_DIR / "FDA_informed_consent_2023_footnotes_review.json"
REPORT_PATH = DOCS_DIR / "experiments/fda_guidance_parsing_validation.md"

DOCUMENT_ID = "fda-informed-consent-2023"
SOURCE_URL = "https://www.fda.gov/media/88915/download"
REPEATED_HEADER = "Contains Nonbinding Recommendations"

# Footnote 4 begins at y=432.9, the highest footnote region in this PDF.
FOOTNOTE_START_MIN_Y = 420.0
FOOTNOTE_MARKER_MAX_SIZE = 7.2
SUPERSCRIPT_MAX_SIZE = 8.0
FOOTER_MIN_Y = 690.0


@dataclass
class TextLine:
    page_index: int
    order: int
    block_index: int
    line_index: int
    text: str
    bbox: tuple[float, float, float, float]
    spans: list[dict[str, Any]]
    category: str = "main"

    @property
    def position(self) -> tuple[int, int]:
        return self.page_index, self.order

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)

    def text_without_superscripts(self) -> str:
        return normalize_text(
            "".join(
                span["text"]
                for span in self.spans
                if span["size"] > SUPERSCRIPT_MAX_SIZE
            )
        )

    def superscript_numbers(self) -> list[str]:
        numbers: list[str] = []
        for span in self.spans:
            if span["size"] <= SUPERSCRIPT_MAX_SIZE:
                numbers.extend(re.findall(r"\d+", span["text"]))
        return numbers


@dataclass
class Bookmark:
    index: int
    level: int
    title: str
    page_index: int
    target_y: float
    heading_lines: list[TextLine] = field(default_factory=list)
    clean_heading: str = ""
    section_id: str = ""

    @property
    def start_position(self) -> tuple[int, int]:
        return self.heading_lines[0].position

    @property
    def end_position(self) -> tuple[int, int]:
        return self.heading_lines[-1].position


@dataclass
class Footnote:
    number: str
    text: str
    pdf_page: int


@dataclass
class ExtractionResult:
    sections: list[dict[str, Any]]
    unresolved_footnotes: list[dict[str, Any]]
    footnotes: list[Footnote]
    bookmarks: list[Bookmark]
    printed_page_numbers: dict[int, str]
    source_sha256: str
    stats: dict[str, Any]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_heading(text: str, remove_footnote: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = normalize_text(text)
    if remove_footnote:
        # Bookmarks contain Word artifacts such as "40F" while the rendered
        # heading contains a different superscript number.
        text = re.sub(r"\d+[a-z]?\s*(?=\??\s*$)", "", text).strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return normalize_text(text)


def _line_text(line: dict[str, Any]) -> str:
    return "".join(span["text"] for span in line["spans"])


def _is_printed_page_number(line: TextLine) -> bool:
    return bool(
        line.bbox[1] >= FOOTER_MIN_Y and re.fullmatch(r"\d+", line.normalized)
    )


def _is_footnote_start(line: TextLine) -> bool:
    nonempty = [span for span in line.spans if span["text"].strip()]
    if not nonempty or line.bbox[1] < FOOTNOTE_START_MIN_Y:
        return False
    first = nonempty[0]
    return bool(
        first["size"] <= FOOTNOTE_MARKER_MAX_SIZE
        and 65.0 <= first["bbox"][0] <= 85.0
        and re.fullmatch(r"\d+\s*", first["text"])
    )


def extract_page_lines(
    pdf: pymupdf.Document,
) -> tuple[dict[int, list[TextLine]], dict[int, str]]:
    pages: dict[int, list[TextLine]] = {}
    printed_pages: dict[int, str] = {}

    for page_index, page in enumerate(pdf):
        lines: list[TextLine] = []
        order = 0
        for block_index, block in enumerate(page.get_text("dict", sort=True)["blocks"]):
            if block["type"] != 0:
                continue
            for line_index, raw_line in enumerate(block["lines"]):
                text = _line_text(raw_line)
                if not text.strip():
                    continue
                line = TextLine(
                    page_index=page_index,
                    order=order,
                    block_index=block_index,
                    line_index=line_index,
                    text=text,
                    bbox=tuple(raw_line["bbox"]),
                    spans=raw_line["spans"],
                )
                order += 1
                if line.normalized == REPEATED_HEADER:
                    line.category = "header"
                elif _is_printed_page_number(line):
                    line.category = "page_number"
                    printed_pages[page_index + 1] = line.normalized
                lines.append(line)

        footnote_region = False
        for line in lines:
            if line.category in {"header", "page_number"}:
                continue
            if _is_footnote_start(line):
                footnote_region = True
            if footnote_region:
                line.category = "footnote"

        pages[page_index] = lines

    return pages, printed_pages


def _candidate_heading_text(lines: list[TextLine], without_superscripts: bool) -> str:
    parts = [
        line.text_without_superscripts() if without_superscripts else line.normalized
        for line in lines
    ]
    return normalize_text(" ".join(part for part in parts if part))


def match_heading(
    page_lines: list[TextLine], title: str, target_y: float
) -> list[TextLine]:
    eligible = [line for line in page_lines if line.category == "main"]
    candidates: list[tuple[float, float, list[TextLine], str]] = []

    for start in range(len(eligible)):
        for length in range(1, 5):
            group = eligible[start : start + length]
            if len(group) != length:
                continue
            if len({line.block_index for line in group}) != 1:
                continue
            y0 = min(line.bbox[1] for line in group)
            y1 = max(line.bbox[3] for line in group)
            distance = max(y0 - target_y, target_y - y1, 0.0)
            if distance > 24.0:
                continue

            rendered = _candidate_heading_text(group, without_superscripts=False)
            clean_rendered = _candidate_heading_text(group, without_superscripts=True)
            similarities = (
                SequenceMatcher(
                    None, normalize_heading(title), normalize_heading(rendered)
                ).ratio(),
                SequenceMatcher(
                    None,
                    normalize_heading(title, remove_footnote=True),
                    normalize_heading(rendered, remove_footnote=True),
                ).ratio(),
                SequenceMatcher(
                    None,
                    normalize_heading(title, remove_footnote=True),
                    normalize_heading(clean_rendered, remove_footnote=True),
                ).ratio(),
            )
            candidates.append((max(similarities), distance, group, clean_rendered))

    if not candidates:
        raise ValueError(f"No nearby heading text for bookmark: {title}")

    candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    similarity, _, best_lines, rendered = candidates[0]
    if similarity < 0.85:
        raise ValueError(
            f"Low-confidence heading match for {title!r}: "
            f"{rendered!r} ({similarity:.3f})"
        )
    return best_lines


def extract_bookmarks(
    pdf: pymupdf.Document, pages: dict[int, list[TextLine]]
) -> list[Bookmark]:
    bookmarks: list[Bookmark] = []
    hierarchy_ids: list[str] = []

    for index, (level, title, page_number, destination) in enumerate(
        pdf.get_toc(simple=False)
    ):
        point = destination.get("to")
        if point is None:
            raise ValueError(f"Missing destination for bookmark: {title}")
        page_index = page_number - 1
        heading_lines = match_heading(pages[page_index], title, point.y)
        clean_heading = _candidate_heading_text(
            heading_lines, without_superscripts=True
        )
        label_match = re.match(r"^([IVXLCDM]+|[A-Za-z]|\d+)\.\s*", clean_heading)
        if not label_match:
            raise ValueError(f"Cannot derive section ID from heading: {clean_heading}")
        hierarchy_ids = hierarchy_ids[: level - 1]
        hierarchy_ids.append(label_match.group(1))
        bookmarks.append(
            Bookmark(
                index=index,
                level=level,
                title=title,
                page_index=page_index,
                target_y=point.y,
                heading_lines=heading_lines,
                clean_heading=clean_heading,
                section_id=".".join(hierarchy_ids),
            )
        )

    positions = [bookmark.start_position for bookmark in bookmarks]
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ValueError("Bookmark heading positions are not strictly ordered")
    return bookmarks


def _join_wrapped_lines(lines: list[str]) -> str:
    result = ""
    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line:
            continue
        if not result:
            result = line
        elif result.endswith("-"):
            result += line
        else:
            result += " " + line
    return result


def extract_footnotes(pages: dict[int, list[TextLine]]) -> list[Footnote]:
    footnotes: list[Footnote] = []
    for page_index, lines in pages.items():
        current_number: str | None = None
        current_lines: list[str] = []

        def finish_current() -> None:
            nonlocal current_number, current_lines
            if current_number is not None:
                footnotes.append(
                    Footnote(
                        number=current_number,
                        text=_join_wrapped_lines(current_lines),
                        pdf_page=page_index + 1,
                    )
                )
            current_number = None
            current_lines = []

        for line in lines:
            if line.category != "footnote":
                continue
            if _is_footnote_start(line):
                finish_current()
                first_span = next(
                    span for span in line.spans if span["text"].strip()
                )
                current_number = first_span["text"].strip()
                marker_end = first_span["bbox"][2]
                remainder = "".join(
                    span["text"]
                    for span in line.spans
                    if span["bbox"][0] >= marker_end - 0.1 and span is not first_span
                )
                current_lines.append(remainder)
            elif current_number is not None:
                current_lines.append(line.text)
        finish_current()

    numbers = [int(footnote.number) for footnote in footnotes]
    expected = list(range(1, max(numbers, default=0) + 1))
    if numbers != expected:
        raise ValueError(f"Footnote sequence is incomplete or unordered: {numbers}")
    return footnotes


def _all_main_lines(pages: dict[int, list[TextLine]]) -> list[TextLine]:
    return [
        line
        for page_index in sorted(pages)
        for line in pages[page_index]
        if line.category == "main"
    ]


def _lines_between(
    lines: list[TextLine],
    after: tuple[int, int],
    before: tuple[int, int] | None,
) -> list[TextLine]:
    return [
        line
        for line in lines
        if line.position > after and (before is None or line.position < before)
    ]


def _render_content(lines: list[TextLine]) -> str:
    if not lines:
        return ""

    fragments: list[dict[str, Any]] = []
    for line in lines:
        key = (line.page_index, line.block_index)
        if not fragments or fragments[-1]["key"] != key:
            fragments.append(
                {
                    "key": key,
                    "page_index": line.page_index,
                    "y0": line.bbox[1],
                    "lines": [line.text],
                }
            )
        else:
            fragments[-1]["lines"].append(line.text)

    rendered = ""
    previous_page: int | None = None
    for fragment in fragments:
        text = _join_wrapped_lines(fragment["lines"])
        if not text:
            continue
        page_continuation = (
            previous_page is not None
            and fragment["page_index"] != previous_page
            and fragment["y0"] < 125.0
        )
        if not rendered:
            rendered = text
        elif re.fullmatch(r"\d+(?:\s*,\s*\d+)*", text):
            # A superscript can be emitted as its own block at a page break
            # (footnote reference 69 is the observed case).
            rendered += text
        elif page_continuation:
            rendered += text if rendered.endswith("-") else " " + text
        else:
            rendered += "\n\n" + text
        previous_page = fragment["page_index"]
    return rendered


def _section_title(clean_heading: str) -> str:
    return re.sub(r"^([IVXLCDM]+|[A-Za-z]|\d+)\.\s*", "", clean_heading).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_document(pdf_path: Path = PDF_PATH) -> ExtractionResult:
    pdf_path = Path(pdf_path)
    with pymupdf.open(pdf_path) as pdf:
        pages, printed_pages = extract_page_lines(pdf)
        bookmarks = extract_bookmarks(pdf, pages)
        footnotes = extract_footnotes(pages)
        main_lines = _all_main_lines(pages)

        heading_owner: dict[tuple[int, int], int] = {}
        for bookmark in bookmarks:
            for line in bookmark.heading_lines:
                heading_owner[line.position] = bookmark.index

        section_lines: list[list[TextLine]] = []
        line_owner: dict[tuple[int, int], int] = {}
        for index, bookmark in enumerate(bookmarks):
            next_start = (
                bookmarks[index + 1].start_position
                if index + 1 < len(bookmarks)
                else None
            )
            content_lines = _lines_between(main_lines, bookmark.end_position, next_start)
            content_lines = [
                line for line in content_lines if line.position not in heading_owner
            ]
            section_lines.append(content_lines)
            for line in content_lines:
                if line.position in line_owner:
                    raise ValueError(f"Text line assigned twice: {line.position}")
                line_owner[line.position] = index

        reference_owners: dict[str, set[int]] = {}
        unowned_reference_numbers: set[str] = set()
        for line in main_lines:
            owner = heading_owner.get(line.position, line_owner.get(line.position))
            for number in line.superscript_numbers():
                if owner is not None:
                    reference_owners.setdefault(number, set()).add(owner)
                else:
                    unowned_reference_numbers.add(number)

        associated: dict[int, list[Footnote]] = {
            index: [] for index in range(len(bookmarks))
        }
        unresolved: list[dict[str, Any]] = []
        for footnote in footnotes:
            owners = reference_owners.get(footnote.number, set())
            if len(owners) == 1:
                associated[next(iter(owners))].append(footnote)
            else:
                if footnote.number in unowned_reference_numbers:
                    reason = "reference_before_first_bookmark"
                elif not owners:
                    reason = "reference_not_detected"
                else:
                    reason = "reference_occurs_in_multiple_sections"
                unresolved.append(
                    {
                        "number": footnote.number,
                        "text": footnote.text,
                        "pdf_page": footnote.pdf_page,
                        "reason": reason,
                        "candidate_section_ids": [
                            bookmarks[owner].section_id for owner in sorted(owners)
                        ],
                    }
                )

        hierarchy: list[str] = []
        sections: list[dict[str, Any]] = []
        for index, bookmark in enumerate(bookmarks):
            hierarchy = hierarchy[: bookmark.level - 1]
            hierarchy.append(bookmark.clean_heading)
            end_page = (
                bookmarks[index + 1].page_index + 1
                if index + 1 < len(bookmarks)
                else len(pdf)
            )
            sections.append(
                {
                    "document_id": DOCUMENT_ID,
                    "document_type": "guidance",
                    "section_id": bookmark.section_id,
                    "section_title": _section_title(bookmark.clean_heading),
                    "section_heading": bookmark.clean_heading,
                    "section_level": bookmark.level,
                    "section_path": hierarchy.copy(),
                    "bookmark_title": bookmark.title,
                    "pdf_page_start": bookmark.page_index + 1,
                    "pdf_page_end": end_page,
                    "printed_page_start": printed_pages.get(bookmark.page_index + 1),
                    "printed_page_end": printed_pages.get(end_page),
                    "content": _render_content(section_lines[index]),
                    "footnotes": [
                        {
                            "number": footnote.number,
                            "text": footnote.text,
                            "pdf_page": footnote.pdf_page,
                        }
                        for footnote in associated[index]
                    ],
                    "source_format": "pdf",
                    "source_file": pdf_path.name,
                    "source_url": SOURCE_URL,
                }
            )

        categorized = set(heading_owner) | set(line_owner)
        first_heading = bookmarks[0].start_position
        uncategorized = [
            line.position
            for line in main_lines
            if line.position >= first_heading and line.position not in categorized
        ]
        if uncategorized:
            raise ValueError(
                "Main text was not assigned to a heading or section: "
                f"{uncategorized[:10]}"
            )

        stats = {
            "pdf_pages": len(pdf),
            "bookmarks": len(bookmarks),
            "sections": len(sections),
            "footnotes_extracted": len(footnotes),
            "footnotes_associated": sum(len(section["footnotes"]) for section in sections),
            "footnotes_unresolved": len(unresolved),
            "empty_sections": [
                section["section_id"] for section in sections if not section["content"]
            ],
            "printed_pages_directly_extracted": len(printed_pages),
            "uncategorized_main_lines_after_first_bookmark": len(uncategorized),
        }

    return ExtractionResult(
        sections=sections,
        unresolved_footnotes=unresolved,
        footnotes=footnotes,
        bookmarks=bookmarks,
        printed_page_numbers=printed_pages,
        source_sha256=_sha256(pdf_path),
        stats=stats,
    )


def extract_sections(pdf_path: Path = PDF_PATH) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only section records."""
    return extract_document(pdf_path).sections


def validate_result(result: ExtractionResult) -> list[str]:
    checks: list[str] = []
    sections = result.sections
    if len(sections) != 66 or len(result.bookmarks) != 66:
        raise ValueError("Expected exactly 66 bookmarks and section records")
    checks.append("PASS — 66 bookmarks produced 66 ordered section records")

    if [section["bookmark_title"] for section in sections] != [
        bookmark.title for bookmark in result.bookmarks
    ]:
        raise ValueError("Section order does not match bookmark order")
    checks.append("PASS — section order and levels match the PDF bookmarks")

    ids = [section["section_id"] for section in sections]
    if len(ids) != len(set(ids)):
        raise ValueError("Section IDs are not unique")
    parent = next(section for section in sections if section["section_id"] == "III.E")
    if parent["content"]:
        raise ValueError("III.E should be the one legitimate empty parent section")
    checks.append("PASS — stable IDs are unique and empty parent III.E is preserved")

    for section in sections:
        if REPEATED_HEADER in section["content"]:
            raise ValueError(f"Repeated header remains in {section['section_id']}")
        if re.search(r"(?m)^\s*\d+\s*$", section["content"]):
            raise ValueError(
                f"Standalone page-number-like line remains in {section['section_id']}"
            )
    checks.append("PASS — repeated headers and standalone printed pages are excluded")

    by_id = {section["section_id"]: section for section in sections}
    actual_owners = {
        footnote["number"]: section["section_id"]
        for section in sections
        for footnote in section["footnotes"]
    }
    for number, section_id in {"42": "III.D", "43": "III.D", "44": "III.E.1"}.items():
        if actual_owners.get(number) != section_id:
            raise ValueError(
                f"Footnote {number} owner is {actual_owners.get(number)}, expected {section_id}"
            )
    footnote_44 = next(
        footnote
        for footnote in by_id["III.E.1"]["footnotes"]
        if footnote["number"] == "44"
    )
    if "waive the requirement" not in footnote_44["text"] or not footnote_44["text"].endswith("met.”"):
        raise ValueError("Multiblock footnote 44 was not fully reconstructed")
    checks.append(
        "PASS — page 29 footnotes 42/43 map to III.D and multiblock 44 maps to III.E.1"
    )
    combined_content = "\n".join(section["content"] for section in sections)
    for leaked_text in (
        "For the definition of “applicable clinical trial,” see 42 USC",
        "Note that this statement is not required for child assent.",
        "An IRB shall require documentation of informed consent in accordance",
    ):
        if leaked_text in combined_content:
            raise ValueError("Page 29 footnote text leaked into section content")
    checks.append("PASS — page 29 footnote text is absent from all section bodies")

    if [int(item.number) for item in result.footnotes] != list(range(1, 95)):
        raise ValueError("Expected the complete numbered footnote sequence 1–94")
    checks.append("PASS — all 94 numbered footnotes were reconstructed in order")

    if result.stats["uncategorized_main_lines_after_first_bookmark"]:
        raise ValueError("Some main text was silently unassigned")
    checks.append("PASS — every main-text line after the first bookmark is assigned once")

    if "21 CFR 50.25(a)" not in by_id["III.B"]["content"]:
        raise ValueError("Target citation on PDF page 15 was not preserved")
    if not {"16", "17", "18", "19"}.issubset(
        {item["number"] for item in by_id["III.A.1"]["footnotes"]}
    ):
        raise ValueError("Target footnotes on PDF page 11 were not preserved")
    checks.append("PASS — targeted PDF-page 11 and 15 text checks passed")
    return checks


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_outputs(result: ExtractionResult, checks: list[str]) -> None:
    _write_json(OUTPUT_PATH, result.sections)
    _write_json(
        REVIEW_PATH,
        {
            "document_id": DOCUMENT_ID,
            "source_file": PDF_PATH.name,
            "source_sha256": result.source_sha256,
            "unresolved_footnotes": result.unresolved_footnotes,
        },
    )

    unresolved_rows = "\n".join(
        f"- Footnote {item['number']} (PDF page {item['pdf_page']}): `{item['reason']}`."
        for item in result.unresolved_footnotes
    ) or "- None."
    empty_sections = ", ".join(result.stats["empty_sections"]) or "None"
    check_rows = "\n".join(f"- {check}" for check in checks)
    report = f"""# FDA guidance PDF parsing validation

## Result

- Source: `{PDF_PATH}`
- SHA-256: `{result.source_sha256}`
- FDA source URL: {SOURCE_URL}
- PDF pages: {result.stats['pdf_pages']}
- PDF bookmarks / JSON sections: {result.stats['bookmarks']} / {result.stats['sections']}
- Footnotes extracted / associated / unresolved: {result.stats['footnotes_extracted']} / {result.stats['footnotes_associated']} / {result.stats['footnotes_unresolved']}
- Directly read printed-page labels: {result.stats['printed_pages_directly_extracted']}
- Empty section records: {empty_sections}

`pdf_page_start` and `pdf_page_end` are one-based PDF page positions. `printed_page_start` and `printed_page_end` are read directly from page footers; no fixed offset is assumed.

## Automated validation

{check_rows}

Run the reproducible checks with:

```bash
UV_CACHE_DIR=/private/tmp/regulatory-agent-uv-cache uv run normalize-fda-guidance
UV_CACHE_DIR=/private/tmp/regulatory-agent-uv-cache uv run python -m unittest discover -s tests -p 'test_*.py'
```

## Targeted review

- PDF page 11: footnotes 16–19 are associated with III.A.1, before III.A.2 begins on the same page.
- PDF page 15: the section III.B / III.B.1 boundary and `21 CFR 50.25(a)` citation are preserved.
- PDF page 29: footnotes 42 and 43 are associated with III.D; footnote 44 is reconstructed across three source text blocks and associated with III.E.1.

## Unresolved items and limitations

{unresolved_rows}

Footnote ownership is accepted only when the same numbered superscript occurs in exactly one bookmarked section. Footnote 1 is intentionally left unresolved because its superscript is in the document title, before the first bookmark; assigning it to section I would be a guess. The review sidecar preserves its text.

The automated coverage check operates at extracted PDF text-line granularity: it proves that every non-furniture, non-footnote text line after the first bookmark is assigned exactly once to a heading or section. It cannot prove that the PDF's embedded text layer itself is free of character-encoding defects. For manual audit, compare the target pages above with the matching `section_id` and footnotes in `{OUTPUT_PATH}`; use `{REVIEW_PATH}` for unresolved attribution.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    result = extract_document()
    checks = validate_result(result)
    write_outputs(result, checks)
    print(
        f"Extracted {len(result.sections)} sections and {len(result.footnotes)} "
        f"footnotes ({len(result.unresolved_footnotes)} unresolved)."
    )
    for check in checks:
        print(check)


if __name__ == "__main__":
    main()
