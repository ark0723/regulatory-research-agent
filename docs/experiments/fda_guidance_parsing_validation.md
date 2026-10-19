# FDA guidance PDF parsing validation

## Result

- Source: `data/raw/FDA_informed_consent_2023.pdf`
- SHA-256: `96b26b2edb06e20ee2872ca59f4ec7c04cde8a8b1adf98bc150764e3d52f906a`
- FDA source URL: https://www.fda.gov/media/88915/download
- PDF pages: 66
- PDF bookmarks / JSON sections: 66 / 66
- Footnotes extracted / associated / unresolved: 94 / 93 / 1
- Directly read printed-page labels: 61
- Empty section records: III.E

`pdf_page_start` and `pdf_page_end` are one-based PDF page positions. `printed_page_start` and `printed_page_end` are read directly from page footers; no fixed offset is assumed.

## Automated validation

- PASS — 66 bookmarks produced 66 ordered section records
- PASS — section order and levels match the PDF bookmarks
- PASS — stable IDs are unique and empty parent III.E is preserved
- PASS — repeated headers and standalone printed pages are excluded
- PASS — page 29 footnotes 42/43 map to III.D and multiblock 44 maps to III.E.1
- PASS — page 29 footnote text is absent from all section bodies
- PASS — all 94 numbered footnotes were reconstructed in order
- PASS — every main-text line after the first bookmark is assigned once
- PASS — targeted PDF-page 11 and 15 text checks passed

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

- Footnote 1 (PDF page 6): `reference_not_detected`.

Footnote ownership is accepted only when the same numbered superscript occurs in exactly one bookmarked section. Footnote 1 is intentionally left unresolved because its superscript is in the document title, before the first bookmark; assigning it to section I would be a guess. The review sidecar preserves its text.

The automated coverage check operates at extracted PDF text-line granularity: it proves that every non-furniture, non-footnote text line after the first bookmark is assigned exactly once to a heading or section. It cannot prove that the PDF's embedded text layer itself is free of character-encoding defects. For manual audit, compare the target pages above with the matching `section_id` and footnotes in `data/processed/FDA_informed_consent_2023_sections.json`; use `data/processed/FDA_informed_consent_2023_footnotes_review.json` for unresolved attribution.
