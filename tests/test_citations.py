"""Regression tests for deterministic citation provenance and rendering."""

import unittest

from langchain_core.documents import Document

from regulatory_agent.citations import (
    GeneratedAnswer,
    build_citation_catalog,
    validate_and_render_answer,
)
from regulatory_agent.retrieval import Evidence


INFORMED_CONSENT_ELEMENTS = [
    "연구라는 설명",
    "예측 가능한 위험 또는 불편",
    "합리적으로 예상되는 이익",
    "적절한 대체 절차 또는 치료",
    "기록의 비밀 유지 범위",
    "연구 관련 손상에 대한 보상 또는 치료",
    "연구 및 손상 관련 연락처",
    "자발적 참여와 불이익 없는 철회",
]


def regulation_evidence(*, paragraph_scoped: bool) -> Evidence:
    parent_id = "21-cfr-part50:section:50.25"
    if paragraph_scoped:
        record_id = f"{parent_id}:paragraph-scope:a"
        scope = "paragraph:(a)"
    else:
        record_id = parent_id
        scope = "parent_section"
    text = "\n".join(
        ["§ 50.25 Elements of informed consent.", "(a) Basic elements."]
        + [f"({index}) {element}" for index, element in enumerate(INFORMED_CONSENT_ELEMENTS, 1)]
    )
    document = Document(
        page_content=text,
        metadata={
            "record_id": record_id,
            "chunk_id": record_id,
            "parent_record_id": parent_id,
            "source_record_id": parent_id,
            "synthetic_evidence": paragraph_scoped,
            "document_type": "regulation",
            "section_id": "50.25",
            "paragraph_context": "(a)",
            "source_url": "https://www.ecfr.gov/current/title-21/section-50.25",
        },
    )
    return Evidence(
        document=document,
        score=0.99,
        retrieval_rank=1,
        parent_record_id=parent_id,
        section_complete=not paragraph_scoped,
        evidence_scope=scope,
        scope_complete=True,
    )


class CitationValidationTests(unittest.TestCase):
    def test_informed_consent_answer_preserves_all_eight_elements(self):
        catalog = build_citation_catalog([regulation_evidence(paragraph_scoped=True)])
        answer = "피험자 동의서의 기본 요소는 다음과 같습니다:\n" + "\n".join(
            f"{index}. {element}"
            for index, element in enumerate(INFORMED_CONSENT_ELEMENTS, 1)
        )
        draft = GeneratedAnswer(
            answer=f"{answer} [[E1]]",
            cited_evidence_ids=["E1"],
        )

        result = validate_and_render_answer(draft, catalog)

        self.assertTrue(result.valid)
        self.assertEqual(["21 CFR § 50.25(a)"], result.citations)
        for element in INFORMED_CONSENT_ELEMENTS:
            self.assertIn(element, result.rendered_answer)

    def test_verified_paragraph_scope_resolves_precise_citation(self):
        catalog = build_citation_catalog([regulation_evidence(paragraph_scoped=True)])

        result = validate_and_render_answer(
            GeneratedAnswer(
                answer="21 CFR § 50.25(a)의 기본 요소입니다. [[E1]]",
                cited_evidence_ids=["E1"],
            ),
            catalog,
        )

        self.assertTrue(result.valid)
        self.assertIn("[21 CFR § 50.25(a)]", result.rendered_answer)

    def test_rejects_unresolved_source_text_placeholder(self):
        catalog = build_citation_catalog([regulation_evidence(paragraph_scoped=True)])

        result = validate_and_render_answer(
            GeneratedAnswer(
                answer="필수 요소입니다. 【source_text】 [[E1]]",
                cited_evidence_ids=["E1"],
            ),
            catalog,
        )

        self.assertFalse(result.valid)
        self.assertIn("unresolved_placeholder", [issue.code for issue in result.issues])

    def test_rejects_unknown_or_fabricated_evidence_reference(self):
        catalog = build_citation_catalog([regulation_evidence(paragraph_scoped=True)])

        result = validate_and_render_answer(
            GeneratedAnswer(answer="근거 없는 주장입니다. [[E99]]", cited_evidence_ids=["E99"]),
            catalog,
        )

        self.assertFalse(result.valid)
        self.assertIn("unknown_evidence_reference", [issue.code for issue in result.issues])

    def test_synthetic_evidence_keeps_original_source_identity_separate(self):
        evidence = regulation_evidence(paragraph_scoped=True)

        source = build_citation_catalog([evidence])["E1"]

        self.assertTrue(source.synthetic)
        self.assertEqual("21-cfr-part50:section:50.25", source.source_record_id)
        self.assertEqual(
            ["21-cfr-part50:section:50.25:paragraph-scope:a"],
            source.evidence_record_ids,
        )
        self.assertNotEqual(source.source_record_id, source.evidence_record_ids[0])

    def test_section_evidence_cannot_claim_paragraph_specificity(self):
        catalog = build_citation_catalog([regulation_evidence(paragraph_scoped=False)])

        result = validate_and_render_answer(
            GeneratedAnswer(
                answer="21 CFR § 50.25(a)에 규정되어 있습니다. [[E1]]",
                cited_evidence_ids=["E1"],
            ),
            catalog,
        )

        self.assertFalse(result.valid)
        self.assertIn(
            "unsupported_paragraph_specificity",
            [issue.code for issue in result.issues],
        )

    def test_rejects_unrelated_cfr_claim_not_in_cited_evidence(self):
        catalog = build_citation_catalog([regulation_evidence(paragraph_scoped=True)])

        result = validate_and_render_answer(
            GeneratedAnswer(
                answer="21 CFR § 50.20은 이 절차의 법적 유효성을 보장합니다. [[E1]]",
                cited_evidence_ids=["E1"],
            ),
            catalog,
        )

        self.assertFalse(result.valid)
        self.assertIn("unmapped_cfr_citation", [issue.code for issue in result.issues])

    def test_rejects_malformed_marker(self):
        catalog = build_citation_catalog([regulation_evidence(paragraph_scoped=True)])

        result = validate_and_render_answer(
            GeneratedAnswer(answer="필수 요소입니다. [[source_text]]", cited_evidence_ids=[]),
            catalog,
        )

        self.assertFalse(result.valid)
        self.assertIn(
            "malformed_citation_reference",
            [issue.code for issue in result.issues],
        )


if __name__ == "__main__":
    unittest.main()
