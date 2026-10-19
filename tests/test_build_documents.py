"""Tests for shared LangChain Document metadata."""

import unittest

from regulatory_agent.documents import (
    COMMON_METADATA_KEYS,
    build_all_documents,
    build_cfr_documents,
    build_guidance_documents,
)


class BuildDocumentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guidance_sections, cls.guidance_footnotes = build_guidance_documents()
        cls.cfr_documents = build_cfr_documents()
        cls.documents = build_all_documents()

    def test_expected_document_counts(self):
        self.assertEqual(65, len(self.guidance_sections))
        self.assertEqual(94, len(self.guidance_footnotes))
        self.assertEqual(15, len(self.cfr_documents))
        self.assertEqual(174, len(self.documents))

    def test_common_metadata_contract(self):
        scalar_types = (str, int, float, bool, type(None))
        for document in self.documents:
            self.assertTrue(COMMON_METADATA_KEYS.issubset(document.metadata))
            for value in document.metadata.values():
                self.assertIsInstance(value, scalar_types)

    def test_record_ids_are_stable_and_unique(self):
        record_ids = [document.metadata["record_id"] for document in self.documents]
        self.assertEqual(len(record_ids), len(set(record_ids)))
        self.assertIn("21-cfr-part50:section:50.25", record_ids)
        self.assertIn("fda-informed-consent-2023:section:III.E.1", record_ids)
        self.assertIn("fda-informed-consent-2023:footnote:44", record_ids)

    def test_guidance_footnotes_are_separate_documents(self):
        for document in self.guidance_sections:
            self.assertEqual("section", document.metadata["content_type"])
            self.assertNotIn("footnotes", document.metadata)
            self.assertNotIn("footnote_number", document.metadata)

        footnote_44 = next(
            document
            for document in self.guidance_footnotes
            if document.metadata["footnote_number"] == "44"
        )
        self.assertEqual("footnote", footnote_44.metadata["content_type"])
        self.assertEqual("III.E.1", footnote_44.metadata["section_id"])
        self.assertEqual("associated", footnote_44.metadata["footnote_attribution"])
        self.assertIn("waive the requirement", footnote_44.page_content)

    def test_empty_parent_is_not_embedded(self):
        section_ids = {document.metadata["section_id"] for document in self.guidance_sections}
        self.assertNotIn("III.E", section_ids)
        self.assertIn("III.E.1", section_ids)

    def test_unresolved_footnote_is_preserved(self):
        footnote_1 = next(
            document
            for document in self.guidance_footnotes
            if document.metadata["footnote_number"] == "1"
        )
        self.assertEqual("unresolved", footnote_1.metadata["footnote_attribution"])
        self.assertEqual("", footnote_1.metadata["section_id"])
        self.assertEqual(6, footnote_1.metadata["page_start"])

    def test_cfr_and_guidance_support_common_filters(self):
        guidance = self.guidance_sections[0].metadata
        cfr = self.cfr_documents[0].metadata
        self.assertEqual("FDA", guidance["authority"])
        self.assertEqual("FDA", cfr["authority"])
        self.assertEqual("guidance", guidance["document_type"])
        self.assertEqual("regulation", cfr["document_type"])
        self.assertEqual("section", guidance["content_type"])
        self.assertEqual("section", cfr["content_type"])
        self.assertEqual("US", guidance["jurisdiction"])
        self.assertEqual("US", cfr["jurisdiction"])
        self.assertEqual("Scope.", cfr["section_title"])
        self.assertEqual("21 CFR 50.1", cfr["citation"])


if __name__ == "__main__":
    unittest.main()
