"""Regression tests for the FDA guidance PDF normalizer."""

import json
import re
import unittest

import pymupdf

from regulatory_agent.ingestion.fda_guidance import (
    OUTPUT_PATH,
    PDF_PATH,
    REPEATED_HEADER,
    REVIEW_PATH,
    extract_document,
    validate_result,
)


class FdaGuidanceParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = extract_document()
        cls.sections = cls.result.sections
        cls.by_id = {section["section_id"]: section for section in cls.sections}

    def test_count_order_hierarchy_and_boundaries(self):
        self.assertEqual(66, len(self.sections))
        with pymupdf.open(PDF_PATH) as pdf:
            toc = pdf.get_toc(simple=False)
        self.assertEqual(
            [row[1] for row in toc],
            [section["bookmark_title"] for section in self.sections],
        )
        self.assertEqual(
            [row[0] for row in toc],
            [section["section_level"] for section in self.sections],
        )
        starts = [section["pdf_page_start"] for section in self.sections]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(self.sections), len({s["section_id"] for s in self.sections}))

    def test_empty_parent_and_clean_title(self):
        parent = self.by_id["III.E"]
        self.assertEqual("", parent["content"])
        self.assertEqual("Documentation of Informed Consent", parent["section_title"])
        self.assertEqual(
            "D. Element of Informed Consent for “Applicable Clinical Trials”",
            self.by_id["III.D"]["section_heading"],
        )

    def test_page_furniture_is_excluded(self):
        for section in self.sections:
            self.assertNotIn(REPEATED_HEADER, section["content"])
            self.assertIsNone(re.search(r"(?m)^\s*\d+\s*$", section["content"]))
        self.assertEqual("24", self.by_id["III.E.1"]["printed_page_start"])
        self.assertEqual(29, self.by_id["III.E.1"]["pdf_page_start"])

    def test_page_29_footnote_attribution_and_reconstruction(self):
        owners = {
            footnote["number"]: section["section_id"]
            for section in self.sections
            for footnote in section["footnotes"]
        }
        self.assertEqual("III.D", owners["42"])
        self.assertEqual("III.D", owners["43"])
        self.assertEqual("III.E.1", owners["44"])
        footnote_44 = next(
            item
            for item in self.by_id["III.E.1"]["footnotes"]
            if item["number"] == "44"
        )
        self.assertIn("waive the requirement", footnote_44["text"])
        self.assertTrue(footnote_44["text"].endswith("met.”"))
        combined_content = "\n".join(section["content"] for section in self.sections)
        self.assertNotIn(
            "For the definition of “applicable clinical trial,” see 42 USC",
            combined_content,
        )
        self.assertNotIn(
            "Note that this statement is not required for child assent.",
            combined_content,
        )
        self.assertNotIn(
            "An IRB shall require documentation of informed consent in accordance",
            combined_content,
        )
        for number in ("42", "43", "44"):
            self.assertFalse(
                any(
                    footnote["number"] == number
                    for section in self.sections
                    if section["section_id"] not in {"III.D", "III.E.1"}
                    for footnote in section["footnotes"]
                )
            )

    def test_all_footnotes_and_target_pages(self):
        self.assertEqual(list(range(1, 95)), [int(f.number) for f in self.result.footnotes])
        self.assertEqual(93, sum(len(s["footnotes"]) for s in self.sections))
        self.assertEqual(["1"], [f["number"] for f in self.result.unresolved_footnotes])
        self.assertTrue(
            {"16", "17", "18", "19"}.issubset(
                {f["number"] for f in self.by_id["III.A.1"]["footnotes"]}
            )
        )
        self.assertIn("21 CFR 50.25(a)", self.by_id["III.B"]["content"])

    def test_no_unassigned_or_duplicate_main_text(self):
        self.assertEqual(
            0, self.result.stats["uncategorized_main_lines_after_first_bookmark"]
        )
        self.assertTrue(validate_result(self.result))

    def test_written_outputs_match_fresh_extraction(self):
        output = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        self.assertEqual(self.sections, output)
        self.assertEqual(self.result.unresolved_footnotes, review["unresolved_footnotes"])


if __name__ == "__main__":
    unittest.main()
