"""Validation for normalized CFR records."""

import json
import unittest

from regulatory_agent.paths import PROCESSED_DATA_DIR


class CfrNormalizeTests(unittest.TestCase):
    def test_normalized_records(self):
        records = json.loads(
            (PROCESSED_DATA_DIR / "21_cfr_part50.json").read_text(encoding="utf-8")
        )
        self.assertEqual(15, len(records))

        section_numbers = [record["section"] for record in records]
        self.assertEqual(len(section_numbers), len(set(section_numbers)))
        self.assertIn("50.20", section_numbers)
        self.assertIn("50.25", section_numbers)

        for record in records:
            self.assertTrue(record["section_title"])
            self.assertTrue(record["content"])
            self.assertTrue(record["paragraphs"])
            self.assertEqual("regulation", record["document_type"])
            reconstructed = "\n\n".join(
                paragraph["text"] for paragraph in record["paragraphs"]
            )
            self.assertEqual(reconstructed, record["content"])


if __name__ == "__main__":
    unittest.main()
