"""Validation for the extracted CFR Part 50 XML."""

import unittest
import xml.etree.ElementTree as ET

from regulatory_agent.paths import RAW_DATA_DIR


class CfrExtractTests(unittest.TestCase):
    def test_part_50_sections_are_preserved(self):
        part = ET.parse(RAW_DATA_DIR / "21_cfr_part50.xml").getroot()
        self.assertEqual("DIV5", part.tag)
        self.assertEqual("50", part.get("N"))
        self.assertEqual("PART", part.get("TYPE"))

        sections = part.findall(".//DIV8[@TYPE='SECTION']")
        section_numbers = {section.get("N") for section in sections}
        self.assertEqual(15, len(sections))
        self.assertIn("§ 50.20", section_numbers)
        self.assertIn("§ 50.25", section_numbers)

        for section in sections:
            self.assertTrue(section.findtext("HEAD"))
            self.assertTrue(any(text.strip() for text in section.itertext() if text))


if __name__ == "__main__":
    unittest.main()
