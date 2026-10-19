"""Normalize eCFR Part XML into section-level JSON records."""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from regulatory_agent.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

INPUT_PATH = RAW_DATA_DIR / "21_cfr_part50.xml"
OUTPUT_PATH = PROCESSED_DATA_DIR / "21_cfr_part50.json"

SOURCE_URL = "https://www.ecfr.gov/current/title-21/part-50"


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace without changing the wording."""
    return re.sub(r"\s+", " ", text).strip()


def element_text(element: ET.Element) -> str:
    """Extract text, including text nested inside inline XML tags."""
    return normalize_whitespace("".join(element.itertext()))


def parse_section(
    section: ET.Element,
    subpart: ET.Element,
) -> dict:
    """Convert a DIV8 SECTION into a structured record."""

    section_label = section.get("N", "")
    section_number = section_label.replace("§", "").strip()

    heading = (
        element_text(section.find("HEAD")) if section.find("HEAD") is not None else ""
    )

    paragraphs = []

    for child in section:
        if child.tag == "HEAD":
            continue

        # P = regulation paragraph
        if child.tag == "P":
            text = element_text(child)

            if text:
                paragraphs.append(
                    {
                        "type": "paragraph",
                        "text": text,
                    }
                )

        # Other elements (e.g. citations) are retained separately
        elif child.tag in {"CITA", "NOTE", "SOURCE"}:
            text = element_text(child)

            if text:
                paragraphs.append(
                    {
                        "type": child.tag.lower(),
                        "text": text,
                    }
                )

        else:
            # Preserve unexpected elements rather than silently dropping them
            text = element_text(child)

            if text:
                paragraphs.append(
                    {
                        "type": child.tag.lower(),
                        "text": text,
                    }
                )

    content = "\n\n".join(paragraph["text"] for paragraph in paragraphs)

    subpart_heading = subpart.find("HEAD")
    subpart_title = element_text(subpart_heading) if subpart_heading is not None else ""

    return {
        "document_id": "21-cfr-part50",
        "document_type": "regulation",
        "title": "21",
        "part": "50",
        "subpart": subpart.get("N"),
        "subpart_title": subpart_title,
        "section": section_number,
        "section_label": section_label,
        "section_title": heading,
        "node": section.get("NODE"),
        "paragraphs": paragraphs,
        "content": content,
        "source_format": "ecfr_xml",
        "source_url": SOURCE_URL,
    }


def normalize_part(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> list[dict]:
    """Parse all sections and save them as JSON."""

    root = ET.parse(input_path).getroot()

    if root.tag != "DIV5" or root.get("TYPE") != "PART":
        raise ValueError("Input XML must be a CFR Part (DIV5).")

    part_number = root.get("N")

    if part_number != "50":
        raise ValueError(f"Expected Part 50, got Part {part_number}")

    records = []

    # Each DIV6 represents a Subpart
    for subpart in root.findall("./DIV6[@TYPE='SUBPART']"):
        for section in subpart.findall(".//DIV8[@TYPE='SECTION']"):
            records.append(parse_section(section, subpart))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Normalized sections: {len(records)}")
    print(f"Saved: {output_path}")

    return records


if __name__ == "__main__":
    normalize_part()
