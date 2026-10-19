"""
Extract a CFR Part from the eCFR Title 21 bulk XML.

Input:
    data/raw/ECFR-title21.xml

Output:
    data/raw/21_cfr_part50.xml
    data/raw/21_cfr_part50.metadata.json

Confirmed structure:
DLPSTEXTCLASS
└── TEXT
    └── BODY
        └── ECFRBRWS
            ├── AMDDATE
            └── DIV1 (Volume)
                └── DIV3 (Chapter)
                    └── DIV4 (Subchapter)
                        └── DIV5 (Part)
                            └── DIV6 (Subpart)
                                └── DIV8 (Section)
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import json

from regulatory_agent.paths import RAW_DATA_DIR

INPUT_PATH = RAW_DATA_DIR / "ECFR-title21.xml"
OUTPUT_DIR = RAW_DATA_DIR


def extract_part(
    input_path: Path,
    part_number: str,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_path = output_dir / f"21_cfr_part{part_number}.xml"
    metadata_path = output_dir / f"21_cfr_part{part_number}.metadata.json"

    stack = []
    current_amendment_date = None
    target_depth = None

    context = ET.iterparse(input_path, events=("start", "end"))

    for event, element in context:

        if event == "start":
            stack.append(element)

            if element.tag == "ECFRBRWS":
                current_amendment_date = None

            if (
                element.tag == "DIV5"
                and element.get("TYPE") == "PART"
                and element.get("N") == part_number
            ):
                target_depth = len(stack)

            continue

        # END event: element contents are complete

        if element.tag == "AMDDATE":
            current_amendment_date = "".join(element.itertext()).strip()

        if (
            target_depth is not None
            and len(stack) == target_depth
            and element.tag == "DIV5"
            and element.get("TYPE") == "PART"
            and element.get("N") == part_number
        ):
            ancestors = stack[:-1]

            def ancestor(tag, element_type):
                return next(
                    (
                        node
                        for node in reversed(ancestors)
                        if node.tag == tag and node.get("TYPE") == element_type
                    ),
                    None,
                )

            volume = ancestor("DIV1", "TITLE")
            chapter = ancestor("DIV3", "CHAPTER")
            subchapter = ancestor("DIV4", "SUBCHAP")

            metadata = {
                "title": "21",
                "part": part_number,
                "part_heading": element.findtext("HEAD"),
                "node": element.get("NODE"),
                "volume": volume.get("N") if volume is not None else None,
                "chapter": chapter.get("N") if chapter is not None else None,
                "subchapter": (subchapter.get("N") if subchapter is not None else None),
                "amendment_date": current_amendment_date,
                "source_file": input_path.name,
                "source_format": "ecfr_xml",
            }

            ET.indent(element, space="  ")
            ET.ElementTree(element).write(
                xml_path,
                encoding="utf-8",
                xml_declaration=True,
            )

            metadata_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            print(f"Saved XML: {xml_path}")
            print(f"Saved metadata: {metadata_path}")

            return xml_path, metadata

        stack.pop()

        # Keep every child of the target Part until it is saved.
        if target_depth is None:
            element.clear()

    raise ValueError(f"Part {part_number} not found")


def inspect_structure(
    element: ET.Element,
    depth: int = 0,
    max_depth: int = 3,
    max_children: int = 5,
) -> None:
    """Print the structure of an XML element."""

    if depth > max_depth:
        return

    heading = element.findtext("HEAD") or ""

    print(
        "  " * depth,
        element.tag,
        element.attrib,
        heading.strip()[:100],
    )

    if depth == max_depth:
        return

    for child in list(element)[:max_children]:
        inspect_structure(
            child,
            depth + 1,
            max_depth,
            max_children,
        )


def main():
    xml_path, metadata = extract_part(
        input_path=INPUT_PATH,
        part_number="50",
    )

    print("\n--- Metadata ---")
    print(json.dumps(metadata, indent=2))

    print("\n--- Part 50 Structure ---")

    # Parse only the extracted small XML
    part = ET.parse(xml_path).getroot()

    inspect_structure(part)

    sections = part.findall(".//DIV8[@TYPE='SECTION']")

    print(f"\nTotal sections: {len(sections)}")


if __name__ == "__main__":
    main()
