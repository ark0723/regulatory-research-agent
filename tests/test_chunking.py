"""Validation for baseline and section-aware retrieval preparation."""

import unittest
from collections import Counter, defaultdict

from regulatory_agent.chunking import (
    DEFAULT_CHUNK_SIZE,
    build_baseline_documents,
    build_chunked_documents,
    reconstruct_body,
)
from regulatory_agent.documents import build_cfr_documents, build_guidance_documents


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guidance_sections, cls.all_footnotes = build_guidance_documents()
        cls.cfr_sections = build_cfr_documents()
        cls.source_documents = (
            cls.cfr_sections
            + cls.guidance_sections
            + [
                document
                for document in cls.all_footnotes
                if document.metadata.get("footnote_attribution") == "associated"
            ]
        )
        cls.source_by_id = {
            document.metadata["record_id"]: document
            for document in cls.source_documents
        }
        cls.baseline = build_baseline_documents(cls.source_documents)
        cls.chunked = build_chunked_documents(cls.source_documents)
        cls.chunks_by_parent = defaultdict(list)
        for document in cls.chunked:
            cls.chunks_by_parent[document.metadata["parent_record_id"]].append(
                document
            )

    def test_baseline_count_content_and_metadata_preservation(self):
        self.assertEqual(173, len(self.baseline))
        counts = Counter(
            (document.metadata["document_type"], document.metadata["content_type"])
            for document in self.baseline
        )
        self.assertEqual(65, counts[("guidance", "section")])
        self.assertEqual(93, counts[("guidance", "footnote")])
        self.assertEqual(15, counts[("regulation", "section")])

        for document in self.baseline:
            source = self.source_by_id[document.metadata["record_id"]]
            self.assertEqual(source.page_content, document.page_content)
            for key in (
                "document_type",
                "content_type",
                "section_id",
                "citation",
                "source_url",
                "page_start",
                "page_end",
            ):
                self.assertEqual(source.metadata[key], document.metadata[key])
            self.assertEqual(source.metadata["record_id"], document.metadata["chunk_id"])
            self.assertEqual("section", document.metadata["chunking_strategy"])

    def test_source_documents_are_not_mutated(self):
        for source in self.source_documents:
            self.assertNotIn("chunk_id", source.metadata)
            self.assertNotIn("parent_record_id", source.metadata)

    def test_chunk_ids_are_unique_and_deterministic(self):
        chunk_ids = [document.metadata["chunk_id"] for document in self.chunked]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        repeated = build_chunked_documents(self.source_documents)
        self.assertEqual(
            [(d.metadata["chunk_id"], d.page_content, d.metadata) for d in self.chunked],
            [(d.metadata["chunk_id"], d.page_content, d.metadata) for d in repeated],
        )

    def test_no_empty_chunks(self):
        self.assertTrue(self.chunked)
        for document in self.chunked:
            self.assertTrue(document.page_content.strip())
            self.assertGreater(
                document.metadata["body_end"], document.metadata["body_start"]
            )

    def test_short_sections_remain_intact(self):
        parent_id = "fda-informed-consent-2023:section:III.A.3"
        chunks = self.chunks_by_parent[parent_id]
        self.assertEqual(1, len(chunks))
        self.assertEqual(parent_id, chunks[0].metadata["record_id"])
        self.assertEqual(self.source_by_id[parent_id].page_content, chunks[0].page_content)

    def test_representative_long_sections_are_split(self):
        for parent_id in (
            "21-cfr-part50:section:50.23",
            "21-cfr-part50:section:50.3",
            "21-cfr-part50:section:50.24",
            "fda-informed-consent-2023:section:V.5",
            "fda-informed-consent-2023:section:V.1",
            "fda-informed-consent-2023:section:III.A.2",
        ):
            self.assertGreater(len(self.chunks_by_parent[parent_id]), 1, parent_id)

    def test_cfr_identifiers_and_parent_context_are_preserved(self):
        chunks = self.chunks_by_parent["21-cfr-part50:section:50.23"]
        self.assertTrue(any(chunk.metadata["paragraph_start"] for chunk in chunks))
        self.assertTrue(
            any("(d)(1)" in chunk.metadata["paragraph_context"] for chunk in chunks)
        )
        for chunk in chunks:
            self.assertEqual("50.23", chunk.metadata["section_id"])
            self.assertEqual("21 CFR 50.23", chunk.metadata["citation"])
            if chunk.metadata["paragraph_context"]:
                self.assertIn(
                    f"Paragraph context: {chunk.metadata['paragraph_context']}",
                    chunk.page_content,
                )

    def test_guidance_chunks_never_cross_section_boundaries(self):
        for parent_id, chunks in self.chunks_by_parent.items():
            source = self.source_by_id[parent_id]
            if source.metadata["document_type"] != "guidance":
                continue
            for chunk in chunks:
                self.assertEqual(source.metadata["section_id"], chunk.metadata["section_id"])
                self.assertEqual(source.metadata["section_path"], chunk.metadata["section_path"])

    def test_footnotes_are_linked_only_and_never_split(self):
        unresolved_source = [
            document
            for document in self.all_footnotes
            if document.metadata.get("footnote_attribution") == "unresolved"
        ]
        self.assertEqual(["1"], [d.metadata["footnote_number"] for d in unresolved_source])
        footnotes = [
            document
            for document in self.chunked
            if document.metadata["content_type"] == "footnote"
        ]
        self.assertEqual(93, len(footnotes))
        self.assertNotIn("1", {d.metadata["footnote_number"] for d in footnotes})
        for footnote in footnotes:
            self.assertEqual(1, footnote.metadata["chunk_count"])
            self.assertEqual(
                footnote.metadata["record_id"], footnote.metadata["parent_record_id"]
            )

    def test_lossless_body_reconstruction_and_overlap(self):
        observed_overlap = False
        for parent_id, chunks in self.chunks_by_parent.items():
            source = self.source_by_id[parent_id]
            ordered = sorted(chunks, key=lambda chunk: chunk.metadata["chunk_index"])
            self.assertEqual(
                list(range(len(ordered))),
                [chunk.metadata["chunk_index"] for chunk in ordered],
            )
            self.assertTrue(
                all(chunk.metadata["chunk_count"] == len(ordered) for chunk in ordered)
            )
            starts = [chunk.metadata["body_start"] for chunk in ordered]
            self.assertEqual(starts, sorted(starts))
            previous_end = 0
            for chunk in ordered:
                self.assertEqual(
                    max(0, previous_end - chunk.metadata["body_start"]),
                    chunk.metadata["overlap_chars"],
                )
                previous_end = chunk.metadata["body_end"]
            reconstructed = reconstruct_body(source, chunks)
            _, separator, original_body = source.page_content.partition("\n\n")
            if not separator:
                original_body = source.page_content
            self.assertEqual(original_body, reconstructed)
            observed_overlap |= any(
                chunk.metadata["overlap_chars"] > 0 for chunk in chunks
            )
        self.assertTrue(observed_overlap)

    def test_metadata_and_citations_are_preserved(self):
        for parent_id, chunks in self.chunks_by_parent.items():
            source = self.source_by_id[parent_id]
            for chunk in chunks:
                self.assertEqual(parent_id, chunk.metadata["parent_record_id"])
                self.assertEqual(source.metadata["citation"], chunk.metadata["citation"])
                self.assertEqual(source.metadata["source_url"], chunk.metadata["source_url"])
                self.assertEqual(source.metadata["page_start"], chunk.metadata["page_start"])
                self.assertEqual(source.metadata["page_end"], chunk.metadata["page_end"])
                self.assertEqual("structure_aware", chunk.metadata["chunking_strategy"])

    def test_chunk_length_limit_and_reported_exceptions(self):
        exceptions = [
            document
            for document in self.chunked
            if len(document.page_content) > DEFAULT_CHUNK_SIZE
        ]
        self.assertEqual(
            exceptions,
            [
                document
                for document in self.chunked
                if document.metadata["chunk_size_exception"]
            ],
        )
        self.assertEqual([], exceptions)


if __name__ == "__main__":
    unittest.main()
