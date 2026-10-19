"""Tests for scored retrieval and exact parent-section expansion."""

import unittest

from langchain_core.documents import Document

from regulatory_agent.retrieval import (
    Evidence,
    expand_parent_sections,
    render_evidence_context,
    retrieve_initial_evidence,
    sibling_vector_ids,
    supporting_citations,
)
from regulatory_agent.vector_db import build_index_documents, vector_document_id


class FakeStore:
    def __init__(self):
        self.search_results = []
        self.documents_by_id = {}
        self.search_calls = []
        self.get_calls = []

    def similarity_search_with_score(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_results

    def get_by_ids(self, ids):
        self.get_calls.append(list(ids))
        # Deliberately reverse storage order; expansion must restore chunk order.
        return [
            self.documents_by_id[document_id]
            for document_id in reversed(ids)
            if document_id in self.documents_by_id
        ]


def make_chunk(
    *,
    parent: str,
    index: int,
    count: int,
    body: str,
    start: int,
    end: int,
) -> Document:
    chunk_id = f"{parent}:structure-aware:1500-200:{index:03d}"
    prefix = "Document header\n\n"
    return Document(
        page_content=prefix + body,
        metadata={
            "record_id": chunk_id,
            "chunk_id": chunk_id,
            "parent_record_id": parent,
            "chunk_index": index,
            "chunk_count": count,
            "body_start": start,
            "body_end": end,
            "context_char_count": len(prefix),
            "document_type": "regulation",
            "content_type": "section",
            "section_id": "50.25",
            "citation": "21 CFR 50.25",
            "paragraph_context": "(a)",
            "source_url": "https://example.test",
            "page_start": None,
            "page_end": None,
        },
    )


class RetrievalTests(unittest.TestCase):
    def test_retrieval_preserves_scores_metadata_and_removes_duplicate_hits(self):
        store = FakeStore()
        first = make_chunk(
            parent="parent-a", index=0, count=1, body="A", start=0, end=1
        )
        store.search_results = [(first, 0.11), (first, 0.11)]

        evidence = retrieve_initial_evidence(store, "consent", k=5)

        self.assertEqual(1, len(evidence))
        self.assertEqual(0.11, evidence[0].score)
        self.assertEqual("21 CFR 50.25", evidence[0].document.metadata["citation"])
        self.assertEqual(
            {"query": "consent", "k": 5, "filter": None},
            store.search_calls[0],
        )

    def test_50_25_regression_expands_all_five_chunks_in_order(self):
        chunks = [
            document
            for document in build_index_documents()
            if document.metadata["parent_record_id"]
            == "21-cfr-part50:section:50.25"
        ]
        self.assertEqual(5, len(chunks))
        store = FakeStore()
        store.documents_by_id = {
            vector_document_id(document): document for document in chunks
        }
        initial = [
            Evidence(
                document=chunks[0],
                score=0.2,
                retrieval_rank=1,
                parent_record_id="21-cfr-part50:section:50.25",
            )
        ]

        result = expand_parent_sections(store, initial)

        self.assertEqual([], result.warnings)
        self.assertEqual(5, len(result.evidence))
        self.assertEqual(
            [0, 1, 2, 3, 4],
            [item.document.metadata["chunk_index"] for item in result.evidence],
        )
        self.assertTrue(all(item.section_complete for item in result.evidence))
        self.assertEqual(sibling_vector_ids(chunks[0]), store.get_calls[0])
        self.assertTrue(
            all(
                item.document.metadata["citation"] == "21 CFR 50.25"
                for item in result.evidence
            )
        )

    def test_50_25_paragraph_expansion_contains_all_a_elements_without_b(self):
        chunks = [
            document
            for document in build_index_documents()
            if document.metadata["parent_record_id"]
            == "21-cfr-part50:section:50.25"
        ]
        store = FakeStore()
        store.documents_by_id = {
            vector_document_id(document): document for document in chunks
        }
        hit = Evidence(
            document=chunks[0],
            score=0.1,
            retrieval_rank=1,
            parent_record_id="21-cfr-part50:section:50.25",
        )

        result = expand_parent_sections(
            store,
            [hit],
            max_expanded_chunks=5,
            max_context_chars=3_000,
            paragraph_aware=True,
        )
        context = render_evidence_context(result.evidence)

        self.assertEqual([], result.warnings)
        self.assertEqual(1, len(result.evidence))
        self.assertEqual("paragraph:(a)", result.evidence[0].evidence_scope)
        self.assertTrue(result.evidence[0].scope_complete)
        self.assertFalse(result.evidence[0].section_complete)
        self.assertTrue(
            result.evidence[0].document.metadata["synthetic_evidence"]
        )
        self.assertEqual(
            "21-cfr-part50:section:50.25",
            result.evidence[0].document.metadata["source_record_id"],
        )
        self.assertNotEqual(
            result.evidence[0].document.metadata["source_record_id"],
            result.evidence[0].document.metadata["record_id"],
        )
        for element in range(1, 9):
            self.assertIn(f"({element})", context)
        self.assertNotIn("(b) Additional elements", context)
        self.assertIn("citation: 21 CFR § 50.25", context)
        self.assertIn("paragraph_contexts: (a)", context)

    def test_multiple_hits_from_same_parent_do_not_duplicate_expansion(self):
        chunks = [
            make_chunk(
                parent="parent-a",
                index=index,
                count=2,
                body=body,
                start=index,
                end=index + len(body),
            )
            for index, body in enumerate(("AB", "BC"))
        ]
        store = FakeStore()
        store.documents_by_id = {
            vector_document_id(document): document for document in chunks
        }
        initial = [
            Evidence(chunks[0], 0.1, 1, "parent-a"),
            Evidence(chunks[1], 0.2, 2, "parent-a"),
            Evidence(chunks[0], 0.1, 3, "parent-a"),
        ]

        result = expand_parent_sections(store, initial)

        self.assertEqual(2, len(result.evidence))
        self.assertEqual(1, len(store.get_calls))
        self.assertEqual(
            len({item.document.metadata["record_id"] for item in result.evidence}),
            2,
        )

    def test_incomplete_exact_lookup_is_preserved_and_reported(self):
        chunks = [
            make_chunk(
                parent="parent-a",
                index=index,
                count=2,
                body=body,
                start=index,
                end=index + len(body),
            )
            for index, body in enumerate(("AB", "BC"))
        ]
        store = FakeStore()
        store.documents_by_id = {vector_document_id(chunks[0]): chunks[0]}
        result = expand_parent_sections(
            store, [Evidence(chunks[0], 0.1, 1, "parent-a")]
        )

        self.assertEqual(1, len(result.evidence))
        self.assertFalse(result.evidence[0].section_complete)
        self.assertIn("expected 2 chunks, found 1", result.warnings[0])

    def test_long_section_is_bounded_and_marked_incomplete(self):
        chunks = [
            make_chunk(
                parent="parent-long",
                index=index,
                count=3,
                body=character * 120,
                start=index * 120,
                end=(index + 1) * 120,
            )
            for index, character in enumerate(("A", "B", "C"))
        ]
        store = FakeStore()
        store.documents_by_id = {
            vector_document_id(document): document for document in chunks
        }

        result = expand_parent_sections(
            store,
            [Evidence(chunks[1], 0.1, 1, "parent-long")],
            max_expanded_chunks=3,
            max_context_chars=170,
        )

        self.assertEqual(1, len(result.evidence))
        self.assertFalse(result.evidence[0].section_complete)
        self.assertFalse(result.evidence[0].scope_complete)
        self.assertLessEqual(result.context_character_count, 170)
        self.assertTrue(
            any("truncated by the context budget" in warning for warning in result.warnings)
        )

    def test_wrong_chunk_identity_does_not_count_as_complete(self):
        chunks = [
            make_chunk(
                parent="parent-a",
                index=index,
                count=2,
                body=body,
                start=index,
                end=index + len(body),
            )
            for index, body in enumerate(("AB", "BC"))
        ]
        wrong = make_chunk(
            parent="parent-wrong",
            index=1,
            count=2,
            body="BC",
            start=1,
            end=3,
        )
        store = FakeStore()
        store.documents_by_id = {
            vector_document_id(chunks[0]): chunks[0],
            vector_document_id(chunks[1]): wrong,
        }

        result = expand_parent_sections(
            store, [Evidence(chunks[0], 0.1, 1, "parent-a")]
        )

        self.assertTrue(all(not item.section_complete for item in result.evidence))
        self.assertTrue(result.warnings)

    def test_render_context_removes_only_known_overlap(self):
        first = make_chunk(
            parent="parent-a", index=0, count=2, body="ABCDE", start=0, end=5
        )
        second = make_chunk(
            parent="parent-a", index=1, count=2, body="DEFGH", start=3, end=8
        )
        evidence = [
            Evidence(first, 0.1, 1, "parent-a"),
            Evidence(second, None, None, "parent-a"),
        ]

        context = render_evidence_context(evidence)

        self.assertIn("ABCDEFGH", context.replace("\n", ""))
        self.assertNotIn("ABCDEDEFGH", context.replace("\n", ""))
        self.assertIn("source_type: Binding CFR regulation", context)
        self.assertEqual(["21 CFR § 50.25"], supporting_citations(evidence))


if __name__ == "__main__":
    unittest.main()
