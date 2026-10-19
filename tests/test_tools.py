"""Tests for source-specific regulatory retrieval tools."""

import unittest

from langchain_core.documents import Document

from regulatory_agent.tools import (
    create_cfr_regulation_tool,
    create_fda_guidance_tool,
    retrieve_rerank_expand,
)


def document(record_id: str, document_type: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "record_id": record_id,
            "chunk_id": record_id,
            "parent_record_id": record_id,
            "chunk_index": 0,
            "chunk_count": 1,
            "body_start": 0,
            "body_end": len(text),
            "context_char_count": 0,
            "document_type": document_type,
            "content_type": "section",
            "section_id": "50.25" if document_type == "regulation" else "III.A",
            "citation": (
                "21 CFR 50.25"
                if document_type == "regulation"
                else "FDA Informed Consent Guidance (2023), Section III.A"
            ),
            "paragraph_context": "",
            "source_url": "https://example.test",
            "page_start": None,
            "page_end": None,
        },
    )


class FakeStore:
    def __init__(self):
        self.search_calls = []

    def similarity_search_with_score(self, **kwargs):
        self.search_calls.append(kwargs)
        document_type = kwargs["filter"]["document_type"]
        return [
            (
                document(
                    f"{document_type}-record",
                    document_type,
                    f"Relevant {document_type} evidence.",
                ),
                0.1,
            )
        ]

    def get_by_ids(self, ids):
        raise AssertionError("single-chunk sections should not require exact lookup")


class FakeReranker:
    def score(self, question, documents):
        return [0.95 for _ in documents]


class ScoreReranker:
    score_representation = "test_scores"

    def __init__(self, scores):
        self.scores = scores

    def score(self, question, documents):
        return self.scores


class RegulatoryToolTests(unittest.TestCase):
    def test_pipeline_filters_to_exact_document_type(self):
        store = FakeStore()

        result = retrieve_rerank_expand(
            store=store,
            reranker=FakeReranker(),
            question="What is required?",
            document_type="regulation",
            candidate_k=15,
            top_n=5,
        )

        self.assertEqual(
            {"document_type": "regulation"},
            store.search_calls[0]["filter"],
        )
        self.assertEqual(1, len(result.reranked))
        self.assertEqual(1, len(result.expanded_evidence))

    def test_tools_have_distinct_names_and_source_scopes(self):
        store = FakeStore()
        reranker = FakeReranker()
        cfr_tool = create_cfr_regulation_tool(
            store=store,
            reranker=reranker,
        )
        guidance_tool = create_fda_guidance_tool(
            store=store,
            reranker=reranker,
        )

        cfr_output = cfr_tool.invoke({"question": "What is required?"})
        guidance_output = guidance_tool.invoke({"question": "What does FDA recommend?"})

        self.assertEqual("search_cfr_regulations", cfr_tool.name)
        self.assertEqual("search_fda_guidance", guidance_tool.name)
        self.assertIn("binding CFR regulation", cfr_output)
        self.assertIn("21 CFR § 50.25", cfr_output)
        self.assertIn("nonbinding FDA guidance", guidance_output)
        self.assertIn("FDA Informed Consent Guidance", guidance_output)
        self.assertEqual(
            [
                {"document_type": "regulation"},
                {"document_type": "guidance"},
            ],
            [call["filter"] for call in store.search_calls],
        )

    def test_optional_score_filter_excludes_weak_parent_sections(self):
        store = FakeStore()
        store.similarity_search_with_score = lambda **kwargs: [
            (document("strong", "regulation", "Strong evidence"), 0.1),
            (document("weak-a", "regulation", "Weak evidence A"), 0.2),
            (document("weak-b", "regulation", "Weak evidence B"), 0.3),
        ]

        result = retrieve_rerank_expand(
            store=store,
            reranker=ScoreReranker([0.9, 0.02, 0.01]),
            question="What is required?",
            document_type="regulation",
            candidate_k=3,
            top_n=3,
            min_rerank_score=0.1,
        )

        self.assertEqual(["strong"], result.diagnostics.selected_parent_sections)
        self.assertEqual(
            ["weak-a", "weak-b"],
            result.diagnostics.excluded_due_to_relevance,
        )
        self.assertEqual("test_scores", result.diagnostics.rerank_score_representation)
        self.assertGreaterEqual(result.diagnostics.total_latency_ms, 0)
        self.assertEqual(3, result.diagnostics.candidate_count)
        self.assertEqual(3, result.diagnostics.reranked_count)

    def test_multiple_candidates_from_same_parent_expand_once(self):
        store = FakeStore()
        first = document("shared", "regulation", "First hit")
        second = document("shared", "regulation", "Second duplicate hit")
        store.similarity_search_with_score = lambda **kwargs: [
            (first, 0.1),
            (second, 0.2),
        ]

        result = retrieve_rerank_expand(
            store=store,
            reranker=ScoreReranker([0.8]),
            question="What is required?",
            document_type="regulation",
            candidate_k=2,
            top_n=2,
        )

        self.assertEqual(["shared"], result.diagnostics.selected_parent_sections)
        self.assertEqual(1, result.diagnostics.expanded_chunk_count)

    def test_pipeline_preserves_global_rerank_and_assigns_selection_rank(self):
        store = FakeStore()
        first = document("shared-1", "regulation", "Best shared-parent hit")
        duplicate_parent = document(
            "shared-2", "regulation", "Second shared-parent hit"
        )
        other = document("other-1", "regulation", "Other parent hit")
        first.metadata["parent_record_id"] = "shared"
        duplicate_parent.metadata["parent_record_id"] = "shared"
        other.metadata["parent_record_id"] = "other"
        store.similarity_search_with_score = lambda **kwargs: [
            (first, 0.1),
            (duplicate_parent, 0.2),
            (other, 0.3),
        ]

        result = retrieve_rerank_expand(
            store=store,
            reranker=ScoreReranker([0.9, 0.8, 0.7]),
            question="What is required?",
            document_type="regulation",
            candidate_k=3,
            top_n=2,
        )

        self.assertEqual([1, 3], [item.rerank_rank for item in result.reranked])
        self.assertEqual([1, 2], [item.selection_rank for item in result.reranked])
        self.assertEqual(
            [1, 3], result.diagnostics.selected_parent_rerank_ranks
        )
        self.assertEqual(
            [1, 2], result.diagnostics.selected_parent_selection_ranks
        )


if __name__ == "__main__":
    unittest.main()
