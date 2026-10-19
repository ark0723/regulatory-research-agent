"""Tests for cross-encoder reranking independent of model downloads."""

import unittest

from langchain_core.documents import Document

from regulatory_agent.rerank import rerank_evidence
from regulatory_agent.retrieval import Evidence


def evidence(record_id: str, parent_id: str, text: str, rank: int) -> Evidence:
    return Evidence(
        document=Document(
            page_content=text,
            metadata={"record_id": record_id, "chunk_id": record_id},
        ),
        score=rank / 10,
        retrieval_rank=rank,
        parent_record_id=parent_id,
    )


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, question, documents):
        self.calls.append((question, list(documents)))
        return self.scores


class RerankTests(unittest.TestCase):
    def test_rerank_orders_descending_and_keeps_one_chunk_per_parent(self):
        candidates = [
            evidence("a-1", "parent-a", "A1", 1),
            evidence("a-2", "parent-a", "A2", 2),
            evidence("b-1", "parent-b", "B1", 3),
        ]
        reranker = FakeReranker([0.9, 0.8, 0.7])

        result = rerank_evidence(
            "question",
            candidates,
            reranker=reranker,
            top_n=2,
        )

        self.assertEqual(
            ["a-1", "b-1"],
            [x.evidence.document.metadata["record_id"] for x in result],
        )
        self.assertEqual([0.9, 0.7], [x.rerank_score for x in result])
        self.assertEqual([1, 3], [x.rerank_rank for x in result])
        self.assertEqual([1, 2], [x.selection_rank for x in result])
        self.assertEqual([0.1, 0.3], [x.evidence.score for x in result])

    def test_rerank_without_parent_selection_has_no_selection_rank(self):
        candidates = [
            evidence("a-1", "parent-a", "A1", 1),
            evidence("a-2", "parent-a", "A2", 2),
            evidence("b-1", "parent-b", "B1", 3),
        ]

        result = rerank_evidence(
            "question",
            candidates,
            reranker=FakeReranker([0.8, 0.9, 0.7]),
            top_n=3,
            unique_parents=False,
        )

        self.assertEqual([1, 2, 3], [x.rerank_rank for x in result])
        self.assertEqual([None, None, None], [x.selection_rank for x in result])

    def test_reranker_must_return_one_score_per_candidate(self):
        with self.assertRaisesRegex(ValueError, "one score"):
            rerank_evidence(
                "question",
                [evidence("a", "parent-a", "A", 1)],
                reranker=FakeReranker([]),
            )


if __name__ == "__main__":
    unittest.main()
