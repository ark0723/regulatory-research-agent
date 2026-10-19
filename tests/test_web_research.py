"""Tests for the Tavily search, extraction, and BGE reranking agent."""

import unittest

from regulatory_agent.web_research import (
    TavilyWebResearchAgent,
    render_web_research_result,
)


class FakeTool:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ContentReranker:
    score_representation = "test_scores"

    def __init__(self):
        self.calls = []

    def score(self, question, documents):
        self.calls.append((question, documents))
        scores = []
        for document in documents:
            text = document.page_content
            if "final best" in text:
                scores.append(0.99)
            elif "initial best" in text:
                scores.append(0.90)
            elif "initial second" in text:
                scores.append(0.80)
            elif "weak extracted" in text:
                scores.append(0.10)
            else:
                scores.append(0.20)
        return scores


def search_response():
    return {
        "results": [
            {
                "title": "Source A",
                "url": "https://a.example/article",
                "content": "initial best",
                "score": 0.4,
            },
            {
                "title": "Source B",
                "url": "https://b.example/article",
                "content": "initial second",
                "score": 0.9,
            },
            {
                "title": "Source C",
                "url": "https://c.example/article",
                "content": "not selected",
                "score": 1.0,
            },
        ],
        "response_time": 0.2,
    }


class TavilyWebResearchAgentTests(unittest.TestCase):
    def test_search_extract_and_final_rerank_use_bge_order(self):
        search = FakeTool(search_response())
        extract = FakeTool(
            {
                "results": [
                    {
                        "url": "https://a.example/article",
                        "raw_content": "weak extracted",
                    },
                    {
                        "url": "https://b.example/article",
                        "raw_content": "final best",
                    },
                ],
                "failed_results": [],
                "response_time": 0.3,
            }
        )
        reranker = ContentReranker()
        agent = TavilyWebResearchAgent(
            reranker=reranker,
            search_tool=search,
            extract_tool=extract,
            candidate_k=3,
            extract_k=2,
            top_n=2,
        )

        result = agent.ask(
            "practical safeguards",
            include_domains=["example"],
            time_range="month",
        )

        self.assertEqual(3, result.candidate_count)
        self.assertEqual(2, len(reranker.calls))
        self.assertEqual(
            {
                "query": "practical safeguards",
                "include_domains": ["example"],
                "time_range": "month",
            },
            search.calls[0],
        )
        self.assertEqual(
            {
                "urls": [
                    "https://a.example/article",
                    "https://b.example/article",
                ],
                "query": "practical safeguards",
            },
            extract.calls[0],
        )
        self.assertEqual(
            ["https://b.example/article", "https://a.example/article"],
            [source.url for source in result.sources],
        )
        self.assertEqual(["W1", "W2"], [source.evidence_id for source in result.sources])
        self.assertEqual([2, 1], [source.search_rank for source in result.sources])
        self.assertEqual([2, 1], [source.initial_rerank_rank for source in result.sources])
        self.assertEqual([1, 2], [source.rerank_rank for source in result.sources])

    def test_partial_extract_failure_falls_back_to_search_snippet(self):
        search = FakeTool(search_response())
        extract = FakeTool(
            {
                "results": [
                    {
                        "url": "https://a.example/article",
                        "raw_content": "weak extracted",
                    }
                ],
                "failed_results": [
                    {
                        "url": "https://b.example/article",
                        "error": "blocked",
                    }
                ],
            }
        )
        agent = TavilyWebResearchAgent(
            reranker=ContentReranker(),
            search_tool=search,
            extract_tool=extract,
            candidate_k=3,
            extract_k=2,
            top_n=2,
        )

        result = agent.ask("practical safeguards")

        source_b = next(
            source
            for source in result.sources
            if source.url == "https://b.example/article"
        )
        self.assertFalse(source_b.extracted)
        self.assertEqual("initial second", source_b.content)
        self.assertTrue(any("blocked" in warning for warning in result.warnings))

    def test_search_failure_returns_explicit_empty_result(self):
        agent = TavilyWebResearchAgent(
            reranker=ContentReranker(),
            search_tool=FakeTool(RuntimeError("service unavailable")),
            use_extract=False,
            candidate_k=3,
            extract_k=2,
            top_n=1,
        )

        result = agent.ask("current implementation information")

        self.assertEqual([], result.sources)
        self.assertTrue(any("service unavailable" in item for item in result.warnings))

    def test_duplicate_urls_are_removed_before_reranking(self):
        response = search_response()
        response["results"].insert(
            1,
            {
                "title": "Duplicate A",
                "url": "https://a.example/article#fragment",
                "content": "duplicate",
                "score": 1.0,
            },
        )
        reranker = ContentReranker()
        agent = TavilyWebResearchAgent(
            reranker=reranker,
            search_tool=FakeTool(response),
            use_extract=False,
            candidate_k=4,
            extract_k=3,
            top_n=2,
        )

        result = agent.ask("practical safeguards")

        self.assertEqual(3, result.candidate_count)
        self.assertEqual(3, len(reranker.calls[0][1]))

    def test_agent_can_be_exposed_as_a_text_tool(self):
        agent = TavilyWebResearchAgent(
            reranker=ContentReranker(),
            search_tool=FakeTool(search_response()),
            use_extract=False,
            candidate_k=3,
            extract_k=2,
            top_n=1,
        )

        output = agent.as_tool().invoke(
            {"question": "practical safeguards"}
        )

        self.assertIn('<web_evidence id="W1">', output)
        self.assertIn("https://a.example/article", output)
        self.assertIn("public web sources", output)

    def test_graph_is_visible_and_result_renderer_handles_empty_sources(self):
        agent = TavilyWebResearchAgent(
            reranker=ContentReranker(),
            search_tool=FakeTool(search_response()),
            use_extract=False,
            candidate_k=3,
            extract_k=2,
            top_n=1,
        )

        self.assertIn("search_web", agent.mermaid())
        result = agent.ask("practical safeguards")
        self.assertIn("W1", render_web_research_result(result))

    def test_invalid_pipeline_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            TavilyWebResearchAgent(
                reranker=ContentReranker(),
                search_tool=FakeTool(search_response()),
                candidate_k=2,
                extract_k=3,
                top_n=1,
            )


if __name__ == "__main__":
    unittest.main()
