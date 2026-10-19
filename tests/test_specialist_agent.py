"""Tests for the bounded corrective regulatory specialist graph."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from langchain_core.documents import Document
from regulatory_agent.citations import GeneratedAnswer
from regulatory_agent.retrieval import Evidence
from regulatory_agent.specialist_agent import (
    ANSWER_PROMPT,
    ASSESSMENT_PROMPT,
    CorrectiveRegulatorySpecialist,
    EvidenceAssessment,
    ExtractedEvidence,
    QueryRewrite,
    REWRITE_PROMPT,
)
from regulatory_agent.tools import RegulatorySearchResult, SearchDiagnostics


def search_result(text: str = "Consent must include required information."):
    document = Document(
        page_content=text,
        metadata={
            "record_id": "record-1",
            "chunk_id": "record-1",
            "parent_record_id": "parent-1",
            "chunk_index": 0,
            "chunk_count": 1,
            "body_start": 0,
            "body_end": len(text),
            "context_char_count": 0,
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
    evidence = Evidence(document, 0.1, 1, "parent-1")
    diagnostics = SearchDiagnostics(
        candidate_count=1,
        reranked_count=1,
        selected_parent_sections=["parent-1"],
        expanded_chunk_count=1,
        deduplicated_chunk_count=1,
        approximate_context_tokens=10,
        excluded_due_to_relevance=[],
        excluded_due_to_budget=[],
        evidence_complete=True,
        rerank_score_representation="test",
        retrieval_latency_ms=0,
        reranking_latency_ms=0,
        expansion_latency_ms=0,
        total_latency_ms=0,
    )
    return RegulatorySearchResult(
        document_type="regulation",
        candidates=[evidence],
        reranked=[],
        expanded_evidence=[evidence],
        warnings=[],
        diagnostics=diagnostics,
    )


def sufficient_assessment() -> EvidenceAssessment:
    return EvidenceAssessment(
        findings=[
            ExtractedEvidence(
                content="Consent must contain the required information.",
                citation="21 CFR § 50.25",
                supporting_evidence_ids=["E1"],
                relevance_score=0.95,
                faithfulness_score=0.95,
            )
        ],
        is_sufficient=True,
        answerability_score=0.95,
        rationale="The evidence directly answers the question.",
    )


def insufficient_assessment(missing: str) -> EvidenceAssessment:
    return EvidenceAssessment(
        is_sufficient=False,
        answerability_score=0.2,
        missing_information=[missing],
        rationale="More targeted evidence is required.",
    )


class StructuredInvoker:
    def __init__(self, responses):
        self.responses = responses

    def invoke(self, messages):
        return self.responses.pop(0)


class FakeLLM:
    def __init__(self, *, assessments, rewrites=None, answers=None):
        self.structured = {
            EvidenceAssessment: list(assessments),
            QueryRewrite: list(rewrites or []),
            GeneratedAnswer: list(answers or []),
        }
        self.answer_calls = []

    def with_structured_output(self, schema):
        if schema is GeneratedAnswer:
            self.answer_calls.append(schema)
        return StructuredInvoker(self.structured[schema])


class SearchSequence:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def __call__(self, query):
        self.queries.append(query)
        return self.results.pop(0)


class SpecialistAgentTests(unittest.TestCase):
    def test_prompt_templates_have_explicit_structured_inputs(self):
        self.assertEqual(
            {
                "active_query",
                "citation_catalog",
                "evidence_context",
                "evidence_warnings",
                "question",
            },
            set(ASSESSMENT_PROMPT.input_variables),
        )
        self.assertEqual(
            {
                "active_query",
                "missing_information",
                "query_history",
                "question",
                "rationale",
            },
            set(REWRITE_PROMPT.input_variables),
        )
        self.assertEqual(
            {
                "citation_catalog",
                "evidence_context",
                "evidence_warnings",
                "extracted_findings",
                "question",
            },
            set(ANSWER_PROMPT.input_variables),
        )

    def test_sufficient_evidence_generates_answer_without_rewrite(self):
        search = SearchSequence([search_result()])
        llm = FakeLLM(
            assessments=[sufficient_assessment()],
            answers=[
                GeneratedAnswer(
                    answer="Grounded answer. [[E1]]",
                    cited_evidence_ids=["E1"],
                )
            ],
        )
        agent = CorrectiveRegulatorySpecialist(
            llm=llm,
            document_type="regulation",
            search=search,
        )

        result = agent.ask("What must consent contain?")

        self.assertEqual(["What must consent contain?"], search.queries)
        self.assertEqual(0, result["rewrite_count"])
        self.assertEqual("Grounded answer. [21 CFR § 50.25]", result["answer"])
        self.assertEqual("answered", result["termination_reason"])
        self.assertEqual(["21 CFR § 50.25"], result["citations"])

    def test_insufficient_evidence_rewrites_then_retrieves_again(self):
        search = SearchSequence([search_result("partial"), search_result("complete")])
        llm = FakeLLM(
            assessments=[
                insufficient_assessment("the eight required elements"),
                sufficient_assessment(),
            ],
            rewrites=[
                QueryRewrite(
                    rewritten_query="21 CFR 50.25(a) eight elements",
                    rationale="Target the controlling paragraph.",
                )
            ],
            answers=[
                GeneratedAnswer(
                    answer="Answer after corrective retrieval. [[E1]]",
                    cited_evidence_ids=["E1"],
                )
            ],
        )
        agent = CorrectiveRegulatorySpecialist(
            llm=llm,
            document_type="regulation",
            search=search,
            max_rewrites=1,
        )

        result = agent.ask("What are the basic elements?")

        self.assertEqual(
            ["What are the basic elements?", "21 CFR 50.25(a) eight elements"],
            search.queries,
        )
        self.assertEqual(1, result["rewrite_count"])
        self.assertEqual(2, len(result["query_history"]))
        self.assertEqual(
            "Answer after corrective retrieval. [21 CFR § 50.25]",
            result["answer"],
        )

    def test_retry_limit_returns_explicit_insufficient_answer(self):
        search = SearchSequence([search_result("partial"), search_result("still partial")])
        llm = FakeLLM(
            assessments=[
                insufficient_assessment("applicability facts"),
                insufficient_assessment("applicability facts"),
            ],
            rewrites=[
                QueryRewrite(
                    rewritten_query="specific applicability requirements",
                    rationale="Target missing applicability evidence.",
                )
            ],
        )
        agent = CorrectiveRegulatorySpecialist(
            llm=llm,
            document_type="regulation",
            search=search,
            max_rewrites=1,
        )

        result = agent.ask("Does this rule apply?")

        self.assertEqual("insufficient_after_retries", result["termination_reason"])
        self.assertIn("insufficient", result["answer"])
        self.assertEqual([], llm.answer_calls)

    def test_invalid_generated_citation_returns_controlled_failure(self):
        search = SearchSequence([search_result()])
        llm = FakeLLM(
            assessments=[sufficient_assessment()],
            answers=[
                GeneratedAnswer(
                    answer="Grounded answer. 【source_text】 [[E1]]",
                    cited_evidence_ids=["E1"],
                )
            ],
        )
        agent = CorrectiveRegulatorySpecialist(
            llm=llm,
            document_type="regulation",
            search=search,
        )

        result = agent.ask("What must consent contain?")

        self.assertEqual("citation_validation_failed", result["termination_reason"])
        self.assertEqual([], result["citations"])
        self.assertNotIn("【source_text】", result["answer"])
        self.assertFalse(result["citation_validation"].valid)

    def test_specialist_can_be_exposed_as_a_routing_tool(self):
        agent = CorrectiveRegulatorySpecialist(
            llm=FakeLLM(
                assessments=[sufficient_assessment()],
                answers=[
                    GeneratedAnswer(
                        answer="Grounded answer. [[E1]]",
                        cited_evidence_ids=["E1"],
                    )
                ],
            ),
            document_type="regulation",
            search=SearchSequence([search_result()]),
        )

        output = agent.as_tool().invoke(
            {"question": "What must consent contain?"}
        )

        self.assertIn("Specialist source: regulation", output)
        self.assertIn("Termination: answered", output)
        self.assertIn("21 CFR § 50.25", output)

    def test_synthetic_chunk_id_is_not_accepted_as_original_record_id(self):
        result = search_result()
        original = result.expanded_evidence[0]
        synthetic_id = "parent-1:paragraph-scope:a"
        original.document.metadata.update(
            {
                "record_id": synthetic_id,
                "chunk_id": synthetic_id,
                "source_record_id": "parent-1",
                "synthetic_evidence": True,
            }
        )
        synthetic = Evidence(
            document=original.document,
            score=original.score,
            retrieval_rank=original.retrieval_rank,
            parent_record_id="parent-1",
            section_complete=False,
            evidence_scope="paragraph:(a)",
            scope_complete=True,
        )
        result = replace(result, expanded_evidence=[synthetic])
        assessment = EvidenceAssessment(
            findings=[
                ExtractedEvidence(
                    content="Consent must contain the required information.",
                    citation="21 CFR § 50.25(a)",
                    supporting_record_ids=[synthetic_id],
                    relevance_score=0.95,
                    faithfulness_score=0.95,
                )
            ],
            is_sufficient=True,
            answerability_score=0.95,
            rationale="The evidence directly answers the question.",
        )
        agent = CorrectiveRegulatorySpecialist(
            llm=FakeLLM(assessments=[assessment]),
            document_type="regulation",
            search=SearchSequence([result]),
        )

        graph_result = agent.ask("What must consent contain?")

        self.assertEqual(
            "citation_validation_failed", graph_result["termination_reason"]
        )
        self.assertIn(
            "synthetic_record_id_reference",
            [issue.code for issue in graph_result["citation_validation_issues"]],
        )

    def test_answer_cannot_cite_retrieved_evidence_not_selected_by_findings(self):
        result = search_result()
        unrelated_document = Document(
            page_content="An unrelated regulation.",
            metadata={
                **result.expanded_evidence[0].document.metadata,
                "record_id": "record-2",
                "chunk_id": "record-2",
                "parent_record_id": "parent-2",
                "section_id": "50.20",
            },
        )
        unrelated = Evidence(unrelated_document, 0.2, 2, "parent-2")
        result = replace(
            result,
            expanded_evidence=[result.expanded_evidence[0], unrelated],
        )
        llm = FakeLLM(
            assessments=[sufficient_assessment()],
            answers=[
                GeneratedAnswer(
                    answer="The answer relies on another result. [[E2]]",
                    cited_evidence_ids=["E2"],
                )
            ],
        )
        agent = CorrectiveRegulatorySpecialist(
            llm=llm,
            document_type="regulation",
            search=SearchSequence([result]),
        )

        graph_result = agent.ask("What must consent contain?")

        self.assertEqual(
            "citation_validation_failed", graph_result["termination_reason"]
        )
        self.assertIn(
            "unknown_evidence_reference",
            [issue.code for issue in graph_result["citation_validation_issues"]],
        )

    def test_graph_can_be_rendered_and_saved_as_mermaid(self):
        agent = CorrectiveRegulatorySpecialist(
            llm=FakeLLM(assessments=[]),
            document_type="regulation",
            search=SearchSequence([]),
        )

        mermaid = agent.mermaid()

        self.assertIn("retrieve", mermaid)
        self.assertIn("extract_and_evaluate", mermaid)
        self.assertIn("rewrite_query", mermaid)
        self.assertIn("generate_answer", mermaid)
        with tempfile.TemporaryDirectory() as directory:
            path = agent.save_mermaid(Path(directory) / "specialist.mmd")
            self.assertEqual(mermaid + "\n", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
