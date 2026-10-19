"""Tests for structured, tool-assisted answer evaluation."""

import unittest
from unittest.mock import patch

from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage

from regulatory_agent.evaluation import (
    EVALUATION_INPUT_PROMPT,
    EVALUATION_RESPONSE_SCHEMA,
    AnswerEvaluationDraft,
    EvaluationScores,
    ToolCallingAnswerEvaluator,
)
from regulatory_agent.multi_agent import SynthesisSource


class FakeReviewer:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.state

    def get_graph(self):
        raise NotImplementedError


def scores(**overrides):
    values = {
        "accuracy": 9,
        "relevance": 9,
        "completeness": 8,
        "citation_accuracy": 9,
        "clarity_conciseness": 8,
        "objectivity": 9,
    }
    values.update(overrides)
    return EvaluationScores(**values)


class EvaluationTests(unittest.TestCase):
    def test_input_prompt_has_explicit_fields(self):
        self.assertEqual(
            {"answer", "citations", "question", "sources"},
            set(EVALUATION_INPUT_PROMPT.input_variables),
        )

    @patch("regulatory_agent.evaluation.create_agent")
    def test_evaluator_uses_tool_strategy_with_search_tools(self, create_agent):
        reviewer = FakeReviewer({"messages": []})
        create_agent.return_value = reviewer

        ToolCallingAnswerEvaluator(llm=object(), tools=[])

        response_format = create_agent.call_args.kwargs["response_format"]
        self.assertIsInstance(response_format, ToolStrategy)
        self.assertEqual("answer_evaluation_draft", response_format.schema["title"])
        self.assertEqual(EVALUATION_RESPONSE_SCHEMA, response_format.schema)

    def test_invalid_structured_response_schema_is_an_explicit_failure(self):
        evaluator = ToolCallingAnswerEvaluator(
            reviewer=FakeReviewer(
                {"structured_response": {"brief_evaluation": "Missing scores"}}
            )
        )

        with self.assertRaisesRegex(ValueError, "scores"):
            evaluator.evaluate(
                question="Question",
                answer="Answer",
                citations=[],
                sources=[],
            )

    def test_total_and_pass_decision_are_deterministic(self):
        draft = AnswerEvaluationDraft(
            scores=scores(),
            brief_evaluation="Grounded and complete.",
        )
        reviewer = FakeReviewer({"structured_response": draft, "messages": []})
        evaluator = ToolCallingAnswerEvaluator(reviewer=reviewer)

        result = evaluator.evaluate(
            question="What is required?",
            answer="A supported answer.",
            citations=["21 CFR § 50.25(a)"],
            sources=[
                SynthesisSource(
                    source_id="S1",
                    source_type="regulation",
                    content="A requirement.",
                    citation="21 CFR § 50.25(a)",
                )
            ],
        )

        self.assertEqual(52, result.total_score)
        self.assertEqual("pass", result.decision)
        prompt_text = reviewer.calls[0]["messages"][0].content
        self.assertIn("A supported answer", prompt_text)
        self.assertIn('"source_id": "S1"', prompt_text)

    def test_research_flag_overrides_a_high_score(self):
        reviewer = FakeReviewer(
            {
                "structured_response": AnswerEvaluationDraft(
                    scores=scores(),
                    brief_evaluation="A current fact still needs verification.",
                    needs_additional_research=True,
                    missing_information=["Current enforcement status"],
                ),
                "messages": [],
            }
        )

        result = ToolCallingAnswerEvaluator(reviewer=reviewer).evaluate(
            question="What changed?",
            answer="A tentative answer.",
            citations=[],
            sources=[],
        )

        self.assertEqual("research_again", result.decision)

    def test_human_review_has_highest_decision_priority(self):
        reviewer = FakeReviewer(
            {
                "structured_response": AnswerEvaluationDraft(
                    scores=scores(accuracy=6),
                    brief_evaluation="Applicability depends on unresolved facts.",
                    needs_additional_research=True,
                    requires_human_review=True,
                ),
                "messages": [],
            }
        )

        result = ToolCallingAnswerEvaluator(reviewer=reviewer).evaluate(
            question="Does this apply to us?",
            answer="It may apply.",
            citations=[],
            sources=[],
        )

        self.assertEqual("human_review", result.decision)

    def test_called_tools_are_recorded_once_in_first_use_order(self):
        tool_messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_cfr_regulations",
                        "args": {"question": "consent requirements"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_fda_guidance",
                        "args": {"question": "consent guidance"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                    {
                        "name": "answer_evaluation_draft",
                        "args": {},
                        "id": "structured-output",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_cfr_regulations",
                        "args": {"question": "verify"},
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
        reviewer = FakeReviewer(
            {
                "structured_response": AnswerEvaluationDraft(
                    scores=scores(),
                    brief_evaluation="Verified with tools.",
                ),
                "messages": tool_messages,
            }
        )

        result = ToolCallingAnswerEvaluator(reviewer=reviewer).evaluate(
            question="What is required?",
            answer="Verified answer.",
            citations=[],
            sources=[],
        )

        self.assertEqual(
            ["search_cfr_regulations", "search_fda_guidance"],
            result.tools_used,
        )

    def test_missing_structured_response_is_an_explicit_failure(self):
        evaluator = ToolCallingAnswerEvaluator(
            reviewer=FakeReviewer({"messages": []})
        )

        with self.assertRaisesRegex(ValueError, "no structured_response"):
            evaluator.evaluate(
                question="Question",
                answer="Answer",
                citations=[],
                sources=[],
            )


if __name__ == "__main__":
    unittest.main()
