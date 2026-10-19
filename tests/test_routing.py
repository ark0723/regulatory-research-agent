"""Tests for structured multi-label routing and parallel tool dispatch."""

import unittest

from regulatory_agent.routing import (
    ROUTING_PROMPT,
    RegulatoryQuestionRouter,
    RouteChoice,
    RoutedResearchCoordinator,
    RoutingDecision,
)


class StructuredInvoker:
    def __init__(self, responses):
        self.responses = responses

    def invoke(self, prompt):
        return self.responses.pop(0)


class FakeLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return StructuredInvoker(self.decisions)


class FakeTool:
    def __init__(self, name, response=None, error=None):
        self.name = name
        self.response = response or f"{name} result"
        self.error = error
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


def decision(*targets):
    return RoutingDecision(
        routes=[
            RouteChoice(target=target, reason=f"Needs {target} evidence")
            for target in targets
        ],
        rationale="Use the selected specialists.",
    )


class RegulatoryQuestionRouterTests(unittest.TestCase):
    def test_prompt_has_one_explicit_question_input(self):
        self.assertEqual(["question"], ROUTING_PROMPT.input_variables)

    def test_router_deduplicates_and_uses_stable_target_order(self):
        router = RegulatoryQuestionRouter(
            llm=FakeLLM([decision("web", "regulation", "web")])
        )

        result = router.route(
            "What is required and what are current implementation practices?"
        )

        self.assertEqual(
            ["regulation", "web"],
            [choice.target for choice in result.routes],
        )

    def test_coordinator_dispatches_only_selected_tools(self):
        regulation = FakeTool("regulation_tool")
        guidance = FakeTool("guidance_tool")
        web = FakeTool("web_tool")
        coordinator = RoutedResearchCoordinator(
            router=RegulatoryQuestionRouter(
                llm=FakeLLM([decision("regulation", "web")])
            ),
            tools={
                "regulation": regulation,
                "guidance": guidance,
                "web": web,
            },
        )

        result = coordinator.ask(
            "What is required and what are current implementation practices?"
        )

        self.assertEqual(
            ["regulation", "web"],
            [item.target for item in result.results],
        )
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual([], guidance.calls)
        self.assertEqual(1, len(web.calls))
        self.assertEqual(
            {
                "question": (
                    "What is required and what are current implementation practices?"
                )
            },
            regulation.calls[0],
        )

    def test_one_tool_failure_is_returned_without_losing_other_results(self):
        coordinator = RoutedResearchCoordinator(
            router=RegulatoryQuestionRouter(
                llm=FakeLLM([decision("guidance", "web")])
            ),
            tools={
                "regulation": FakeTool("regulation_tool"),
                "guidance": FakeTool(
                    "guidance_tool", error=RuntimeError("guidance unavailable")
                ),
                "web": FakeTool("web_tool", response="web evidence"),
            },
        )

        result = coordinator.ask("What does FDA recommend currently?")

        self.assertFalse(result.results[0].success)
        self.assertIn("guidance unavailable", result.results[0].error)
        self.assertTrue(result.results[1].success)
        self.assertEqual("web evidence", result.results[1].content)

    def test_coordinator_requires_every_architecture_target(self):
        with self.assertRaises(ValueError):
            RoutedResearchCoordinator(
                router=RegulatoryQuestionRouter(llm=FakeLLM([decision("web")])),
                tools={
                    "regulation": FakeTool("regulation_tool"),
                    "guidance": FakeTool("guidance_tool"),
                },
            )

    def test_graph_is_visible(self):
        coordinator = RoutedResearchCoordinator(
            router=RegulatoryQuestionRouter(llm=FakeLLM([decision("web")])),
            tools={
                "regulation": FakeTool("regulation_tool"),
                "guidance": FakeTool("guidance_tool"),
                "web": FakeTool("web_tool"),
            },
        )

        mermaid = coordinator.mermaid()

        self.assertIn("route_question", mermaid)
        self.assertIn("execute_tool", mermaid)


if __name__ == "__main__":
    unittest.main()
