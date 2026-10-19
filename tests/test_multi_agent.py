"""Regression tests for routed multi-agent synthesis and citation safety."""

import unittest

from regulatory_agent.evaluation import AnswerEvaluation, EvaluationScores
from regulatory_agent.multi_agent import (
    FINAL_ANSWER_PROMPT,
    EvaluationRetryPolicy,
    FinalAnswerDraft,
    ResearchRetryPlan,
    ResearchSource,
    RegulatoryMultiAgent,
    SynthesisValidationResult,
    SynthesisSource,
    TargetedResearchQuery,
    merge_synthesis_sources,
    validate_research_retry_plan,
    validate_and_render_synthesis,
)
from regulatory_agent.routing import RouteChoice, RoutingDecision
from regulatory_agent.specialist_agent import EvidenceAssessment, ExtractedEvidence
from regulatory_agent.web_research import WebEvidence, WebResearchResult


class FakeRouter:
    def __init__(self, *targets):
        self.targets = targets
        self.calls = []

    def route(self, question):
        self.calls.append(question)
        return RoutingDecision(
            routes=[
                RouteChoice(target=target, reason=f"Need {target}")
                for target in self.targets
            ],
            rationale="Use selected source agents.",
        )


class FakeSpecialist:
    def __init__(self, *, citation, answer="Specialist answer", error=None):
        self.citation = citation
        self.answer = answer
        self.error = error
        self.calls = []

    def ask(self, question):
        self.calls.append(question)
        if self.error:
            raise self.error
        assessment = EvidenceAssessment(
            findings=[
                ExtractedEvidence(
                    content="A grounded specialist proposition.",
                    citation=self.citation,
                    supporting_evidence_ids=["E1"],
                    relevance_score=0.9,
                    faithfulness_score=0.9,
                )
            ],
            is_sufficient=True,
            answerability_score=0.9,
            rationale="Evidence answers the question.",
        )
        return {
            "assessment": assessment,
            "answer": self.answer,
            "citations": [self.citation],
            "termination_reason": "answered",
        }


class FakeWebAgent:
    def __init__(self, *, empty=False):
        self.empty = empty
        self.calls = []

    def ask(self, question):
        self.calls.append(question)
        sources = []
        if not self.empty:
            sources = [
                WebEvidence(
                    evidence_id="W1",
                    title="FDA Update",
                    url="https://www.fda.gov/example",
                    content="Current supplemental implementation context.",
                    search_rank=1,
                    tavily_score=0.8,
                    initial_rerank_score=0.9,
                    initial_rerank_rank=1,
                    extracted=True,
                    rerank_score=0.95,
                    rerank_rank=1,
                    selection_rank=1,
                )
            ]
        return WebResearchResult(
            question=question,
            candidate_count=len(sources),
            initially_selected_count=len(sources),
            sources=sources,
        )


class StructuredInvoker:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return self.responses.pop(0)


class FakeLLM:
    def __init__(self, drafts):
        self.invoker = StructuredInvoker(list(drafts))
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self.invoker


class FakeEvaluator:
    def __init__(self, evaluation=None, error=None):
        self.evaluations = (
            list(evaluation) if isinstance(evaluation, list) else [evaluation]
        )
        self.error = error
        self.calls = []

    def evaluate(self, **payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        if len(self.evaluations) > 1:
            result = self.evaluations.pop(0)
        else:
            result = self.evaluations[0]
        if isinstance(result, Exception):
            raise result
        return result


def passing_evaluation():
    return AnswerEvaluation(
        scores=EvaluationScores(
            accuracy=9,
            relevance=9,
            completeness=8,
            citation_accuracy=9,
            clarity_conciseness=8,
            objectivity=9,
        ),
        total_score=52,
        decision="pass",
        brief_evaluation="Grounded answer.",
    )


_DEFAULT_EVALUATOR = object()


def build_agent(
    *,
    targets,
    drafts,
    web_empty=False,
    regulation_error=None,
    evaluator=_DEFAULT_EVALUATOR,
    human_review_policy="on_reviewer_request",
    retry_policy=None,
    available_research_targets=None,
):
    regulation = FakeSpecialist(
        citation="21 CFR § 50.25(a)", error=regulation_error
    )
    guidance = FakeSpecialist(citation="FDA Consent Guidance (2023), III.B")
    web = FakeWebAgent(empty=web_empty)
    llm = FakeLLM(drafts)
    if evaluator is _DEFAULT_EVALUATOR:
        evaluator = FakeEvaluator(evaluation=passing_evaluation())
    agent = RegulatoryMultiAgent(
        llm=llm,
        router=FakeRouter(*targets),
        regulation_agent=regulation,
        guidance_agent=guidance,
        web_agent=web,
        evaluator=evaluator,
        human_review_policy=human_review_policy,
        retry_policy=retry_policy,
        available_research_targets=available_research_targets,
    )
    return agent, llm, regulation, guidance, web


class MultiAgentTests(unittest.TestCase):
    def test_prompt_has_explicit_structured_inputs(self):
        self.assertEqual(
            {"agent_summaries", "question", "sources"},
            set(FINAL_ANSWER_PROMPT.input_variables),
        )

    def test_selected_nodes_merge_once_and_render_source_specific_citations(self):
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation", "web"),
            drafts=[
                FinalAnswerDraft(
                    answer="A binding rule applies. [[S1]] Current context differs. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                )
            ],
        )

        result = agent.ask("What is required and what is current practice?")

        self.assertEqual("answered", result.termination_reason)
        self.assertEqual("success", result.evaluation_history[0].outcome)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual([], guidance.calls)
        self.assertEqual(1, len(web.calls))
        self.assertEqual(1, len(llm.invoker.calls))
        self.assertEqual(
            ["regulation", "web"],
            [item.target for item in result.contributions],
        )
        self.assertEqual(["S1", "S2"], [item.source_id for item in result.sources])
        self.assertIn("[21 CFR § 50.25(a)]", result.final_answer)
        self.assertIn("[FDA Update](https://www.fda.gov/example)", result.final_answer)

    def test_one_agent_failure_does_not_discard_other_grounded_result(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation", "web"),
            drafts=[
                FinalAnswerDraft(
                    answer="Supplemental context. [[S1]]",
                    cited_source_ids=["S1"],
                )
            ],
            regulation_error=RuntimeError("database unavailable"),
        )

        result = agent.ask("What changed recently?")

        self.assertEqual("answered", result.termination_reason)
        self.assertFalse(result.contributions[0].success)
        self.assertIn("database unavailable", result.contributions[0].error)
        self.assertTrue(result.contributions[1].success)
        self.assertEqual("web", result.sources[0].source_type)

    def test_no_sources_returns_insufficient_without_calling_synthesis_llm(self):
        agent, llm, _, _, _ = build_agent(
            targets=("web",), drafts=[], web_empty=True
        )

        result = agent.ask("알 수 없는 최신 정보가 있나요?")

        self.assertEqual("insufficient_evidence", result.termination_reason)
        self.assertIn("충분한 근거", result.final_answer)
        self.assertEqual([], llm.invoker.calls)

    def test_unknown_final_source_id_returns_controlled_failure(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Unsupported claim. [[S99]]",
                    cited_source_ids=["S99"],
                )
            ],
        )

        result = agent.ask("What is required?")

        self.assertEqual("synthesis_validation_failed", result.termination_reason)
        self.assertEqual([], result.citations)
        self.assertEqual("unknown_source_id", result.validation.issues[0].code)

    def test_successful_answer_runs_evaluation_node(self):
        evaluator = FakeEvaluator(evaluation=passing_evaluation())
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Supported answer. [[S1]]",
                    cited_source_ids=["S1"],
                )
            ],
            evaluator=evaluator,
        )

        result = agent.ask("What is required?")

        self.assertEqual("pass", result.evaluation.decision)
        self.assertIsNone(result.evaluation_error)
        self.assertEqual(1, len(evaluator.calls))
        self.assertEqual(result.final_answer, evaluator.calls[0]["answer"])

    def test_non_answer_termination_skips_evaluation(self):
        evaluator = FakeEvaluator(evaluation=passing_evaluation())
        agent, _, _, _, _ = build_agent(
            targets=("web",),
            drafts=[],
            web_empty=True,
            evaluator=evaluator,
        )

        result = agent.ask("Unknown question")

        self.assertIsNone(result.evaluation)
        self.assertEqual([], evaluator.calls)

    def test_evaluation_failure_is_reported_without_hiding_answer(self):
        evaluator = FakeEvaluator(error=RuntimeError("review model unavailable"))
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Supported answer. [[S1]]",
                    cited_source_ids=["S1"],
                )
            ],
            evaluator=evaluator,
        )

        result = agent.ask("What is required?")

        self.assertEqual("evaluation_failed", result.termination_reason)
        self.assertIn("Supported answer", result.final_answer)
        self.assertEqual("review model unavailable", result.evaluation_error)
        self.assertEqual("evaluation_failed", result.evaluation_history[0].outcome)

    def test_missing_evaluation_is_a_controlled_failure(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Supported answer. [[S1]]",
                    cited_source_ids=["S1"],
                )
            ],
            evaluator=None,
        )

        result = agent.ask("What is required?")

        self.assertEqual("evaluation_failed", result.termination_reason)
        self.assertIn("missing", result.evaluation_error)
        self.assertEqual("invalid", result.evaluation_history[0].decision)

    def test_revise_uses_existing_evidence_then_passes_re_evaluation(self):
        revision = passing_evaluation().model_copy(
            update={
                "decision": "revise",
                "brief_evaluation": "Remove the unsupported absolute claim.",
                "unsupported_claims": ["The requirement always applies."],
                "missing_information": ["Explain the supported condition."],
            }
        )
        evaluator = FakeEvaluator(evaluation=[revision, passing_evaluation()])
        agent, llm, regulation, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="The requirement always applies. [[S1]]",
                    cited_source_ids=["S1"],
                ),
                FinalAnswerDraft(
                    answer="The requirement applies under the supported condition. [[S1]]",
                    cited_source_ids=["S1"],
                ),
            ],
            evaluator=evaluator,
        )

        result = agent.ask("What is required?")

        self.assertEqual("answered", result.termination_reason)
        self.assertNotIn("always applies", result.final_answer)
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(1, result.revision_count)
        self.assertEqual(0, result.research_retry_count)
        self.assertEqual(
            ["revision_started", "success"],
            [entry.outcome for entry in result.evaluation_history],
        )
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(2, len(llm.invoker.calls))
        revision_prompt = llm.invoker.calls[1].to_string()
        self.assertIn("Remove the unsupported absolute claim", revision_prompt)
        self.assertIn('"source_id": "S1"', revision_prompt)

    def test_two_consecutive_revisions_are_allowed(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Draft 0. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Draft 1. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Draft 2. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=FakeEvaluator(
                evaluation=[revise, revise, passing_evaluation()]
            ),
        )

        result = agent.ask("What is required?")

        self.assertEqual("answered", result.termination_reason)
        self.assertIn("Draft 2", result.final_answer)
        self.assertEqual(2, result.evaluation_retry_count)
        self.assertEqual(2, result.revision_count)
        self.assertEqual(
            ["revision_started", "revision_started", "success"],
            [entry.outcome for entry in result.evaluation_history],
        )

    def test_third_revision_is_blocked_by_default_revision_limit(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Draft 0. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Draft 1. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Draft 2. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=FakeEvaluator(evaluation=[revise, revise, revise]),
        )

        result = agent.ask("What is required?")

        self.assertEqual("retry_limit_reached", result.termination_reason)
        self.assertEqual(2, result.evaluation_retry_count)
        self.assertEqual(2, result.revision_count)
        self.assertEqual(
            ["revision_started", "revision_started", "retry_limit_reached"],
            [entry.outcome for entry in result.evaluation_history],
        )

    def test_total_retry_limit_blocks_an_additional_revision(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Draft 0. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Draft 1. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=FakeEvaluator(evaluation=[revise, revise]),
            retry_policy=EvaluationRetryPolicy(
                max_total_automatic_retries=1,
                max_answer_revisions=2,
                max_research_retries=1,
            ),
        )

        result = agent.ask("What is required?")

        self.assertEqual("retry_limit_reached", result.termination_reason)
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(1, result.revision_count)

    def test_revision_with_unknown_citation_is_rejected_and_not_published(self):
        revise = passing_evaluation().model_copy(
            update={"decision": "revise", "brief_evaluation": "Tighten the claim."}
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Original supported answer. [[S1]]",
                    cited_source_ids=["S1"],
                ),
                FinalAnswerDraft(
                    answer="Fabricated support. [[S99]]",
                    cited_source_ids=["S99"],
                ),
            ],
            evaluator=FakeEvaluator(evaluation=revise),
        )

        result = agent.ask("What is required?")

        self.assertEqual("synthesis_validation_failed", result.termination_reason)
        self.assertIn("Original supported answer", result.final_answer)
        self.assertNotIn("Fabricated support", result.final_answer)
        self.assertEqual("unknown_source_id", result.validation.issues[0].code)
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(1, result.revision_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual([], guidance.calls)
        self.assertEqual([], web.calls)
        self.assertEqual(2, len(llm.invoker.calls))

    def test_revision_model_failure_is_controlled_and_counted_once(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        agent, llm, regulation, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Original supported answer. [[S1]]",
                    cited_source_ids=["S1"],
                ),
                RuntimeError("revision model unavailable"),
            ],
            evaluator=FakeEvaluator(evaluation=revise),
        )
        original_invoke = llm.invoker.invoke

        def invoke_or_raise(prompt):
            response = original_invoke(prompt)
            if isinstance(response, Exception):
                raise response
            return response

        llm.invoker.invoke = invoke_or_raise

        result = agent.ask("What is required?")

        self.assertEqual("revision_failed", result.termination_reason)
        self.assertIn("revision model unavailable", result.revision_error)
        self.assertIn("Original supported answer", result.final_answer)
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(1, result.revision_count)
        self.assertEqual(1, len(regulation.calls))

    def test_invalid_revision_schema_is_a_controlled_failure(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Original supported answer. [[S1]]",
                    cited_source_ids=["S1"],
                ),
                {"not_an_answer": True},
            ],
            evaluator=FakeEvaluator(evaluation=revise),
        )

        result = agent.ask("What is required?")

        self.assertEqual("revision_failed", result.termination_reason)
        self.assertIn("Revision model failed", result.revision_error)
        self.assertIn("Original supported answer", result.final_answer)
        self.assertEqual(1, result.revision_count)

    def test_failed_re_evaluation_does_not_replay_stale_revise_decision(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        evaluator = FakeEvaluator(
            evaluation=[revise, RuntimeError("second review unavailable")]
        )
        agent, _, regulation, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Draft 0. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Draft 1. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=evaluator,
        )

        result = agent.ask("What is required?")

        self.assertEqual("evaluation_failed", result.termination_reason)
        self.assertIn("second review unavailable", result.evaluation_error)
        self.assertEqual(1, result.revision_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(
            ["revision_started", "evaluation_failed"],
            [entry.outcome for entry in result.evaluation_history],
        )

    def test_revision_missing_required_inputs_fails_without_model_call(self):
        revise = passing_evaluation().model_copy(
            update={"decision": "revise", "brief_evaluation": ""}
        )
        agent, llm, _, _, _ = build_agent(targets=("regulation",), drafts=[])
        source = SynthesisSource(
            source_id="S1",
            source_type="regulation",
            content="Requirement",
            citation="21 CFR § 50.25(a)",
        )

        cases = (
            ({"evaluation": revise, "final_answer": "Answer", "sources": [source]}, "feedback"),
            ({"evaluation": passing_evaluation(), "final_answer": "", "sources": [source]}, "previous answer"),
            ({"evaluation": passing_evaluation(), "final_answer": "Answer", "sources": []}, "evidence"),
        )
        for state, expected in cases:
            with self.subTest(expected=expected):
                result = agent.revise_answer(state)
                self.assertEqual("revision_failed", result["termination_reason"])
                self.assertIn(expected, result["revision_error"].lower())
        self.assertEqual([], llm.invoker.calls)

    def test_research_again_executes_selected_retry_then_passes(self):
        research = passing_evaluation().model_copy(
            update={
                "decision": "research_again",
                "brief_evaluation": "FDA implementation guidance is missing.",
                "missing_information": ["FDA recommended implementation steps"],
                "unsupported_claims": ["The answer treats guidance as binding."],
            }
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["FDA recommended implementation steps"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA informed consent implementation guidance",
                    reason="The current evidence contains only binding requirements.",
                )
            ],
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Answer needing more evidence. [[S1]]",
                    cited_source_ids=["S1"],
                ),
                plan,
                FinalAnswerDraft(
                    answer="Binding rule. [[S1]] Guidance context. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(evaluation=[research, passing_evaluation()]),
        )

        result = agent.ask("What is required?")

        self.assertEqual("answered", result.termination_reason)
        self.assertEqual(
            ["research_retry_started", "success"],
            [entry.outcome for entry in result.evaluation_history],
        )
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(1, result.research_retry_count)
        self.assertEqual(plan, result.research_retry_plan)
        self.assertEqual([plan], result.research_retry_plans)
        self.assertIsNone(result.research_retry_error)
        self.assertEqual(2, len(result.evaluation_history))
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(
            ["FDA informed consent implementation guidance"], guidance.calls
        )
        self.assertEqual([], web.calls)
        self.assertEqual(3, len(llm.invoker.calls))
        self.assertEqual(2, len(result.sources))
        self.assertEqual(["S1", "S2"], [source.source_id for source in result.sources])
        self.assertEqual(1, len(result.research_retry_diagnostics))
        self.assertEqual(
            "evaluation_passed", result.research_retry_diagnostics[0].outcome
        )
        planner_prompt = llm.invoker.calls[1].to_string()
        self.assertIn("FDA implementation guidance is missing", planner_prompt)
        self.assertIn("FDA recommended implementation steps", planner_prompt)
        self.assertIn("The answer treats guidance as binding", planner_prompt)
        self.assertIn("regulation: 1 sources", planner_prompt)
        self.assertIn("original regulation query: What is required?", planner_prompt)
        self.assertNotIn("A grounded specialist proposition", planner_prompt)

    def test_research_retry_plan_supports_distinct_queries_per_target(self):
        research = passing_evaluation().model_copy(
            update={
                "decision": "research_again",
                "brief_evaluation": "Need binding and current evidence.",
                "missing_information": ["Exception text", "Current FDA activity"],
            }
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["Exception text", "Current FDA activity"],
            queries=[
                TargetedResearchQuery(
                    target="regulation",
                    query="21 CFR informed consent exception requirements",
                    reason="Verify the binding exception language.",
                ),
                TargetedResearchQuery(
                    target="web",
                    query="recent FDA informed consent enforcement activity",
                    reason="The requested enforcement context is time-sensitive.",
                ),
            ],
        )
        agent, _, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Partial. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Rule and exception. [[S1]] Current context. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(evaluation=[research, passing_evaluation()]),
        )

        result = agent.ask("What is required and what is current?")

        self.assertEqual(
            ["regulation", "web"],
            [item.target for item in result.research_retry_plan.queries],
        )
        self.assertNotEqual(
            result.research_retry_plan.queries[0].query,
            result.research_retry_plan.queries[1].query,
        )
        self.assertEqual("answered", result.termination_reason)
        self.assertEqual(
            [
                "What is required and what is current?",
                "21 CFR informed consent exception requirements",
            ],
            regulation.calls,
        )
        self.assertEqual([], guidance.calls)
        self.assertEqual(
            ["recent FDA informed consent enforcement activity"], web.calls
        )
        self.assertEqual(1, result.research_retry_count)

    def test_retry_executes_multiple_queries_for_one_specialist_as_one_attempt(self):
        research = passing_evaluation().model_copy(
            update={
                "decision": "research_again",
                "missing_information": ["Two distinct guidance issues"],
            }
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["Two distinct guidance issues"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA consent documentation recommendations",
                    reason="Clarify documentation recommendations.",
                ),
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA electronic consent presentation recommendations",
                    reason="Clarify presentation recommendations.",
                ),
            ],
        )
        agent, _, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Rule. [[S1]] Guidance. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(evaluation=[research, passing_evaluation()]),
        )

        result = agent.ask("What is required?")

        self.assertEqual("answered", result.termination_reason)
        self.assertEqual(1, result.research_retry_count)
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(
            [query.query for query in plan.queries], guidance.calls
        )
        self.assertEqual([], web.calls)
        self.assertEqual(2, len(result.sources))
        self.assertEqual(2, len(result.research_retry_diagnostics[0].executions))

    def test_source_merge_uses_persistent_identity_and_rebuilds_source_ids(self):
        existing = ResearchSource(
            source_id="S7",
            source_type="guidance",
            title="Consent guidance",
            content="First supported proposition.",
            citation="FDA Guidance, III.B",
            source_record_ids=["record-1"],
            evidence_record_ids=["chunk-1"],
        )
        duplicate = ResearchSource(
            source_id="temporary",
            source_type="guidance",
            title="Consent guidance",
            content="Complementary supported proposition.",
            citation="FDA Guidance, III.B",
            source_record_ids=["record-1"],
            evidence_record_ids=["chunk-2"],
        )
        distinct = ResearchSource(
            source_id="temporary-2",
            source_type="guidance",
            title="Consent guidance",
            content="A different record with the same display metadata.",
            citation="FDA Guidance, III.B",
            source_record_ids=["record-2"],
            evidence_record_ids=["chunk-3"],
        )

        merged, duplicates, excluded = merge_synthesis_sources(
            [existing], [duplicate, distinct], max_sources=10
        )

        self.assertEqual(1, duplicates)
        self.assertEqual(0, excluded)
        self.assertEqual(["S1", "S2"], [source.source_id for source in merged])
        self.assertIn("First supported proposition", merged[0].content)
        self.assertIn("Complementary supported proposition", merged[0].content)
        self.assertEqual(["chunk-1", "chunk-2"], merged[0].evidence_record_ids)
        self.assertEqual(["record-2"], merged[1].source_record_ids)

    def test_source_merge_keeps_same_title_and_citation_for_distinct_web_urls(self):
        sources = [
            ResearchSource(
                source_id=f"old-{index}",
                source_type="web",
                title="FDA Update",
                content=f"Update {index}",
                citation="FDA Update",
                url=f"https://example.test/{index}",
            )
            for index in range(1, 4)
        ]

        merged, duplicates, excluded = merge_synthesis_sources(
            [], sources, max_sources=2
        )

        self.assertEqual(0, duplicates)
        self.assertEqual(1, excluded)
        self.assertEqual(["S1", "S2"], [source.source_id for source in merged])
        self.assertNotEqual(merged[0].url, merged[1].url)

    def test_retry_plan_deduplicates_identical_target_query_pairs(self):
        plan = ResearchRetryPlan(
            evidence_gaps=["Guidance scope"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA electronic consent guidance",
                    reason="Clarify recommendations.",
                ),
                TargetedResearchQuery(
                    target="guidance",
                    query="  FDA electronic consent guidance  ",
                    reason="Duplicate wording.",
                ),
            ],
        )

        validated = validate_research_retry_plan(
            plan,
            available_targets={"regulation", "guidance", "web"},
            max_queries=3,
        )

        self.assertEqual(1, len(validated.queries))

    def test_retry_plan_rejects_empty_unsupported_and_unavailable_queries(self):
        invalid_cases = (
            ResearchRetryPlan(
                evidence_gaps=["Guidance scope"],
                queries=[
                    TargetedResearchQuery(
                        target="guidance", query="   ", reason="Needed"
                    )
                ],
            ),
            ResearchRetryPlan.model_construct(
                evidence_gaps=["Other evidence"],
                queries=[
                    TargetedResearchQuery.model_construct(
                        target="other", query="Focused query", reason="Needed"
                    )
                ],
            ),
            ResearchRetryPlan(
                evidence_gaps=["Current evidence"],
                queries=[
                    TargetedResearchQuery(
                        target="web", query="Recent FDA update", reason="Current fact"
                    )
                ],
            ),
        )
        available = (
            {"regulation", "guidance", "web"},
            {"regulation", "guidance", "web"},
            {"regulation", "guidance"},
        )

        for plan, targets in zip(invalid_cases, available):
            with self.subTest(plan=plan):
                with self.assertRaises(ValueError):
                    validate_research_retry_plan(
                        plan,
                        available_targets=targets,
                        max_queries=3,
                    )

    def test_retry_plan_respects_query_limit_and_rejects_completion_claims(self):
        too_many = ResearchRetryPlan(
            evidence_gaps=["Several gaps"],
            queries=[
                TargetedResearchQuery(
                    target=target,
                    query=f"Focused {target} query",
                    reason="Needed for the gap.",
                )
                for target in ("regulation", "guidance", "web")
            ],
        )
        completed_claim = ResearchRetryPlan(
            evidence_gaps=["Guidance interpretation"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA consent guidance",
                    reason="We found guidance evidence confirming the answer.",
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "maximum"):
            validate_research_retry_plan(
                too_many,
                available_targets={"regulation", "guidance", "web"},
                max_queries=2,
            )
        with self.assertRaisesRegex(ValueError, "already retrieved"):
            validate_research_retry_plan(
                completed_claim,
                available_targets={"regulation", "guidance", "web"},
                max_queries=3,
            )

    def test_retry_planning_failure_is_controlled_and_does_not_run_research(self):
        research = passing_evaluation().model_copy(
            update={
                "decision": "research_again",
                "brief_evaluation": "Need guidance evidence.",
                "missing_information": ["Guidance scope"],
            }
        )
        agent, _, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Partial. [[S1]]", cited_source_ids=["S1"]),
                {"not_a_plan": True},
            ],
            evaluator=FakeEvaluator(evaluation=research),
        )

        result = agent.ask("What is required?")

        self.assertEqual(
            "research_retry_planning_failed", result.termination_reason
        )
        self.assertIn("evidence_gaps", result.research_retry_error)
        self.assertIsNone(result.research_retry_plan)
        self.assertEqual(1, len(result.evaluation_history))
        self.assertEqual(
            "research_retry_planning_failed",
            result.evaluation_history[0].outcome,
        )
        self.assertEqual(0, result.evaluation_retry_count)
        self.assertEqual(0, result.research_retry_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual([], guidance.calls)
        self.assertEqual([], web.calls)

    def test_retry_rejects_fabricated_source_id_without_reusing_stale_validation(self):
        research = passing_evaluation().model_copy(
            update={"decision": "research_again", "missing_information": ["Guidance"]}
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["Guidance"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA consent guidance",
                    reason="Add recommendation evidence.",
                )
            ],
        )
        evaluator = FakeEvaluator(evaluation=research)
        agent, _, _, guidance, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Unsupported retry citation. [[S99]]",
                    cited_source_ids=["S99"],
                ),
            ],
            evaluator=evaluator,
        )

        result = agent.ask("What is required?")

        self.assertEqual("synthesis_validation_failed", result.termination_reason)
        self.assertFalse(result.validation.valid)
        self.assertIsNone(result.evaluation)
        self.assertEqual("unknown_source_id", result.validation.issues[0].code)
        self.assertEqual("Rule. [21 CFR § 50.25(a)]", result.final_answer)
        self.assertEqual(1, len(evaluator.calls))
        self.assertEqual(["FDA consent guidance"], guidance.calls)

    def test_partial_retry_failure_preserves_successes_and_stops_conservatively(self):
        research = passing_evaluation().model_copy(
            update={"decision": "research_again", "missing_information": ["Two gaps"]}
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["Two gaps"],
            queries=[
                TargetedResearchQuery(
                    target="regulation",
                    query="binding exception",
                    reason="Need binding text.",
                ),
                TargetedResearchQuery(
                    target="guidance",
                    query="recommended implementation",
                    reason="Need implementation guidance.",
                ),
            ],
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("web",),
            drafts=[
                FinalAnswerDraft(answer="Current fact. [[S1]]", cited_source_ids=["S1"]),
                plan,
            ],
            evaluator=FakeEvaluator(evaluation=research),
            regulation_error=RuntimeError("database unavailable"),
        )

        result = agent.ask("What is required and current?")

        self.assertEqual("research_retry_partial_failure", result.termination_reason)
        self.assertEqual(1, result.research_retry_count)
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(["binding exception"], regulation.calls)
        self.assertEqual(["recommended implementation"], guidance.calls)
        self.assertEqual(["What is required and current?"], web.calls)
        self.assertEqual(2, len(llm.invoker.calls))
        self.assertEqual(2, len(result.sources))
        diagnostic = result.research_retry_diagnostics[0]
        self.assertEqual("partial_failure", diagnostic.outcome)
        self.assertEqual([False, True], [item.success for item in diagnostic.executions])

    def test_complete_retry_failure_preserves_existing_evidence(self):
        research = passing_evaluation().model_copy(
            update={"decision": "research_again", "missing_information": ["Binding text"]}
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["Binding text"],
            queries=[
                TargetedResearchQuery(
                    target="regulation",
                    query="binding exception",
                    reason="Need binding text.",
                )
            ],
        )
        agent, _, regulation, _, web = build_agent(
            targets=("web",),
            drafts=[
                FinalAnswerDraft(answer="Current fact. [[S1]]", cited_source_ids=["S1"]),
                plan,
            ],
            evaluator=FakeEvaluator(evaluation=research),
            regulation_error=RuntimeError("database unavailable"),
        )

        result = agent.ask("What is current?")

        self.assertEqual("research_retry_failed", result.termination_reason)
        self.assertEqual(1, result.research_retry_count)
        self.assertEqual(["binding exception"], regulation.calls)
        self.assertEqual(["What is current?"], web.calls)
        self.assertEqual(1, len(result.sources))
        self.assertEqual("failed", result.research_retry_diagnostics[0].outcome)

    def test_second_research_decision_stops_at_retry_limit_without_reexecution(self):
        research = passing_evaluation().model_copy(
            update={"decision": "research_again", "missing_information": ["Guidance"]}
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["Guidance"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA consent guidance",
                    reason="Need recommendation evidence.",
                )
            ],
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Rule. [[S1]] Guidance. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(evaluation=[research, research]),
        )

        result = agent.ask("What is required?")

        self.assertEqual("retry_limit_reached", result.termination_reason)
        self.assertEqual(1, result.research_retry_count)
        self.assertEqual(1, result.evaluation_retry_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(["FDA consent guidance"], guidance.calls)
        self.assertEqual([], web.calls)
        self.assertEqual(3, len(llm.invoker.calls))
        self.assertEqual(
            ["research_retry_started", "retry_limit_reached"],
            [entry.outcome for entry in result.evaluation_history],
        )

    def test_revision_after_research_reuses_merged_evidence_without_new_research(self):
        research = passing_evaluation().model_copy(update={"decision": "research_again"})
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        plan = ResearchRetryPlan(
            evidence_gaps=["Guidance"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA consent guidance",
                    reason="Need guidance context.",
                )
            ],
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Draft. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Retry draft. [[S1]] [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
                FinalAnswerDraft(
                    answer="Revised retry draft. [[S1]] [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(
                evaluation=[research, revise, passing_evaluation()]
            ),
        )

        result = agent.ask("What is required?")

        self.assertEqual("answered", result.termination_reason)
        self.assertEqual(2, result.evaluation_retry_count)
        self.assertEqual(1, result.research_retry_count)
        self.assertEqual(1, result.revision_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(["FDA consent guidance"], guidance.calls)
        self.assertEqual([], web.calls)
        self.assertEqual(4, len(llm.invoker.calls))
        self.assertEqual(
            ["research_retry_started", "revision_started", "success"],
            [entry.outcome for entry in result.evaluation_history],
        )

    def test_human_review_after_research_resumes_without_duplicate_execution(self):
        research = passing_evaluation().model_copy(update={"decision": "research_again"})
        review = passing_evaluation().model_copy(update={"decision": "human_review"})
        plan = ResearchRetryPlan(
            evidence_gaps=["Guidance"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA consent guidance",
                    reason="Need guidance context.",
                )
            ],
        )
        agent, _, regulation, guidance, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Draft. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Retry draft. [[S1]] [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(evaluation=[research, review]),
        )

        paused = agent.ask("What is required?", thread_id="retry-review")
        self.assertEqual("awaiting_human_review", paused.termination_reason)
        self.assertEqual(1, paused.research_retry_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(["FDA consent guidance"], guidance.calls)

        resumed = agent.resume_human_review(
            thread_id="retry-review", approved=True, feedback="Approved."
        )

        self.assertEqual("human_approved", resumed.termination_reason)
        self.assertEqual(1, resumed.research_retry_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(["FDA consent guidance"], guidance.calls)

    def test_unavailable_planned_specialist_produces_controlled_failure(self):
        research = passing_evaluation().model_copy(
            update={
                "decision": "research_again",
                "brief_evaluation": "Need current external evidence.",
                "missing_information": ["Current FDA activity"],
            }
        )
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Partial. [[S1]]", cited_source_ids=["S1"]),
                ResearchRetryPlan(
                    evidence_gaps=["Current FDA activity"],
                    queries=[
                        TargetedResearchQuery(
                            target="web",
                            query="Recent FDA informed consent activity",
                            reason="Current evidence is needed.",
                        )
                    ],
                ),
            ],
            evaluator=FakeEvaluator(evaluation=research),
            available_research_targets={"regulation", "guidance"},
        )

        result = agent.ask("What is current?")

        self.assertEqual(
            "research_retry_planning_failed", result.termination_reason
        )
        self.assertIn("unavailable: web", result.research_retry_error)
        self.assertIsNone(result.research_retry_plan)
        self.assertEqual(1, len(result.evaluation_history))

    def test_research_retry_limit_is_checked_before_planner_invocation(self):
        research = passing_evaluation().model_copy(
            update={"decision": "research_again"}
        )
        agent, llm, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[],
            retry_policy=EvaluationRetryPolicy(max_research_retries=1),
        )

        command = agent.route_after_evaluation(
            {
                "termination_reason": "answered",
                "evaluation": research,
                "evaluation_retry_count": 1,
                "revision_count": 0,
                "research_retry_count": 1,
                "evaluation_history": [],
            }
        )

        self.assertEqual("retry_limit_reached", command.update["termination_reason"])
        self.assertEqual("__end__", command.goto)
        self.assertEqual([], llm.invoker.calls)

    def test_retry_limit_is_non_successful_and_does_not_increment_counters(self):
        revision = passing_evaluation().model_copy(update={"decision": "revise"})
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[],
            retry_policy=EvaluationRetryPolicy(
                max_total_automatic_retries=3,
                max_answer_revisions=2,
                max_research_retries=1,
            ),
        )
        state = {
            "termination_reason": "answered",
            "evaluation": revision,
            "evaluation_retry_count": 3,
            "revision_count": 2,
            "research_retry_count": 1,
            "evaluation_history": [],
        }

        command = agent.route_after_evaluation(state)

        self.assertEqual("retry_limit_reached", command.update["termination_reason"])
        self.assertEqual(3, command.update["evaluation_history"][0].evaluation_retry_count)
        self.assertNotIn("evaluation_retry_count", command.update)

    def test_alternating_retry_counts_cannot_bypass_total_limit(self):
        research = passing_evaluation().model_copy(
            update={"decision": "research_again"}
        )
        agent, _, _, _, _ = build_agent(targets=("regulation",), drafts=[])
        state = {
            "termination_reason": "answered",
            "evaluation": research,
            # Deliberately inconsistent total demonstrates the defensive sum check.
            "evaluation_retry_count": 0,
            "revision_count": 2,
            "research_retry_count": 1,
            "evaluation_history": [],
        }

        command = agent.route_after_evaluation(state)

        self.assertEqual("retry_limit_reached", command.update["termination_reason"])
        self.assertEqual("retry_limit_reached", command.update["evaluation_history"][0].outcome)

    def test_invalid_evaluation_decision_is_a_controlled_failure(self):
        invalid = passing_evaluation().model_copy(update={"decision": "unsupported"})
        agent, _, _, _, _ = build_agent(targets=("regulation",), drafts=[])

        command = agent.route_after_evaluation(
            {
                "termination_reason": "answered",
                "evaluation": invalid,
                "evaluation_history": [],
            }
        )

        self.assertEqual("evaluation_failed", command.update["termination_reason"])
        self.assertEqual("invalid", command.update["evaluation_history"][0].decision)

    def test_pass_requires_successful_citation_validation(self):
        agent, _, _, _, _ = build_agent(targets=("regulation",), drafts=[])

        command = agent.route_after_evaluation(
            {
                "termination_reason": "answered",
                "evaluation": passing_evaluation(),
                "validation": SynthesisValidationResult(valid=False),
                "evaluation_history": [],
            }
        )

        self.assertEqual("evaluation_failed", command.update["termination_reason"])
        self.assertIn("citation validation", command.update["evaluation_error"])

    def test_human_review_interrupt_can_be_approved_and_resumed(self):
        review_required = passing_evaluation().model_copy(
            update={
                "decision": "human_review",
                "brief_evaluation": "Applicability depends on user facts.",
            }
        )
        evaluator = FakeEvaluator(evaluation=review_required)
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Conditionally applicable. [[S1]]",
                    cited_source_ids=["S1"],
                )
            ],
            evaluator=evaluator,
        )

        paused = agent.ask("Does this apply?", thread_id="review-approve")

        self.assertEqual("awaiting_human_review", paused.termination_reason)
        self.assertEqual("review-approve", paused.thread_id)
        self.assertEqual(
            "Applicability depends on user facts.",
            paused.human_review_request.reason,
        )
        resumed = agent.resume_human_review(
            thread_id=paused.thread_id,
            approved=True,
            feedback="Approved after confirming applicability facts.",
        )
        self.assertEqual("human_approved", resumed.termination_reason)
        self.assertTrue(resumed.human_review.approved)
        self.assertEqual(1, len(resumed.evaluation_history))

    def test_resume_after_revision_does_not_repeat_revision_attempt(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        review = passing_evaluation().model_copy(
            update={
                "decision": "human_review",
                "brief_evaluation": "Confirm applicability.",
            }
        )
        evaluator = FakeEvaluator(evaluation=[revise, review])
        agent, llm, regulation, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Draft 0. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Draft 1. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=evaluator,
        )

        paused = agent.ask("Does this apply?", thread_id="revision-review")
        self.assertEqual("awaiting_human_review", paused.termination_reason)
        self.assertEqual(1, paused.evaluation_retry_count)
        self.assertEqual(1, paused.revision_count)
        self.assertEqual(2, len(paused.evaluation_history))

        resumed = agent.resume_human_review(
            thread_id="revision-review",
            approved=True,
            feedback="Applicability confirmed.",
        )

        self.assertEqual("human_approved", resumed.termination_reason)
        self.assertEqual(1, resumed.evaluation_retry_count)
        self.assertEqual(1, resumed.revision_count)
        self.assertEqual(2, len(resumed.evaluation_history))
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(2, len(llm.invoker.calls))
        self.assertEqual(2, len(evaluator.calls))

    def test_rejection_requires_feedback_and_is_recorded(self):
        review_required = passing_evaluation().model_copy(
            update={"decision": "human_review"}
        )
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Conditionally applicable. [[S1]]",
                    cited_source_ids=["S1"],
                )
            ],
            evaluator=FakeEvaluator(evaluation=review_required),
        )
        paused = agent.ask("Does this apply?", thread_id="review-reject")

        with self.assertRaisesRegex(ValueError, "feedback is required"):
            agent.resume_human_review(
                thread_id=paused.thread_id,
                approved=False,
            )
        with self.assertRaisesRegex(ValueError, "must agree"):
            agent.resume_human_review(
                thread_id=paused.thread_id,
                approved=True,
                action="revise",
                feedback="Revise it.",
            )
        resumed = agent.resume_human_review(
            thread_id=paused.thread_id,
            approved=False,
            feedback="The factual premise is incorrect.",
        )

        self.assertEqual("human_rejected", resumed.termination_reason)
        self.assertEqual("edit_question", resumed.human_review.action)
        self.assertEqual(1, len(resumed.human_review_history))
        self.assertEqual(
            "The factual premise is incorrect.", resumed.human_review.feedback
        )

    def test_human_rejection_can_request_grounded_revision(self):
        review_required = passing_evaluation().model_copy(
            update={
                "decision": "human_review",
                "brief_evaluation": "A human must confirm the presentation.",
            }
        )
        evaluator = FakeEvaluator(
            evaluation=[review_required, passing_evaluation()]
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(
                    answer="Human-feedback revision. [[S1]]",
                    cited_source_ids=["S1"],
                ),
            ],
            evaluator=evaluator,
        )
        paused = agent.ask("Does this apply?", thread_id="human-revise")

        resumed = agent.resume_human_review(
            thread_id=paused.thread_id,
            action="revise",
            feedback="State the limitation before the conclusion.",
        )

        self.assertEqual("answered", resumed.termination_reason)
        self.assertEqual("revise", resumed.human_review.action)
        self.assertEqual(1, resumed.evaluation_retry_count)
        self.assertEqual(1, resumed.revision_count)
        self.assertEqual(0, resumed.research_retry_count)
        self.assertTrue(resumed.validation.valid)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual([], guidance.calls)
        self.assertEqual([], web.calls)
        self.assertEqual(2, len(evaluator.calls))
        self.assertIn(
            "State the limitation before the conclusion.",
            llm.invoker.calls[1].to_string(),
        )

    def test_human_rejection_can_request_selective_research_retry(self):
        review_required = passing_evaluation().model_copy(
            update={
                "decision": "human_review",
                "brief_evaluation": "A human must decide whether more evidence is needed.",
            }
        )
        plan = ResearchRetryPlan(
            evidence_gaps=["Implementation recommendation"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA implementation recommendation",
                    reason="The reviewer requested additional guidance support.",
                )
            ],
        )
        evaluator = FakeEvaluator(
            evaluation=[review_required, passing_evaluation()]
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Rule. [[S1]] Recommendation. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=evaluator,
        )
        paused = agent.ask("What is required?", thread_id="human-research")

        resumed = agent.resume_human_review(
            thread_id=paused.thread_id,
            action="research_again",
            feedback="Find FDA guidance on practical implementation.",
        )

        self.assertEqual("answered", resumed.termination_reason)
        self.assertEqual("research_again", resumed.human_review.action)
        self.assertEqual(1, resumed.evaluation_retry_count)
        self.assertEqual(0, resumed.revision_count)
        self.assertEqual(1, resumed.research_retry_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(["FDA implementation recommendation"], guidance.calls)
        self.assertEqual([], web.calls)
        self.assertEqual(1, len(resumed.research_retry_diagnostics))
        self.assertEqual(
            "evaluation_passed", resumed.research_retry_diagnostics[0].outcome
        )
        self.assertIn(
            "Find FDA guidance on practical implementation.",
            llm.invoker.calls[1].to_string(),
        )

    def test_human_requested_retry_respects_existing_budget(self):
        review_required = passing_evaluation().model_copy(
            update={"decision": "human_review"}
        )
        agent, llm, regulation, guidance, web = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=FakeEvaluator(evaluation=review_required),
            retry_policy=EvaluationRetryPolicy(max_research_retries=0),
        )
        paused = agent.ask("What is required?", thread_id="human-research-limit")

        resumed = agent.resume_human_review(
            thread_id=paused.thread_id,
            action="research_again",
            feedback="Find more guidance.",
        )

        self.assertEqual("retry_limit_reached", resumed.termination_reason)
        self.assertEqual(0, resumed.evaluation_retry_count)
        self.assertEqual(0, resumed.research_retry_count)
        self.assertEqual(1, len(llm.invoker.calls))
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual([], guidance.calls)
        self.assertEqual([], web.calls)

    def test_always_policy_pauses_even_when_evaluator_passes(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Supported answer. [[S1]]",
                    cited_source_ids=["S1"],
                )
            ],
            evaluator=FakeEvaluator(evaluation=passing_evaluation()),
            human_review_policy="always",
        )

        result = agent.ask("What is required?", thread_id="always-review")

        self.assertEqual("awaiting_human_review", result.termination_reason)

    def test_validator_requires_contiguous_first_appearance_citation_list(self):
        sources = [
            SynthesisSource(
                source_id="S1",
                source_type="regulation",
                content="Requirement",
                citation="21 CFR § 50.25(a)",
            ),
            SynthesisSource(
                source_id="S2",
                source_type="guidance",
                content="Recommendation",
                citation="FDA Guidance, III.B",
            ),
        ]
        validation = validate_and_render_synthesis(
            FinalAnswerDraft(
                answer="Recommendation. [[S2]] Requirement. [[S1]]",
                cited_source_ids=["S1", "S2"],
            ),
            sources,
        )

        self.assertFalse(validation.valid)
        self.assertEqual("citation_list_mismatch", validation.issues[0].code)

    def test_graph_visualization_exposes_all_source_nodes_and_final_node(self):
        agent, _, _, _, _ = build_agent(targets=("web",), drafts=[])

        mermaid = agent.mermaid()

        for node in (
            "route_question",
            "regulation_agent",
            "guidance_agent",
            "web_agent",
            "generate_answer",
            "evaluate_answer",
            "revise_answer",
            "validate_revision",
            "prepare_research_retry",
            "validate_retry_plan",
            "execute_research_query",
            "merge_research_evidence",
            "regenerate_answer",
            "human_review",
            "handle_human_review",
        ):
            self.assertIn(node, mermaid)


if __name__ == "__main__":
    unittest.main()
