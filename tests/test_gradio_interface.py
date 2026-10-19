"""Deterministic service-contract tests for the Gradio MVP."""

from __future__ import annotations

import unittest

from regulatory_agent.conversation_service import ConversationService, RunStatus
from regulatory_agent.gradio_interface import (
    GradioController,
    WebSessionState,
    create_gradio_app,
)
from regulatory_agent.multi_agent import (
    FinalAnswerDraft,
    ResearchRetryPlan,
    TargetedResearchQuery,
)
from tests.test_conversation_service import human_review_evaluation
from tests.test_multi_agent import FakeEvaluator, build_agent, passing_evaluation


def completed_service(*answers: str) -> ConversationService:
    agent, _, _, _, _ = build_agent(
        targets=("regulation",),
        drafts=[
            FinalAnswerDraft(answer=f"{answer} [[S1]]", cited_source_ids=["S1"])
            for answer in answers
        ],
        evaluator=FakeEvaluator(
            evaluation=[passing_evaluation() for _ in answers]
        ),
    )
    return ConversationService(agent=agent)


def review_service(*, revisions: int = 0) -> ConversationService:
    count = revisions + 1
    agent, _, _, _, _ = build_agent(
        targets=("regulation",),
        drafts=[
            FinalAnswerDraft(
                answer=f"Answer {index}. [[S1]]", cited_source_ids=["S1"]
            )
            for index in range(count)
        ],
        evaluator=FakeEvaluator(
            evaluation=[human_review_evaluation() for _ in range(count)]
        ),
    )
    return ConversationService(agent=agent)


class RecordingService:
    def __init__(self, service, *, fail_first_review: bool = False) -> None:
        self.service = service
        self.fail_first_review = fail_first_review
        self.review_calls = []

    def __getattr__(self, name):
        return getattr(self.service, name)

    def submit_human_review(self, **kwargs):
        self.review_calls.append(dict(kwargs))
        if self.fail_first_review and len(self.review_calls) == 1:
            raise RuntimeError(
                "postgresql://private-user:private-password@db.internal/app"
            )
        return self.service.submit_human_review(**kwargs)


class FailingRunService:
    def __init__(self, service) -> None:
        self.service = service

    def __getattr__(self, name):
        return getattr(self.service, name)

    def start_run(self, conversation_id, question):
        raise RuntimeError(
            "postgresql://private-user:private-password@db.internal/app"
        )


class GradioControllerTests(unittest.TestCase):
    def test_creates_selects_conversation_and_new_run_per_question(self):
        service = completed_service("First", "Second")
        controller = GradioController(service)
        created = controller.create_conversation()

        first = controller.submit_question("First question", created.session)
        second = controller.submit_question("Second question", first.session)
        messages = service.list_conversation_messages(
            created.session.conversation_id
        )
        question_runs = [
            message.run_id for message in messages if message.kind == "question"
        ]

        self.assertEqual(2, len(question_runs))
        self.assertNotEqual(question_runs[0], question_runs[1])
        self.assertIn("Second", second.chat_messages[-2]["content"])
        selected = controller.select_conversation(
            str(created.session.conversation_id), WebSessionState()
        )
        self.assertEqual(created.session.conversation_id, selected.session.conversation_id)

    def test_history_renders_answers_citations_and_status_from_messages(self):
        service = completed_service("Grounded answer")
        controller = GradioController(service)
        created = controller.create_conversation()
        view = controller.submit_question("Question", created.session)

        transcript = "\n".join(item["content"] for item in view.chat_messages)
        self.assertIn("Grounded answer", transcript)
        self.assertIn("21 CFR § 50.25(a)", transcript)
        self.assertIn("completed", transcript)
        self.assertIn("regulation", view.status_markdown)

    def test_pending_run_prevents_another_question(self):
        service = review_service()
        controller = GradioController(service)
        created = controller.create_conversation()
        pending = controller.submit_question("First question", created.session)
        blocked = controller.submit_question("Second question", pending.session)

        messages = service.list_conversation_messages(
            created.session.conversation_id
        )
        self.assertTrue(pending.review_visible)
        self.assertIn("Complete the active run", blocked.error_markdown)
        self.assertEqual(1, sum(item.kind == "question" for item in messages))

    def test_review_action_mapping_and_new_request_after_revision(self):
        wrapped = RecordingService(review_service(revisions=1))
        controller = GradioController(wrapped)
        created = controller.create_conversation()
        pending = controller.submit_question("Question", created.session)
        old_request = pending.session.pending_review_request_id

        revised = controller.submit_review(
            "revise", "Put the limitation first.", pending.session
        )

        self.assertEqual("revise", wrapped.review_calls[0]["action"])
        self.assertEqual(
            "Put the limitation first.", wrapped.review_calls[0]["feedback"]
        )
        self.assertTrue(revised.review_visible)
        self.assertNotEqual(old_request, revised.session.pending_review_request_id)

    def test_approve_and_edit_question_behaviors(self):
        approve_service = review_service()
        approve_controller = GradioController(approve_service)
        created = approve_controller.create_conversation()
        pending = approve_controller.submit_question("Approve me", created.session)
        approved = approve_controller.submit_review("approve", "", pending.session)
        self.assertFalse(approved.review_visible)
        self.assertIn("completed", approved.status_markdown)

        edit_agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Edited. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=FakeEvaluator(
                evaluation=[human_review_evaluation(), passing_evaluation()]
            ),
        )
        edit_service = ConversationService(agent=edit_agent)
        edit_controller = GradioController(edit_service)
        created = edit_controller.create_conversation()
        pending = edit_controller.submit_question("Original question", created.session)
        closed = edit_controller.submit_review(
            "edit_question", "The scope needs correction.", pending.session
        )
        self.assertEqual("Original question", closed.question_value)
        self.assertTrue(closed.question_interactive)
        self.assertIn("closed_for_edit", closed.status_markdown)
        edited = edit_controller.submit_question(
            "Edited question", closed.session
        )
        questions = [
            item
            for item in edit_service.list_conversation_messages(
                created.session.conversation_id
            )
            if item.kind == "question"
        ]
        self.assertEqual(2, len(questions))
        self.assertNotEqual(questions[0].run_id, questions[1].run_id)
        self.assertIn("completed", edited.status_markdown)

    def test_research_again_maps_to_existing_phase3_path(self):
        plan = ResearchRetryPlan(
            evidence_gaps=["Implementation recommendation"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA implementation recommendation",
                    reason="The reviewer requested implementation evidence.",
                )
            ],
        )
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Regulation. [[S1]] Guidance. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(
                evaluation=[human_review_evaluation(), human_review_evaluation()]
            ),
        )
        wrapped = RecordingService(ConversationService(agent=agent))
        controller = GradioController(wrapped)
        created = controller.create_conversation()
        pending = controller.submit_question("Question", created.session)

        researched = controller.submit_review(
            "research_again", "Find implementation guidance.", pending.session
        )

        self.assertEqual("research_again", wrapped.review_calls[0]["action"])
        self.assertTrue(researched.review_visible)
        run = wrapped.get_run(researched.session.active_run_id)
        self.assertEqual(1, run.research_retry_count)

    def test_stale_review_refreshes_to_current_request(self):
        service = review_service(revisions=1)
        controller = GradioController(service)
        created = controller.create_conversation()
        pending = controller.submit_question("Question", created.session)
        stale_session = pending.session.model_copy(deep=True)
        revised = controller.submit_review("revise", "Improve it.", pending.session)

        stale = controller.submit_review("approve", "", stale_session)

        self.assertIn("review request changed", stale.error_markdown)
        self.assertEqual(
            revised.session.pending_review_request_id,
            stale.session.pending_review_request_id,
        )

    def test_review_retry_reuses_idempotency_key(self):
        wrapped = RecordingService(review_service(), fail_first_review=True)
        controller = GradioController(wrapped)
        created = controller.create_conversation()
        pending = controller.submit_question("Question", created.session)

        failed = controller.submit_review("approve", "", pending.session)
        succeeded = controller.submit_review("approve", "", failed.session)

        self.assertIn("could not be completed", failed.error_markdown)
        self.assertEqual(
            wrapped.review_calls[0]["idempotency_key"],
            wrapped.review_calls[1]["idempotency_key"],
        )
        self.assertIn("completed", succeeded.status_markdown)

    def test_refresh_reconstructs_persisted_history_with_fresh_session(self):
        service = completed_service("Persisted")
        first_controller = GradioController(service)
        created = first_controller.create_conversation()
        original = first_controller.submit_question("Question", created.session)

        recreated_service = ConversationService(
            agent=service.agent,
            repository=service.repository,
        )
        restored = GradioController(recreated_service).initialize(WebSessionState())

        self.assertEqual(created.session.conversation_id, restored.session.conversation_id)
        self.assertEqual(original.chat_messages, restored.chat_messages)

    def test_unexpected_errors_do_not_expose_credentials_or_tracebacks(self):
        service = FailingRunService(completed_service("Unused"))
        controller = GradioController(service)
        created = controller.create_conversation()

        failed = controller.submit_question("Question", created.session)

        self.assertIn("could not be completed", failed.error_markdown)
        self.assertNotIn("private-password", failed.error_markdown)
        self.assertNotIn("Traceback", failed.error_markdown)

    def test_gradio_blocks_build_without_redefining_the_graph(self):
        service = completed_service("Unused")
        app = create_gradio_app(service=service)
        try:
            self.assertEqual(service, app.controller.service)
            self.assertIsNone(app._service_context)
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
