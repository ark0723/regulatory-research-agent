"""Phase 4A regression tests for run-isolated conversation lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier, Lock
import unittest
from uuid import UUID, uuid4

from regulatory_agent.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
    HumanReviewActionRecord,
    InMemoryConversationRepository,
    IdempotencyConflictError,
    InvalidReviewActionError,
    InvalidRunTransitionError,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    ReviewClaimConflictError,
    ReviewExecutionClaim,
    StaleReviewRequestError,
    _now,
)
from regulatory_agent.multi_agent import (
    FinalAnswerDraft,
    ResearchRetryPlan,
    TargetedResearchQuery,
)
from tests.test_multi_agent import FakeEvaluator, build_agent, passing_evaluation


def human_review_evaluation():
    return passing_evaluation().model_copy(
        update={
            "decision": "human_review",
            "brief_evaluation": "A human must confirm this answer.",
        }
    )


class CountingAgent:
    """Count resume calls while preserving the real graph implementation."""

    def __init__(self, agent):
        self.agent = agent
        self.resume_calls = 0
        self._lock = Lock()

    def ask(self, question, *, thread_id=None):
        return self.agent.ask(question, thread_id=thread_id)

    def resume_human_review(
        self, *, thread_id, approved=None, feedback="", action=None
    ):
        with self._lock:
            self.resume_calls += 1
        return self.agent.resume_human_review(
            thread_id=thread_id,
            approved=approved,
            feedback=feedback,
            action=action,
        )


class UnsafeDirectThreadReuseCharacterizationTests(unittest.TestCase):
    """Document why application callers must not reuse graph thread IDs."""

    def test_reusing_completed_thread_accumulates_cross_route_evidence(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="First rule. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(
                    answer="Old rule. [[S1]] New context. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
        )

        first = agent.ask("First question", thread_id="reused-thread")
        agent.router.targets = ("web",)
        second = agent.ask("Second question", thread_id="reused-thread")

        self.assertEqual(["regulation"], [item.target for item in first.contributions])
        self.assertEqual(["web"], [item.target for item in second.decision.routes])
        self.assertEqual(
            ["regulation", "web"],
            [source.source_type for source in second.sources],
        )
        self.assertIn("21 CFR § 50.25(a)", second.citations)

    def test_new_ask_on_pending_thread_supersedes_the_visible_interrupt(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="First. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(
                    answer="First and second. [[S1]]", cited_source_ids=["S1"]
                ),
            ],
            evaluator=FakeEvaluator(
                evaluation=[human_review_evaluation(), human_review_evaluation()]
            ),
        )

        first = agent.ask("First question", thread_id="pending-thread")
        second = agent.ask("Second question", thread_id="pending-thread")
        snapshot = agent.graph.get_state(
            {"configurable": {"thread_id": "pending-thread"}}
        )

        self.assertEqual("First question", first.human_review_request.question)
        self.assertEqual("Second question", second.human_review_request.question)
        self.assertEqual(2, len(snapshot.values["contributions"]))
        self.assertEqual(1, len(snapshot.interrupts))

    def test_non_answering_reuse_exposes_stale_validation_and_evaluation(self):
        agent, _, regulation, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Grounded. [[S1]]", cited_source_ids=["S1"])
            ],
        )

        first = agent.ask("First question", thread_id="stale-evaluation-thread")
        regulation.error = RuntimeError("database unavailable")
        second = agent.ask("Second question", thread_id="stale-evaluation-thread")

        self.assertEqual("answered", first.termination_reason)
        self.assertEqual("insufficient_evidence", second.termination_reason)
        self.assertTrue(second.validation.valid)
        self.assertEqual("pass", second.evaluation.decision)


class ConversationServiceTests(unittest.TestCase):
    def test_each_question_gets_a_unique_isolated_run_and_thread(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        evaluator = FakeEvaluator(
            evaluation=[revise, passing_evaluation(), passing_evaluation()]
        )
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Initial rule. [[S1]]", cited_source_ids=["S1"]
                ),
                FinalAnswerDraft(
                    answer="Revised rule. [[S1]]", cited_source_ids=["S1"]
                ),
                FinalAnswerDraft(answer="Web only. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=evaluator,
        )
        service = ConversationService(agent=agent)
        conversation = service.create_conversation()

        first = service.start_run(conversation.conversation_id, "First question")
        agent.router.targets = ("web",)
        second = service.start_run(conversation.conversation_id, "Second question")

        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(str(first.run_id), first.thread_id)
        self.assertEqual(str(second.run_id), second.thread_id)
        self.assertEqual(RunStatus.COMPLETED, first.status)
        self.assertEqual(RunStatus.COMPLETED, second.status)
        self.assertEqual(["regulation"], first.source_types)
        self.assertEqual(["web"], second.source_types)
        self.assertEqual(["21 CFR § 50.25(a)"], first.citations)
        self.assertEqual(
            ["[FDA Update](https://www.fda.gov/example)"], second.citations
        )
        self.assertEqual(1, first.revision_count)
        self.assertEqual(1, first.evaluation_retry_count)
        self.assertEqual(0, second.revision_count)
        self.assertEqual(0, second.evaluation_retry_count)
        self.assertEqual(0, second.research_retry_count)
        self.assertEqual("pass", second.evaluation_decision)

    def test_edit_question_closes_run_and_new_question_stays_in_conversation(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(
                    answer="Needs review. [[S1]]", cited_source_ids=["S1"]
                ),
                FinalAnswerDraft(
                    answer="Edited question. [[S1]]", cited_source_ids=["S1"]
                ),
            ],
            evaluator=FakeEvaluator(
                evaluation=[human_review_evaluation(), passing_evaluation()]
            ),
        )
        service = ConversationService(agent=agent)
        conversation = service.create_conversation()
        first = service.start_run(conversation.conversation_id, "Original question")
        request_id = first.pending_review.review_request_id

        closed = service.submit_human_review(
            first.run_id,
            "edit_question",
            "The question needs a material correction.",
            "edit-1",
            request_id,
        )
        second = service.start_run(conversation.conversation_id, "Edited question")

        self.assertEqual(RunStatus.CLOSED_FOR_EDIT, closed.status)
        self.assertEqual("human_rejected", closed.termination_reason)
        self.assertEqual(RunStatus.COMPLETED, second.status)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(0, second.revision_count)
        self.assertEqual(0, second.research_retry_count)
        messages = service.list_conversation_messages(conversation.conversation_id)
        self.assertEqual(
            [first.run_id, second.run_id],
            list(dict.fromkeys(message.run_id for message in messages)),
        )
        self.assertIn("human_feedback", [message.kind for message in messages])
        statuses = [m.content for m in messages if m.kind == "run_status"]
        self.assertEqual("closed_for_edit", statuses[1])

    def test_unknown_ids_and_invalid_transitions_are_explicit(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
        )
        service = ConversationService(agent=agent)

        with self.assertRaises(ConversationNotFoundError):
            service.start_run(uuid4(), "Question")
        with self.assertRaises(RunNotFoundError):
            service.get_run(uuid4())

        conversation = service.create_conversation()
        completed = service.start_run(conversation.conversation_id, "Question")
        with self.assertRaises(InvalidRunTransitionError):
            service.submit_human_review(
                completed.run_id,
                "approve",
                "",
                "late-approval",
                uuid4(),
            )
        with self.assertRaises(InvalidReviewActionError):
            service.submit_human_review(
                completed.run_id,
                "delete",  # type: ignore[arg-type]
                "No",
                "invalid-action",
                uuid4(),
            )

    def test_duplicate_review_submission_is_idempotent(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
        )
        counting = CountingAgent(agent)
        service = ConversationService(agent=counting)
        conversation = service.create_conversation()
        pending = service.start_run(conversation.conversation_id, "Question")
        request_id = pending.pending_review.review_request_id

        first = service.submit_human_review(
            pending.run_id, "approve", "", "approve-once", request_id
        )
        duplicate = service.submit_human_review(
            pending.run_id, "approve", "", "approve-once", request_id
        )

        self.assertEqual(RunStatus.COMPLETED, first.status)
        self.assertEqual(first, duplicate)
        self.assertEqual(1, counting.resume_calls)
        self.assertEqual(1, len(first.review_history))
        with self.assertRaises(IdempotencyConflictError):
            service.submit_human_review(
                pending.run_id,
                "edit_question",
                "Different command.",
                "approve-once",
                request_id,
            )

    def test_concurrent_review_submissions_apply_only_once(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
        )
        counting = CountingAgent(agent)
        service = ConversationService(agent=counting)
        conversation = service.create_conversation()
        pending = service.start_run(conversation.conversation_id, "Question")
        request_id = pending.pending_review.review_request_id
        barrier = Barrier(2)

        def submit(key):
            barrier.wait()
            try:
                return service.submit_human_review(
                    pending.run_id, "approve", "", key, request_id
                )
            except Exception as error:  # asserted below
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(submit, ("browser-a", "browser-b")))

        self.assertEqual(1, sum(hasattr(item, "status") for item in outcomes))
        self.assertEqual(
            1,
            sum(
                isinstance(
                    item, (StaleReviewRequestError, ReviewClaimConflictError)
                )
                for item in outcomes
            ),
        )
        self.assertEqual(1, counting.resume_calls)
        self.assertEqual(1, len(service.get_run(pending.run_id).review_history))

    def test_separate_service_instances_share_review_claim(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
        )
        counting = CountingAgent(agent)
        repository = InMemoryConversationRepository()
        first_service = ConversationService(
            agent=counting, repository=repository, worker_id="worker-a"
        )
        second_service = ConversationService(
            agent=counting, repository=repository, worker_id="worker-b"
        )
        conversation = first_service.create_conversation()
        pending = first_service.start_run(conversation.conversation_id, "Question")
        request_id = pending.pending_review.review_request_id
        barrier = Barrier(2)

        def submit(service_and_key):
            service, key = service_and_key
            barrier.wait()
            try:
                return service.submit_human_review(
                    pending.run_id, "approve", "", key, request_id
                )
            except Exception as error:  # asserted below
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(
                    submit,
                    ((first_service, "key-a"), (second_service, "key-b")),
                )
            )

        self.assertEqual(1, sum(hasattr(item, "status") for item in outcomes))
        self.assertEqual(1, counting.resume_calls)

    def test_expired_unadvanced_claim_restores_same_review_request(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
        )
        repository = InMemoryConversationRepository()
        service = ConversationService(agent=agent, repository=repository)
        conversation = service.create_conversation()
        pending = service.start_run(conversation.conversation_id, "Question")
        request_id = pending.pending_review.review_request_id
        now = _now()
        action = HumanReviewActionRecord(
            review_action_id=uuid4(),
            review_request_id=request_id,
            action="approve",
            feedback="",
            idempotency_key="abandoned",
            created_at=now,
        )
        claim = ReviewExecutionClaim(
            claim_id=uuid4(),
            review_request_id=request_id,
            owner_id="dead-worker",
            action="approve",
            feedback="",
            idempotency_key="abandoned",
            claimed_at=now - timedelta(minutes=2),
            expires_at=now - timedelta(minutes=1),
        )
        repository.claim_review(
            run_id=pending.run_id,
            expected_review_request_id=request_id,
            action_record=action,
            claim=claim,
        )

        reconciled = service.reconcile_run(pending.run_id)

        self.assertEqual(RunStatus.AWAITING_HUMAN_REVIEW, reconciled.status)
        self.assertIsNone(reconciled.review_claim)
        self.assertEqual(request_id, reconciled.pending_review.review_request_id)
        messages = service.list_conversation_messages(conversation.conversation_id)
        self.assertEqual(
            1,
            sum(
                message.kind == "run_status"
                and message.content == "awaiting_human_review"
                for message in messages
            ),
        )

    def test_reconciliation_completes_advanced_claim_and_persists_receipt(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
        )
        repository = InMemoryConversationRepository()
        service = ConversationService(agent=agent, repository=repository)
        conversation = service.create_conversation()
        pending = service.start_run(conversation.conversation_id, "Question")
        request_id = pending.pending_review.review_request_id
        now = _now()
        action = HumanReviewActionRecord(
            review_action_id=uuid4(),
            review_request_id=request_id,
            action="approve",
            feedback="",
            idempotency_key="completed-before-metadata",
            created_at=now,
        )
        claim = ReviewExecutionClaim(
            claim_id=uuid4(),
            review_request_id=request_id,
            owner_id="crashed-worker",
            action="approve",
            feedback="",
            idempotency_key="completed-before-metadata",
            claimed_at=now - timedelta(minutes=2),
            expires_at=now - timedelta(minutes=1),
        )
        repository.claim_review(
            run_id=pending.run_id,
            expected_review_request_id=request_id,
            action_record=action,
            claim=claim,
        )
        graph_result = agent.resume_human_review(
            thread_id=pending.thread_id,
            action="approve",
            feedback="",
        )
        self.assertEqual("human_approved", graph_result.termination_reason)

        reconciled = service.reconcile_run(pending.run_id)
        replayed = service.submit_human_review(
            pending.run_id,
            "approve",
            "",
            "completed-before-metadata",
            request_id,
        )

        self.assertEqual(RunStatus.COMPLETED, reconciled.status)
        self.assertEqual(reconciled, replayed)

    def test_active_claim_is_not_recovered_or_replayed(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
        )
        repository = InMemoryConversationRepository()
        service = ConversationService(agent=agent, repository=repository)
        conversation = service.create_conversation()
        pending = service.start_run(conversation.conversation_id, "Question")
        request_id = pending.pending_review.review_request_id
        now = _now()
        repository.claim_review(
            run_id=pending.run_id,
            expected_review_request_id=request_id,
            action_record=HumanReviewActionRecord(
                review_action_id=uuid4(),
                review_request_id=request_id,
                action="approve",
                feedback="",
                idempotency_key="active",
                created_at=now,
            ),
            claim=ReviewExecutionClaim(
                claim_id=uuid4(),
                review_request_id=request_id,
                owner_id="live-worker",
                action="approve",
                feedback="",
                idempotency_key="active",
                claimed_at=now,
                expires_at=now + timedelta(minutes=5),
            ),
        )

        with self.assertRaises(ReviewClaimConflictError):
            service.reconcile_run(pending.run_id)

    def test_reconciliation_repairs_metadata_from_persisted_checkpoint(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
        )
        repository = InMemoryConversationRepository()
        service = ConversationService(agent=agent, repository=repository)
        conversation = service.create_conversation()
        pending = service.start_run(conversation.conversation_id, "Question")
        simulated_crash = pending.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "pending_review": None,
                "version": pending.version + 1,
            }
        )
        repository.save_run(simulated_crash)

        recovered = service.reconcile_run(pending.run_id)
        repeated = service.reconcile_run(pending.run_id)

        self.assertEqual(RunStatus.AWAITING_HUMAN_REVIEW, recovered.status)
        self.assertIsNotNone(recovered.pending_review)
        self.assertEqual(recovered, repeated)

    def test_reconciliation_never_replays_run_without_checkpoint(self):
        agent, _, _, _, _ = build_agent(targets=("regulation",), drafts=[])
        repository = InMemoryConversationRepository()
        service = ConversationService(agent=agent, repository=repository)
        conversation = service.create_conversation()
        now = _now()
        run = RunRecord(
            run_id=uuid4(),
            conversation_id=conversation.conversation_id,
            thread_id=str(uuid4()),
            question="Question",
            status=RunStatus.CREATED,
            created_at=now,
            updated_at=now,
        )
        repository.create_run(run)

        failed = service.reconcile_run(run.run_id)
        repeated = service.reconcile_run(run.run_id)

        self.assertEqual(RunStatus.FAILED, failed.status)
        self.assertIn("reconciliation_no_checkpoint", failed.error)
        self.assertEqual(failed, repeated)

    def test_revision_creates_new_review_identity_and_rejects_old_action(self):
        evaluator = FakeEvaluator(
            evaluation=[human_review_evaluation(), human_review_evaluation()]
        )
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Revised. [[S1]]", cited_source_ids=["S1"]),
            ],
            evaluator=evaluator,
        )
        service = ConversationService(agent=agent)
        conversation = service.create_conversation()
        first = service.start_run(conversation.conversation_id, "Question")
        first_request = first.pending_review.review_request_id

        revised = service.submit_human_review(
            first.run_id,
            "revise",
            "Put the limitation first.",
            "revise-1",
            first_request,
        )
        second_request = revised.pending_review.review_request_id

        self.assertEqual(RunStatus.AWAITING_HUMAN_REVIEW, revised.status)
        self.assertNotEqual(first_request, second_request)
        self.assertEqual(1, revised.revision_count)
        with self.assertRaises(StaleReviewRequestError):
            service.submit_human_review(
                first.run_id, "approve", "", "stale-click", first_request
            )
        approved = service.submit_human_review(
            first.run_id, "approve", "", "approve-revision", second_request
        )
        self.assertEqual(RunStatus.COMPLETED, approved.status)
        self.assertEqual(2, len(approved.review_history))

    def test_research_retry_creates_new_review_identity_and_preserves_behavior(self):
        plan = ResearchRetryPlan(
            evidence_gaps=["Implementation recommendation"],
            queries=[
                TargetedResearchQuery(
                    target="guidance",
                    query="FDA implementation recommendation",
                    reason="The reviewer requested additional guidance.",
                )
            ],
        )
        agent, _, regulation, guidance, _ = build_agent(
            targets=("regulation",),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
                plan,
                FinalAnswerDraft(
                    answer="Rule. [[S1]] Guidance. [[S2]]",
                    cited_source_ids=["S1", "S2"],
                ),
            ],
            evaluator=FakeEvaluator(
                evaluation=[human_review_evaluation(), human_review_evaluation()]
            ),
        )
        service = ConversationService(agent=agent)
        conversation = service.create_conversation()
        first = service.start_run(conversation.conversation_id, "Question")
        first_request = first.pending_review.review_request_id

        researched = service.submit_human_review(
            first.run_id,
            "research_again",
            "Find implementation guidance.",
            "research-1",
            first_request,
        )

        self.assertEqual(RunStatus.AWAITING_HUMAN_REVIEW, researched.status)
        self.assertNotEqual(
            first_request, researched.pending_review.review_request_id
        )
        self.assertEqual(1, researched.research_retry_count)
        self.assertEqual(1, len(regulation.calls))
        self.assertEqual(["FDA implementation recommendation"], guidance.calls)

    def test_unsuccessful_graph_outcome_is_failed_not_completed(self):
        agent, _, _, _, _ = build_agent(
            targets=("web",), drafts=[], web_empty=True
        )
        service = ConversationService(agent=agent)
        conversation = service.create_conversation()

        run = service.start_run(conversation.conversation_id, "Unknown current fact")

        self.assertEqual(RunStatus.FAILED, run.status)
        self.assertEqual("insufficient_evidence", run.termination_reason)
        self.assertIn("did not find sufficient evidence", run.latest_answer)

    def test_conversation_listing_and_history_are_repository_owned(self):
        agent, _, _, _, _ = build_agent(
            targets=("regulation",),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
        )
        service = ConversationService(agent=agent)
        conversation = service.create_conversation()
        run = service.start_run(conversation.conversation_id, "Question")

        self.assertEqual(
            [conversation.conversation_id],
            [item.conversation_id for item in service.list_conversations()],
        )
        messages = service.list_conversation_messages(conversation.conversation_id)
        self.assertEqual(
            ["question", "answer", "run_status"],
            [message.kind for message in messages],
        )
        self.assertTrue(all(message.run_id == run.run_id for message in messages))
        self.assertEqual(run.citations, messages[1].citations)
        self.assertTrue(all(isinstance(message.run_id, UUID) for message in messages))


if __name__ == "__main__":
    unittest.main()
