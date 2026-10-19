"""PostgreSQL integration tests; requires a dedicated test database."""

from concurrent.futures import ThreadPoolExecutor
import os
from threading import Barrier
import unittest
from urllib.parse import urlparse

from langgraph.checkpoint.postgres import PostgresSaver

from regulatory_agent.conversation_service import (
    ConversationService,
    ReviewClaimConflictError,
    RunStatus,
    StaleReviewRequestError,
)
from regulatory_agent.multi_agent import (
    FinalAnswerDraft,
    RegulatoryMultiAgent,
    create_checkpoint_serializer,
)
from regulatory_agent.persistence import (
    _configure_checkpoint_connection,
    setup_postgres_persistence,
)
from regulatory_agent.postgres_repository import (
    APPLICATION_SCHEMA,
    PostgresConversationRepository,
    create_connection_pool,
)
from tests.test_conversation_service import human_review_evaluation
from tests.test_multi_agent import FakeEvaluator, build_agent, passing_evaluation


TEST_DATABASE_URL = os.getenv("REGULATORY_TEST_DATABASE_URL")


def _persistent_agent(checkpointer, *, evaluator, drafts):
    base, _, _, _, _ = build_agent(
        targets=("regulation",),
        drafts=drafts,
        evaluator=evaluator,
    )
    return RegulatoryMultiAgent(
        llm=base.llm,
        router=base.router,
        regulation_agent=base.regulation_agent,
        guidance_agent=base.guidance_agent,
        web_agent=base.web_agent,
        evaluator=base.evaluator,
        human_review_policy=base.human_review_policy,
        retry_policy=base.retry_policy,
        available_research_targets=base.available_research_targets,
        checkpointer=checkpointer,
    )


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "set REGULATORY_TEST_DATABASE_URL to a dedicated PostgreSQL test database",
)
class PostgresPersistenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database_name = urlparse(TEST_DATABASE_URL).path.rsplit("/", 1)[-1]
        if "test" not in database_name.lower():
            raise unittest.SkipTest(
                "REGULATORY_TEST_DATABASE_URL database name must contain 'test'"
            )
        setup_postgres_persistence(TEST_DATABASE_URL)
        cls.app_pool = create_connection_pool(TEST_DATABASE_URL)
        cls.checkpoint_pool = create_connection_pool(
            TEST_DATABASE_URL,
            configure=_configure_checkpoint_connection,
        )
        cls.repository = PostgresConversationRepository(cls.app_pool)
        cls.checkpointer = PostgresSaver(
            cls.checkpoint_pool,
            serde=create_checkpoint_serializer(),
        )
        cls.conversation_ids = []
        cls.thread_ids = []

    @classmethod
    def tearDownClass(cls):
        for thread_id in cls.thread_ids:
            cls.checkpointer.delete_thread(thread_id)
        if cls.conversation_ids:
            with cls.app_pool.connection() as conn, conn.transaction():
                conn.execute(
                    f"""
                    DELETE FROM {APPLICATION_SCHEMA}.review_submission_receipts
                    WHERE run_id IN (
                        SELECT run_id FROM {APPLICATION_SCHEMA}.runs
                        WHERE conversation_id = ANY(%s)
                    )
                    """,
                    (cls.conversation_ids,),
                )
                conn.execute(
                    f"""
                    DELETE FROM {APPLICATION_SCHEMA}.human_review_actions
                    WHERE run_id IN (
                        SELECT run_id FROM {APPLICATION_SCHEMA}.runs
                        WHERE conversation_id = ANY(%s)
                    )
                    """,
                    (cls.conversation_ids,),
                )
                conn.execute(
                    f"""
                    DELETE FROM {APPLICATION_SCHEMA}.conversation_messages
                    WHERE conversation_id = ANY(%s)
                    """,
                    (cls.conversation_ids,),
                )
                conn.execute(
                    f"""DELETE FROM {APPLICATION_SCHEMA}.runs
                    WHERE conversation_id = ANY(%s)""",
                    (cls.conversation_ids,),
                )
                conn.execute(
                    f"""DELETE FROM {APPLICATION_SCHEMA}.conversations
                    WHERE conversation_id = ANY(%s)""",
                    (cls.conversation_ids,),
                )
        cls.checkpoint_pool.close()
        cls.app_pool.close()

    def _remember(self, conversation, run):
        self.conversation_ids.append(conversation.conversation_id)
        self.thread_ids.append(run.thread_id)

    def test_metadata_and_messages_survive_service_recreation(self):
        agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=passing_evaluation()),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
        )
        first = ConversationService(agent=agent, repository=self.repository)
        conversation = first.create_conversation()
        run = first.start_run(conversation.conversation_id, "Question")
        self._remember(conversation, run)

        recreated_repository = PostgresConversationRepository(self.app_pool)
        recreated = ConversationService(
            agent=agent, repository=recreated_repository
        )

        self.assertEqual(run, recreated.get_run(run.run_id))
        self.assertEqual(
            ["question", "answer", "run_status"],
            [
                message.kind
                for message in recreated.list_conversation_messages(
                    conversation.conversation_id
                )
            ],
        )

    def test_human_review_resumes_after_agent_recreation(self):
        first_agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
        )
        first = ConversationService(agent=first_agent, repository=self.repository)
        conversation = first.create_conversation()
        pending = first.start_run(conversation.conversation_id, "Question")
        self._remember(conversation, pending)

        second_agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=passing_evaluation()),
            drafts=[],
        )
        second = ConversationService(agent=second_agent, repository=self.repository)
        completed = second.submit_human_review(
            pending.run_id,
            "approve",
            "",
            "restart-approval",
            pending.pending_review.review_request_id,
        )

        self.assertEqual(RunStatus.COMPLETED, completed.status)
        self.assertEqual("human_approved", completed.termination_reason)
        self.assertEqual(1, len(completed.review_history))
        replayed = second.submit_human_review(
            pending.run_id,
            "approve",
            "",
            "restart-approval",
            pending.pending_review.review_request_id,
        )
        self.assertEqual(completed, replayed)

    def test_retry_and_evaluation_history_survive_agent_recreation(self):
        revise = passing_evaluation().model_copy(update={"decision": "revise"})
        first_agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=[revise, passing_evaluation()]),
            drafts=[
                FinalAnswerDraft(answer="Initial. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Revised. [[S1]]", cited_source_ids=["S1"]),
            ],
        )
        service = ConversationService(
            agent=first_agent, repository=self.repository
        )
        conversation = service.create_conversation()
        run = service.start_run(conversation.conversation_id, "Question")
        self._remember(conversation, run)

        recreated_agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=passing_evaluation()),
            drafts=[],
        )
        snapshot = recreated_agent.inspect_thread(thread_id=run.thread_id)

        self.assertEqual(1, snapshot.result.revision_count)
        self.assertEqual(1, snapshot.result.evaluation_retry_count)
        self.assertEqual(2, len(snapshot.result.evaluation_history))
        self.assertTrue(snapshot.result.validation.valid)
        self.assertEqual("pass", snapshot.result.evaluation.decision)

    def test_concurrent_workers_resume_only_once(self):
        first_agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
        )
        first = ConversationService(
            agent=first_agent, repository=self.repository, worker_id="worker-a"
        )
        conversation = first.create_conversation()
        pending = first.start_run(conversation.conversation_id, "Question")
        self._remember(conversation, pending)
        second_agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=passing_evaluation()),
            drafts=[],
        )
        second = ConversationService(
            agent=second_agent, repository=self.repository, worker_id="worker-b"
        )
        barrier = Barrier(2)

        def submit(service_and_key):
            service, key = service_and_key
            barrier.wait()
            try:
                return service.submit_human_review(
                    pending.run_id,
                    "approve",
                    "",
                    key,
                    pending.pending_review.review_request_id,
                )
            except Exception as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(submit, ((first, "worker-a"), (second, "worker-b")))
            )

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

    def test_checkpoint_threads_are_isolated_per_run(self):
        agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(
                evaluation=[passing_evaluation(), passing_evaluation()]
            ),
            drafts=[
                FinalAnswerDraft(answer="First. [[S1]]", cited_source_ids=["S1"]),
                FinalAnswerDraft(answer="Second. [[S1]]", cited_source_ids=["S1"]),
            ],
        )
        service = ConversationService(agent=agent, repository=self.repository)
        conversation = service.create_conversation()
        first = service.start_run(conversation.conversation_id, "First question")
        second = service.start_run(conversation.conversation_id, "Second question")
        self._remember(conversation, first)
        self.thread_ids.append(second.thread_id)

        first_snapshot = agent.inspect_thread(thread_id=first.thread_id)
        second_snapshot = agent.inspect_thread(thread_id=second.thread_id)

        self.assertNotEqual(first.thread_id, second.thread_id)
        self.assertEqual("First question", first_snapshot.result.question)
        self.assertEqual("Second question", second_snapshot.result.question)
        self.assertEqual(1, len(first_snapshot.result.contributions))
        self.assertEqual(1, len(second_snapshot.result.contributions))

    def test_reconciliation_recovers_checkpoint_after_metadata_gap(self):
        agent = _persistent_agent(
            self.checkpointer,
            evaluator=FakeEvaluator(evaluation=human_review_evaluation()),
            drafts=[FinalAnswerDraft(answer="Rule. [[S1]]", cited_source_ids=["S1"])],
        )
        service = ConversationService(agent=agent, repository=self.repository)
        conversation = service.create_conversation()
        pending = service.start_run(conversation.conversation_id, "Question")
        self._remember(conversation, pending)
        gap = pending.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "pending_review": None,
                "version": pending.version + 1,
            }
        )
        self.repository.save_run(gap)

        recovered = service.reconcile_run(pending.run_id)
        repeated = service.reconcile_run(pending.run_id)

        self.assertEqual(RunStatus.AWAITING_HUMAN_REVIEW, recovered.status)
        self.assertEqual(recovered, repeated)

    def test_setup_does_not_modify_existing_pgvector_collection(self):
        with self.app_pool.connection() as conn:
            table = conn.execute(
                "SELECT to_regclass('public.langchain_pg_embedding') AS name"
            ).fetchone()["name"]
            if table is None:
                self.skipTest("test database has no pgvector collection")
            before = conn.execute(
                "SELECT count(*) AS count FROM public.langchain_pg_embedding"
            ).fetchone()["count"]

        setup_postgres_persistence(TEST_DATABASE_URL)

        with self.app_pool.connection() as conn:
            after = conn.execute(
                "SELECT count(*) AS count FROM public.langchain_pg_embedding"
            ).fetchone()["count"]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
