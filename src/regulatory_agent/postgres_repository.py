"""PostgreSQL application metadata repository for Phase 4B."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import files
from typing import Any, Iterator
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from regulatory_agent.conversation_service import (
    ConversationMessage,
    ConversationNotFoundError,
    ConversationRecord,
    HumanReviewActionRecord,
    IdempotencyConflictError,
    InvalidReviewActionError,
    InvalidRunTransitionError,
    MessageKind,
    MessageRole,
    PendingHumanReview,
    ReviewClaimConflictError,
    ReviewClaimLostError,
    ReviewClaimOutcome,
    ReviewExecutionClaim,
    ReviewSubmissionReceipt,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    StaleReviewRequestError,
    _now,
)


APPLICATION_SCHEMA = "regulatory_app"
CHECKPOINT_SCHEMA = "regulatory_checkpoint"


class PersistenceInitializationError(RuntimeError):
    """Raised when PostgreSQL persistence cannot be initialized."""


def _json(value: Any) -> Jsonb:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return Jsonb(value)


class PostgresConversationRepository:
    """Transactional PostgreSQL implementation of ``ConversationRepository``."""

    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    def setup(self) -> None:
        """Apply packaged, monotonic application-schema migrations."""

        migration_root = files("regulatory_agent").joinpath("migrations")
        migrations = sorted(
            item for item in migration_root.iterdir() if item.name.endswith(".sql")
        )
        try:
            with self.pool.connection() as conn, conn.transaction():
                conn.execute(f"CREATE SCHEMA IF NOT EXISTS {APPLICATION_SCHEMA}")
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {APPLICATION_SCHEMA}.schema_migrations (
                        version text PRIMARY KEY,
                        applied_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                applied = {
                    row["version"]
                    for row in conn.execute(
                        f"SELECT version FROM {APPLICATION_SCHEMA}.schema_migrations"
                    ).fetchall()
                }
                for migration in migrations:
                    version = migration.name.split("_", 1)[0]
                    if version in applied:
                        continue
                    statements = migration.read_text(encoding="utf-8").split(";")
                    for statement in statements:
                        if statement.strip():
                            conn.execute(statement)
                    conn.execute(
                        f"""
                        INSERT INTO {APPLICATION_SCHEMA}.schema_migrations (version)
                        VALUES (%s)
                        """,
                        (version,),
                    )
        except Exception as error:
            raise PersistenceInitializationError(
                f"application schema setup failed: {error}"
            ) from error

    def create_conversation(self, conversation: ConversationRecord) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {APPLICATION_SCHEMA}.conversations
                    (conversation_id, created_at, updated_at)
                VALUES (%s, %s, %s)
                """,
                (
                    conversation.conversation_id,
                    conversation.created_at,
                    conversation.updated_at,
                ),
            )

    def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                f"""
                SELECT conversation_id, created_at, updated_at
                FROM {APPLICATION_SCHEMA}.conversations
                WHERE conversation_id = %s
                """,
                (conversation_id,),
            ).fetchone()
        return ConversationRecord.model_validate(row) if row else None

    def save_conversation(self, conversation: ConversationRecord) -> None:
        with self.pool.connection() as conn:
            result = conn.execute(
                f"""
                UPDATE {APPLICATION_SCHEMA}.conversations
                SET updated_at = %s
                WHERE conversation_id = %s
                """,
                (conversation.updated_at, conversation.conversation_id),
            )
            if result.rowcount != 1:
                raise ConversationNotFoundError(str(conversation.conversation_id))

    def list_conversations(self) -> list[ConversationRecord]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT conversation_id, created_at, updated_at
                FROM {APPLICATION_SCHEMA}.conversations
                ORDER BY created_at, conversation_id
                """
            ).fetchall()
        return [ConversationRecord.model_validate(row) for row in rows]

    def create_run(self, run: RunRecord) -> None:
        with self.pool.connection() as conn:
            self._insert_run(conn, run)

    def get_run(self, run_id: UUID) -> RunRecord | None:
        with self.pool.connection() as conn:
            return self._get_run(conn, run_id)

    def save_run(self, run: RunRecord) -> None:
        with self.pool.connection() as conn:
            self._update_run(conn, run, expected_version=run.version - 1)

    def append_message(
        self,
        *,
        conversation_id: UUID,
        run_id: UUID,
        role: MessageRole,
        kind: MessageKind,
        content: str,
        citations: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        from uuid import uuid4

        with self.pool.connection() as conn, conn.transaction():
            locked = conn.execute(
                f"""
                SELECT conversation_id
                FROM {APPLICATION_SCHEMA}.conversations
                WHERE conversation_id = %s
                FOR UPDATE
                """,
                (conversation_id,),
            ).fetchone()
            if locked is None:
                raise ConversationNotFoundError(str(conversation_id))
            sequence = conn.execute(
                f"""
                SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence
                FROM {APPLICATION_SCHEMA}.conversation_messages
                WHERE conversation_id = %s
                """,
                (conversation_id,),
            ).fetchone()["sequence"]
            message = ConversationMessage(
                message_id=uuid4(),
                conversation_id=conversation_id,
                run_id=run_id,
                sequence=sequence,
                role=role,
                kind=kind,
                content=content,
                citations=list(citations or []),
                metadata=dict(metadata or {}),
                created_at=_now(),
            )
            conn.execute(
                f"""
                INSERT INTO {APPLICATION_SCHEMA}.conversation_messages (
                    message_id, conversation_id, run_id, sequence, role, kind,
                    content, citations, metadata, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    message.message_id,
                    conversation_id,
                    run_id,
                    sequence,
                    role,
                    kind,
                    content,
                    _json(message.citations),
                    _json(message.metadata),
                    message.created_at,
                ),
            )
        return message

    def list_messages(self, conversation_id: UUID) -> list[ConversationMessage]:
        if self.get_conversation(conversation_id) is None:
            raise ConversationNotFoundError(str(conversation_id))
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT message_id, conversation_id, run_id, sequence, role, kind,
                       content, citations, metadata, created_at
                FROM {APPLICATION_SCHEMA}.conversation_messages
                WHERE conversation_id = %s
                ORDER BY sequence
                """,
                (conversation_id,),
            ).fetchall()
        return [ConversationMessage.model_validate(row) for row in rows]

    def get_review_receipt(
        self, run_id: UUID, idempotency_key: str
    ) -> ReviewSubmissionReceipt | None:
        with self.pool.connection() as conn:
            return self._get_receipt(conn, run_id, idempotency_key)

    def save_review_receipt(self, receipt: ReviewSubmissionReceipt) -> None:
        with self.pool.connection() as conn:
            self._insert_receipt(conn, receipt)

    def claim_review(
        self,
        *,
        run_id: UUID,
        expected_review_request_id: UUID,
        action_record: HumanReviewActionRecord,
        claim: ReviewExecutionClaim,
    ) -> ReviewClaimOutcome:
        """Claim one pending review without holding the transaction during resume."""

        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                f"""
                SELECT * FROM {APPLICATION_SCHEMA}.runs
                WHERE run_id = %s
                FOR UPDATE
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise RunNotFoundError(str(run_id))
            run = self._run_from_row(conn, row)
            receipt = self._get_receipt(conn, run_id, action_record.idempotency_key)
            if receipt is not None:
                if (
                    receipt.review_request_id != expected_review_request_id
                    or receipt.action != action_record.action
                    or receipt.feedback != action_record.feedback
                ):
                    raise IdempotencyConflictError(
                        "idempotency key was already used for a different review"
                    )
                return ReviewClaimOutcome(receipt=receipt)

            existing_action = next(
                (
                    item
                    for item in run.review_history
                    if item.idempotency_key == action_record.idempotency_key
                ),
                None,
            )
            if existing_action is not None:
                if (
                    existing_action.review_request_id != expected_review_request_id
                    or existing_action.action != action_record.action
                    or existing_action.feedback != action_record.feedback
                ):
                    raise IdempotencyConflictError(
                        "idempotency key was already used for a different review"
                    )
                raise ReviewClaimConflictError(
                    "the prior review submission has no completed receipt"
                )
            if run.review_claim is not None:
                raise ReviewClaimConflictError(
                    f"review request is claimed by {run.review_claim.owner_id}"
                )
            if run.status != RunStatus.AWAITING_HUMAN_REVIEW:
                if any(
                    item.review_request_id == expected_review_request_id
                    for item in run.review_history
                ):
                    raise StaleReviewRequestError(
                        "review request is no longer the active interrupt"
                    )
                raise InvalidRunTransitionError(
                    f"run status {run.status.value} cannot accept human review"
                )
            pending = run.pending_review
            if (
                pending is None
                or pending.review_request_id != expected_review_request_id
            ):
                raise StaleReviewRequestError(
                    "review request is no longer the active interrupt"
                )
            if action_record.action not in pending.allowed_actions:
                raise InvalidReviewActionError(
                    f"review action {action_record.action} is not allowed "
                    "for this request"
                )
            claimed = run.model_copy(
                update={
                    "status": RunStatus.RUNNING,
                    "review_claim": claim,
                    "review_history": [*run.review_history, action_record],
                    "version": run.version + 1,
                    "updated_at": _now(),
                }
            )
            conn.execute(
                f"""
                INSERT INTO {APPLICATION_SCHEMA}.human_review_actions (
                    review_action_id, run_id, review_request_id, action,
                    feedback, idempotency_key, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    action_record.review_action_id,
                    run_id,
                    action_record.review_request_id,
                    action_record.action,
                    action_record.feedback,
                    action_record.idempotency_key,
                    action_record.created_at,
                ),
            )
            self._update_run(conn, claimed, expected_version=run.version)
            return ReviewClaimOutcome(run=claimed)

    def complete_review_claim(
        self,
        *,
        run: RunRecord,
        receipt: ReviewSubmissionReceipt,
        claim_id: UUID,
    ) -> None:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                f"""
                SELECT * FROM {APPLICATION_SCHEMA}.runs
                WHERE run_id = %s
                FOR UPDATE
                """,
                (run.run_id,),
            ).fetchone()
            if row is None:
                raise RunNotFoundError(str(run.run_id))
            current = self._run_from_row(conn, row)
            if (
                current.review_claim is None
                or current.review_claim.claim_id != claim_id
            ):
                raise ReviewClaimLostError("review execution claim is no longer owned")
            self._update_run(conn, run, expected_version=current.version)
            self._insert_receipt(conn, receipt)

    @contextmanager
    def lock_run(self, run_id: UUID) -> Iterator[None]:
        """Compatibility shim; production service uses transactional claims."""

        if self.get_run(run_id) is None:
            raise RunNotFoundError(str(run_id))
        yield

    def _insert_run(self, conn: Connection, run: RunRecord) -> None:
        conn.execute(
            f"""
            INSERT INTO {APPLICATION_SCHEMA}.runs (
                run_id, conversation_id, thread_id, question, status,
                termination_reason, latest_answer, citations, source_count,
                source_types, evaluation_decision, evaluation_total_score,
                evaluation_error, validation_is_valid, evaluation_retry_count,
                revision_count, research_retry_count, pending_review,
                review_claim, error, version, created_at, updated_at, completed_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            self._run_values(run),
        )

    def _update_run(
        self, conn: Connection, run: RunRecord, *, expected_version: int
    ) -> None:
        result = conn.execute(
            f"""
            UPDATE {APPLICATION_SCHEMA}.runs SET
                conversation_id=%s, thread_id=%s, question=%s, status=%s,
                termination_reason=%s, latest_answer=%s, citations=%s,
                source_count=%s, source_types=%s, evaluation_decision=%s,
                evaluation_total_score=%s, evaluation_error=%s,
                validation_is_valid=%s, evaluation_retry_count=%s,
                revision_count=%s, research_retry_count=%s, pending_review=%s,
                review_claim=%s, error=%s, version=%s, created_at=%s,
                updated_at=%s, completed_at=%s
            WHERE run_id=%s AND version=%s
            """,
            (*self._run_values(run)[1:], run.run_id, expected_version),
        )
        if result.rowcount != 1:
            exists = conn.execute(
                f"SELECT 1 FROM {APPLICATION_SCHEMA}.runs WHERE run_id=%s",
                (run.run_id,),
            ).fetchone()
            if exists is None:
                raise RunNotFoundError(str(run.run_id))
            raise InvalidRunTransitionError(
                f"run {run.run_id} changed concurrently (expected version "
                f"{expected_version})"
            )

    @staticmethod
    def _run_values(run: RunRecord) -> tuple[Any, ...]:
        return (
            run.run_id,
            run.conversation_id,
            run.thread_id,
            run.question,
            run.status.value,
            run.termination_reason,
            run.latest_answer,
            _json(run.citations),
            run.source_count,
            _json(run.source_types),
            run.evaluation_decision,
            run.evaluation_total_score,
            run.evaluation_error,
            run.validation_is_valid,
            run.evaluation_retry_count,
            run.revision_count,
            run.research_retry_count,
            _json(run.pending_review) if run.pending_review else None,
            _json(run.review_claim) if run.review_claim else None,
            run.error,
            run.version,
            run.created_at,
            run.updated_at,
            run.completed_at,
        )

    def _get_run(self, conn: Connection, run_id: UUID) -> RunRecord | None:
        row = conn.execute(
            f"SELECT * FROM {APPLICATION_SCHEMA}.runs WHERE run_id=%s",
            (run_id,),
        ).fetchone()
        return self._run_from_row(conn, row) if row else None

    def _run_from_row(self, conn: Connection, row: dict[str, Any]) -> RunRecord:
        actions = conn.execute(
            f"""
            SELECT review_action_id, review_request_id, action, feedback,
                   idempotency_key, created_at
            FROM {APPLICATION_SCHEMA}.human_review_actions
            WHERE run_id=%s
            ORDER BY created_at, review_action_id
            """,
            (row["run_id"],),
        ).fetchall()
        payload = dict(row)
        payload["pending_review"] = (
            PendingHumanReview.model_validate(payload["pending_review"])
            if payload["pending_review"]
            else None
        )
        payload["review_claim"] = (
            ReviewExecutionClaim.model_validate(payload["review_claim"])
            if payload["review_claim"]
            else None
        )
        payload["review_history"] = [
            HumanReviewActionRecord.model_validate(item) for item in actions
        ]
        return RunRecord.model_validate(payload)

    def _get_receipt(
        self, conn: Connection, run_id: UUID, idempotency_key: str
    ) -> ReviewSubmissionReceipt | None:
        row = conn.execute(
            f"""
            SELECT run_id, idempotency_key, review_request_id, action,
                   feedback, result
            FROM {APPLICATION_SCHEMA}.review_submission_receipts
            WHERE run_id=%s AND idempotency_key=%s
            """,
            (run_id, idempotency_key),
        ).fetchone()
        return ReviewSubmissionReceipt.model_validate(row) if row else None

    def _insert_receipt(
        self, conn: Connection, receipt: ReviewSubmissionReceipt
    ) -> None:
        conn.execute(
            f"""
            INSERT INTO {APPLICATION_SCHEMA}.review_submission_receipts (
                run_id, idempotency_key, review_request_id, action,
                feedback, result
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                receipt.run_id,
                receipt.idempotency_key,
                receipt.review_request_id,
                receipt.action,
                receipt.feedback,
                _json(receipt.result),
            ),
        )


def create_connection_pool(
    conninfo: str,
    *,
    min_size: int = 1,
    max_size: int = 5,
    configure: Any | None = None,
) -> ConnectionPool:
    """Open and verify a psycopg pool with consistent row/transaction settings."""

    pool = ConnectionPool(
        conninfo,
        min_size=min_size,
        max_size=max_size,
        open=False,
        configure=configure,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    try:
        pool.open(wait=True)
        pool.check()
    except Exception as error:
        pool.close()
        raise PersistenceInitializationError(
            f"PostgreSQL connection failed: {error}"
        ) from error
    return pool
