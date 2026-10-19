"""Run-isolated application service for conversations and human review."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock, RLock
from typing import Any, Iterator, Literal, Protocol, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from regulatory_agent.multi_agent import (
    HumanReviewAction,
    MultiAgentResearchResult,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationServiceError(Exception):
    """Base class for explicit application lifecycle failures."""


class ConversationNotFoundError(ConversationServiceError):
    """Raised when a conversation ID is unknown."""


class RunNotFoundError(ConversationServiceError):
    """Raised when a run ID is unknown."""


class InvalidRunTransitionError(ConversationServiceError):
    """Raised when an operation is invalid for the current run status."""


class InvalidReviewActionError(ConversationServiceError):
    """Raised when a human review action or feedback is invalid."""


class StaleReviewRequestError(ConversationServiceError):
    """Raised when a browser action targets an obsolete review request."""


class IdempotencyConflictError(ConversationServiceError):
    """Raised when an idempotency key is reused for a different command."""


class RunExecutionError(ConversationServiceError):
    """Raised after a graph exception has been recorded as a failed run."""

    def __init__(self, run_id: UUID, message: str) -> None:
        super().__init__(message)
        self.run_id = run_id


class ReviewClaimConflictError(ConversationServiceError):
    """Raised when another worker owns or may have executed a review claim."""


class ReviewClaimLostError(ConversationServiceError):
    """Raised when a worker tries to complete a claim it no longer owns."""


class ReconciliationError(ConversationServiceError):
    """Raised when checkpoint and application state cannot be reconciled safely."""


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    COMPLETED = "completed"
    CLOSED_FOR_EDIT = "closed_for_edit"
    FAILED = "failed"


class ConversationRecord(BaseModel):
    """One user-visible conversation containing independent runs."""

    model_config = ConfigDict(frozen=True)

    conversation_id: UUID
    created_at: datetime
    updated_at: datetime


class PendingHumanReview(BaseModel):
    """Application identity for exactly one LangGraph interrupt."""

    model_config = ConfigDict(frozen=True)

    review_request_id: UUID
    reason: str
    allowed_actions: list[HumanReviewAction]
    evaluation_total_score: int | None = None
    checkpoint_id: str | None = None
    created_at: datetime


class ReviewExecutionClaim(BaseModel):
    """Short database lease guarding one graph resume operation."""

    model_config = ConfigDict(frozen=True)

    claim_id: UUID
    review_request_id: UUID
    owner_id: str
    action: HumanReviewAction
    feedback: str
    idempotency_key: str
    claimed_at: datetime
    expires_at: datetime


class HumanReviewActionRecord(BaseModel):
    """Compact audit record for one accepted human review command."""

    model_config = ConfigDict(frozen=True)

    review_action_id: UUID
    review_request_id: UUID
    action: HumanReviewAction
    feedback: str
    idempotency_key: str
    created_at: datetime


class RunRecord(BaseModel):
    """Application summary of one isolated regulatory research execution."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    conversation_id: UUID
    thread_id: str
    question: str
    status: RunStatus
    termination_reason: str | None = None
    latest_answer: str = ""
    citations: list[str] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)
    source_types: list[str] = Field(default_factory=list)
    evaluation_decision: str | None = None
    evaluation_total_score: int | None = None
    evaluation_error: str | None = None
    validation_is_valid: bool | None = None
    evaluation_retry_count: int = Field(default=0, ge=0)
    revision_count: int = Field(default=0, ge=0)
    research_retry_count: int = Field(default=0, ge=0)
    pending_review: PendingHumanReview | None = None
    review_claim: ReviewExecutionClaim | None = None
    review_history: list[HumanReviewActionRecord] = Field(default_factory=list)
    version: int = Field(default=0, ge=0)
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


MessageRole = Literal["user", "assistant", "system"]
MessageKind = Literal[
    "question",
    "answer",
    "human_review_request",
    "human_feedback",
    "run_status",
]


class ConversationMessage(BaseModel):
    """Display-oriented message independent from LangGraph checkpoint state."""

    model_config = ConfigDict(frozen=True)

    message_id: UUID
    conversation_id: UUID
    run_id: UUID
    sequence: int = Field(ge=1)
    role: MessageRole
    kind: MessageKind
    content: str
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ReviewSubmissionReceipt(BaseModel):
    """Stored response used to make review submission idempotent."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    idempotency_key: str
    review_request_id: UUID
    action: HumanReviewAction
    feedback: str
    result: RunRecord


class ReviewClaimOutcome(BaseModel):
    """Atomic claim result: either replay a receipt or execute the claimed run."""

    model_config = ConfigDict(frozen=True)

    run: RunRecord | None = None
    receipt: ReviewSubmissionReceipt | None = None


class RegulatoryAgentRunner(Protocol):
    """Existing graph API consumed by the application service."""

    def ask(
        self, question: str, *, thread_id: str | None = None
    ) -> MultiAgentResearchResult: ...

    def inspect_thread(self, *, thread_id: str) -> Any | None: ...

    def resume_human_review(
        self,
        *,
        thread_id: str,
        approved: bool | None = None,
        feedback: str = "",
        action: HumanReviewAction | None = None,
    ) -> MultiAgentResearchResult: ...


class ConversationRepository(Protocol):
    """Persistence boundary for a future PostgreSQL implementation."""

    def create_conversation(self, conversation: ConversationRecord) -> None: ...

    def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None: ...

    def save_conversation(self, conversation: ConversationRecord) -> None: ...

    def list_conversations(self) -> list[ConversationRecord]: ...

    def create_run(self, run: RunRecord) -> None: ...

    def get_run(self, run_id: UUID) -> RunRecord | None: ...

    def save_run(self, run: RunRecord) -> None: ...

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
    ) -> ConversationMessage: ...

    def list_messages(self, conversation_id: UUID) -> list[ConversationMessage]: ...

    def get_review_receipt(
        self, run_id: UUID, idempotency_key: str
    ) -> ReviewSubmissionReceipt | None: ...

    def save_review_receipt(self, receipt: ReviewSubmissionReceipt) -> None: ...

    def claim_review(
        self,
        *,
        run_id: UUID,
        expected_review_request_id: UUID,
        action_record: HumanReviewActionRecord,
        claim: ReviewExecutionClaim,
    ) -> ReviewClaimOutcome: ...

    def complete_review_claim(
        self,
        *,
        run: RunRecord,
        receipt: ReviewSubmissionReceipt,
        claim_id: UUID,
    ) -> None: ...

    def lock_run(self, run_id: UUID) -> AbstractContextManager[None]: ...


class InMemoryConversationRepository:
    """Thread-safe deterministic repository for Phase 4A and unit tests."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._conversations: dict[UUID, ConversationRecord] = {}
        self._runs: dict[UUID, RunRecord] = {}
        self._messages: dict[UUID, list[ConversationMessage]] = {}
        self._receipts: dict[tuple[UUID, str], ReviewSubmissionReceipt] = {}
        self._run_locks: dict[UUID, Lock] = {}

    @staticmethod
    def _copy(value: ModelT) -> ModelT:
        return value.model_copy(deep=True)

    def create_conversation(self, conversation: ConversationRecord) -> None:
        with self._lock:
            if conversation.conversation_id in self._conversations:
                raise ValueError("conversation ID already exists")
            self._conversations[conversation.conversation_id] = self._copy(conversation)
            self._messages[conversation.conversation_id] = []

    def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None:
        with self._lock:
            value = self._conversations.get(conversation_id)
            return self._copy(value) if value is not None else None

    def save_conversation(self, conversation: ConversationRecord) -> None:
        with self._lock:
            if conversation.conversation_id not in self._conversations:
                raise ConversationNotFoundError(str(conversation.conversation_id))
            self._conversations[conversation.conversation_id] = self._copy(conversation)

    def list_conversations(self) -> list[ConversationRecord]:
        with self._lock:
            values = sorted(
                self._conversations.values(), key=lambda item: item.created_at
            )
            return [self._copy(item) for item in values]

    def create_run(self, run: RunRecord) -> None:
        with self._lock:
            if run.run_id in self._runs:
                raise ValueError("run ID already exists")
            self._runs[run.run_id] = self._copy(run)
            self._run_locks[run.run_id] = Lock()

    def get_run(self, run_id: UUID) -> RunRecord | None:
        with self._lock:
            value = self._runs.get(run_id)
            return self._copy(value) if value is not None else None

    def save_run(self, run: RunRecord) -> None:
        with self._lock:
            if run.run_id not in self._runs:
                raise RunNotFoundError(str(run.run_id))
            self._runs[run.run_id] = self._copy(run)

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
        with self._lock:
            messages = self._messages.get(conversation_id)
            if messages is None:
                raise ConversationNotFoundError(str(conversation_id))
            message = ConversationMessage(
                message_id=uuid4(),
                conversation_id=conversation_id,
                run_id=run_id,
                sequence=len(messages) + 1,
                role=role,
                kind=kind,
                content=content,
                citations=list(citations or []),
                metadata=dict(metadata or {}),
                created_at=_now(),
            )
            messages.append(message)
            return self._copy(message)

    def list_messages(self, conversation_id: UUID) -> list[ConversationMessage]:
        with self._lock:
            if conversation_id not in self._conversations:
                raise ConversationNotFoundError(str(conversation_id))
            return [self._copy(item) for item in self._messages[conversation_id]]

    def get_review_receipt(
        self, run_id: UUID, idempotency_key: str
    ) -> ReviewSubmissionReceipt | None:
        with self._lock:
            value = self._receipts.get((run_id, idempotency_key))
            return self._copy(value) if value is not None else None

    def save_review_receipt(self, receipt: ReviewSubmissionReceipt) -> None:
        with self._lock:
            self._receipts[(receipt.run_id, receipt.idempotency_key)] = self._copy(
                receipt
            )

    def claim_review(
        self,
        *,
        run_id: UUID,
        expected_review_request_id: UUID,
        action_record: HumanReviewActionRecord,
        claim: ReviewExecutionClaim,
    ) -> ReviewClaimOutcome:
        with self.lock_run(run_id):
            receipt = self.get_review_receipt(
                run_id, action_record.idempotency_key
            )
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

            run = self.get_run(run_id)
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
            self.save_run(claimed)
            return ReviewClaimOutcome(run=claimed)

    def complete_review_claim(
        self,
        *,
        run: RunRecord,
        receipt: ReviewSubmissionReceipt,
        claim_id: UUID,
    ) -> None:
        with self.lock_run(run.run_id):
            current = self.get_run(run.run_id)
            if (
                current.review_claim is None
                or current.review_claim.claim_id != claim_id
            ):
                raise ReviewClaimLostError("review execution claim is no longer owned")
            self.save_run(run)
            self.save_review_receipt(receipt)

    @contextmanager
    def lock_run(self, run_id: UUID) -> Iterator[None]:
        with self._lock:
            run_lock = self._run_locks.get(run_id)
        if run_lock is None:
            raise RunNotFoundError(str(run_id))
        with run_lock:
            yield


_SUCCESSFUL_TERMINATIONS = {"answered", "human_approved"}
_REVIEW_ACTIONS: set[str] = {
    "approve",
    "revise",
    "research_again",
    "edit_question",
}


class ConversationService:
    """Own run IDs and enforce lifecycle safety around one compiled graph."""

    _ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
        RunStatus.CREATED: frozenset({RunStatus.RUNNING}),
        RunStatus.RUNNING: frozenset(
            {
                RunStatus.AWAITING_HUMAN_REVIEW,
                RunStatus.COMPLETED,
                RunStatus.CLOSED_FOR_EDIT,
                RunStatus.FAILED,
            }
        ),
        RunStatus.AWAITING_HUMAN_REVIEW: frozenset({RunStatus.RUNNING}),
        RunStatus.COMPLETED: frozenset(),
        RunStatus.CLOSED_FOR_EDIT: frozenset(),
        RunStatus.FAILED: frozenset(),
    }

    def __init__(
        self,
        *,
        agent: RegulatoryAgentRunner,
        repository: ConversationRepository | None = None,
        worker_id: str | None = None,
        review_claim_ttl_seconds: int = 300,
    ) -> None:
        if review_claim_ttl_seconds <= 0:
            raise ValueError("review_claim_ttl_seconds must be positive")
        self.agent = agent
        self.repository = repository or InMemoryConversationRepository()
        self.worker_id = worker_id or str(uuid4())
        self.review_claim_ttl_seconds = review_claim_ttl_seconds

    def create_conversation(self) -> ConversationRecord:
        now = _now()
        conversation = ConversationRecord(
            conversation_id=uuid4(),
            created_at=now,
            updated_at=now,
        )
        self.repository.create_conversation(conversation)
        return conversation

    def start_run(self, conversation_id: UUID, question: str) -> RunRecord:
        if self.repository.get_conversation(conversation_id) is None:
            raise ConversationNotFoundError(str(conversation_id))
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        run_id = uuid4()
        now = _now()
        run = RunRecord(
            run_id=run_id,
            conversation_id=conversation_id,
            thread_id=str(run_id),
            question=normalized_question,
            status=RunStatus.CREATED,
            created_at=now,
            updated_at=now,
        )
        self.repository.create_run(run)
        self.repository.append_message(
            conversation_id=conversation_id,
            run_id=run_id,
            role="user",
            kind="question",
            content=normalized_question,
        )
        running = self._transition(run, RunStatus.RUNNING)
        self.repository.save_run(running)
        try:
            result = self.agent.ask(normalized_question, thread_id=str(run_id))
        except Exception as error:
            failed = self._transition(
                running,
                RunStatus.FAILED,
                error=str(error),
                completed_at=_now(),
            )
            self.repository.save_run(failed)
            self._append_status(failed)
            raise RunExecutionError(run_id, str(error)) from error
        return self._apply_result(running, result)

    def get_run(self, run_id: UUID) -> RunRecord:
        run = self.repository.get_run(run_id)
        if run is None:
            raise RunNotFoundError(str(run_id))
        return run

    def list_conversations(self) -> list[ConversationRecord]:
        return self.repository.list_conversations()

    def list_conversation_messages(
        self, conversation_id: UUID
    ) -> list[ConversationMessage]:
        return self.repository.list_messages(conversation_id)

    def reconcile_run(self, run_id: UUID) -> RunRecord:
        """Conservatively reconcile application metadata with one checkpoint."""

        run = self.get_run(run_id)
        inspect = getattr(self.agent, "inspect_thread", None)
        if inspect is None:
            raise ReconciliationError("the configured agent cannot inspect checkpoints")
        try:
            snapshot = inspect(thread_id=run.thread_id)
        except Exception as error:
            return self._mark_reconciliation_failure(
                run, f"reconciliation_checkpoint_read_failed: {error}"
            )
        if snapshot is None:
            if run.status in {
                RunStatus.COMPLETED,
                RunStatus.CLOSED_FOR_EDIT,
            }:
                return run
            return self._mark_reconciliation_failure(
                run, "reconciliation_no_checkpoint: graph state is unavailable"
            )

        claim = run.review_claim
        if claim is not None and claim.expires_at > _now():
            raise ReviewClaimConflictError(
                f"review claim {claim.claim_id} is active until "
                f"{claim.expires_at.isoformat()}"
            )

        checkpoint_id = snapshot.checkpoint_id
        result = snapshot.result
        if (
            claim is not None
            and result.termination_reason == "awaiting_human_review"
            and run.pending_review is not None
            and run.pending_review.checkpoint_id is not None
            and run.pending_review.checkpoint_id == checkpoint_id
        ):
            restored = run.model_copy(
                update={
                    "status": RunStatus.AWAITING_HUMAN_REVIEW,
                    "review_claim": None,
                    "error": (
                        "expired review claim released; checkpoint did not advance"
                    ),
                    "version": run.version + 1,
                    "updated_at": _now(),
                    "completed_at": None,
                }
            )
            self.repository.save_run(restored)
            self._ensure_human_feedback_message(run, claim)
            self._record_result_messages(
                restored, restored, result, restored.pending_review
            )
            return restored

        if claim is None and run.status in {
            RunStatus.AWAITING_HUMAN_REVIEW,
            RunStatus.COMPLETED,
            RunStatus.CLOSED_FOR_EDIT,
        }:
            self._record_result_messages(
                run, run, result, run.pending_review
            )
            return run

        working = run
        if run.status == RunStatus.FAILED:
            working = run.model_copy(update={"status": RunStatus.RUNNING})
        try:
            updated, pending = self._updated_from_result(working, result)
        except Exception as error:
            return self._mark_reconciliation_failure(
                run, f"reconciliation_inconsistent_checkpoint: {error}"
            )
        updated = updated.model_copy(update={"review_claim": None})
        if claim is not None:
            self._ensure_human_feedback_message(run, claim)
            receipt = ReviewSubmissionReceipt(
                run_id=run.run_id,
                idempotency_key=claim.idempotency_key,
                review_request_id=claim.review_request_id,
                action=claim.action,
                feedback=claim.feedback,
                result=updated,
            )
            self.repository.complete_review_claim(
                run=updated,
                receipt=receipt,
                claim_id=claim.claim_id,
            )
        else:
            self.repository.save_run(updated)
        self._record_result_messages(run, updated, result, pending)
        return updated

    def submit_human_review(
        self,
        run_id: UUID,
        action: HumanReviewAction,
        feedback: str,
        idempotency_key: str,
        expected_review_request_id: UUID,
    ) -> RunRecord:
        normalized_key = idempotency_key.strip()
        if not normalized_key:
            raise ValueError("idempotency_key must not be empty")
        if action not in _REVIEW_ACTIONS:
            raise InvalidReviewActionError(f"unsupported review action: {action}")
        normalized_feedback = feedback.strip()
        if action != "approve" and not normalized_feedback:
            raise InvalidReviewActionError(
                "feedback is required for a non-approval action"
            )

        now = _now()
        action_record = HumanReviewActionRecord(
            review_action_id=uuid4(),
            review_request_id=expected_review_request_id,
            action=action,
            feedback=normalized_feedback,
            idempotency_key=normalized_key,
            created_at=now,
        )
        claim = ReviewExecutionClaim(
            claim_id=uuid4(),
            review_request_id=expected_review_request_id,
            owner_id=self.worker_id,
            action=action,
            feedback=normalized_feedback,
            idempotency_key=normalized_key,
            claimed_at=now,
            expires_at=now + timedelta(seconds=self.review_claim_ttl_seconds),
        )
        outcome = self.repository.claim_review(
            run_id=run_id,
            expected_review_request_id=expected_review_request_id,
            action_record=action_record,
            claim=claim,
        )
        if outcome.receipt is not None:
            return outcome.receipt.result.model_copy(deep=True)
        running = outcome.run
        if running is None:
            raise ReviewClaimLostError("repository returned an empty review claim")

        self._ensure_human_feedback_message(running, claim)
        try:
            result = self.agent.resume_human_review(
                thread_id=running.thread_id,
                action=action,
                feedback=normalized_feedback,
            )
        except Exception as error:
            failed = self._transition(
                running,
                RunStatus.FAILED,
                error=f"review_execution_ambiguous: {error}",
                completed_at=_now(),
            )
            self.repository.save_run(failed)
            self._append_status(failed)
            raise RunExecutionError(run_id, str(error)) from error

        updated, pending = self._updated_from_result(running, result)
        updated = updated.model_copy(update={"review_claim": None})
        receipt = ReviewSubmissionReceipt(
            run_id=run_id,
            idempotency_key=normalized_key,
            review_request_id=expected_review_request_id,
            action=action,
            feedback=normalized_feedback,
            result=updated,
        )
        self.repository.complete_review_claim(
            run=updated,
            receipt=receipt,
            claim_id=claim.claim_id,
        )
        self._record_result_messages(running, updated, result, pending)
        return updated

    def _apply_result(
        self, run: RunRecord, result: MultiAgentResearchResult
    ) -> RunRecord:
        updated, pending = self._updated_from_result(run, result)
        self.repository.save_run(updated)
        self._record_result_messages(run, updated, result, pending)
        return updated

    def _updated_from_result(
        self, run: RunRecord, result: MultiAgentResearchResult
    ) -> tuple[RunRecord, PendingHumanReview | None]:
        if result.thread_id != run.thread_id:
            raise InvalidRunTransitionError(
                "graph returned a thread ID different from the run ID"
            )
        now = _now()
        if result.termination_reason == "awaiting_human_review":
            request = result.human_review_request
            if request is None:
                raise InvalidRunTransitionError(
                    "awaiting_human_review requires a review request payload"
                )
            status = RunStatus.AWAITING_HUMAN_REVIEW
            pending = PendingHumanReview(
                review_request_id=uuid4(),
                reason=request.reason,
                allowed_actions=list(request.allowed_actions),
                evaluation_total_score=(
                    request.evaluation.total_score if request.evaluation else None
                ),
                checkpoint_id=self._checkpoint_id(run.thread_id),
                created_at=now,
            )
            completed_at = None
        elif result.termination_reason in _SUCCESSFUL_TERMINATIONS:
            status = RunStatus.COMPLETED
            pending = None
            completed_at = now
        elif (
            result.termination_reason == "human_rejected"
            and result.human_review is not None
            and result.human_review.action == "edit_question"
        ):
            status = RunStatus.CLOSED_FOR_EDIT
            pending = None
            completed_at = now
        else:
            status = RunStatus.FAILED
            pending = None
            completed_at = now

        updated = self._transition(
            run,
            status,
            termination_reason=result.termination_reason,
            latest_answer=result.final_answer,
            citations=list(result.citations),
            source_count=len(result.sources),
            source_types=list(
                dict.fromkeys(source.source_type for source in result.sources)
            ),
            evaluation_decision=(
                result.evaluation.decision if result.evaluation else None
            ),
            evaluation_total_score=(
                result.evaluation.total_score if result.evaluation else None
            ),
            evaluation_error=result.evaluation_error,
            validation_is_valid=(
                result.validation.valid if result.validation else None
            ),
            evaluation_retry_count=result.evaluation_retry_count,
            revision_count=result.revision_count,
            research_retry_count=result.research_retry_count,
            pending_review=pending,
            error=self._result_error(result),
            completed_at=completed_at,
        )
        return updated, pending

    def _record_result_messages(
        self,
        previous: RunRecord,
        updated: RunRecord,
        result: MultiAgentResearchResult,
        pending: PendingHumanReview | None,
    ) -> None:
        conversation = self.repository.get_conversation(updated.conversation_id)
        if conversation is not None:
            self.repository.save_conversation(
                conversation.model_copy(update={"updated_at": updated.updated_at})
            )

        messages = self.repository.list_messages(updated.conversation_id)
        if (
            result.final_answer
            and not any(
                message.run_id == updated.run_id
                and message.kind == "answer"
                and message.content == result.final_answer
                and message.citations == list(result.citations)
                for message in messages
            )
        ):
            self.repository.append_message(
                conversation_id=updated.conversation_id,
                run_id=updated.run_id,
                role="assistant",
                kind="answer",
                content=result.final_answer,
                citations=list(result.citations),
                metadata={"termination_reason": result.termination_reason},
            )
        if pending is not None and not any(
            message.run_id == updated.run_id
            and message.kind == "human_review_request"
            and message.metadata.get("review_request_id")
            == str(pending.review_request_id)
            for message in messages
        ):
            self.repository.append_message(
                conversation_id=updated.conversation_id,
                run_id=updated.run_id,
                role="system",
                kind="human_review_request",
                content=pending.reason,
                metadata={
                    "review_request_id": str(pending.review_request_id),
                    "allowed_actions": list(pending.allowed_actions),
                    "evaluation_total_score": pending.evaluation_total_score,
                },
            )
        if not any(
            message.run_id == updated.run_id
            and message.kind == "run_status"
            and message.content == updated.status.value
            and message.metadata.get("termination_reason")
            == updated.termination_reason
            for message in messages
        ):
            self._append_status(updated)

    def _ensure_human_feedback_message(
        self, run: RunRecord, claim: ReviewExecutionClaim
    ) -> None:
        messages = self.repository.list_messages(run.conversation_id)
        if any(
            message.run_id == run.run_id
            and message.kind == "human_feedback"
            and message.metadata.get("claim_id") == str(claim.claim_id)
            for message in messages
        ):
            return
        self.repository.append_message(
            conversation_id=run.conversation_id,
            run_id=run.run_id,
            role="user",
            kind="human_feedback",
            content=claim.feedback or "Approved.",
            metadata={
                "review_request_id": str(claim.review_request_id),
                "action": claim.action,
                "claim_id": str(claim.claim_id),
            },
        )

    def _checkpoint_id(self, thread_id: str) -> str | None:
        inspect = getattr(self.agent, "inspect_thread", None)
        if inspect is None:
            return None
        snapshot = inspect(thread_id=thread_id)
        return snapshot.checkpoint_id if snapshot is not None else None

    def _mark_reconciliation_failure(
        self, run: RunRecord, error: str
    ) -> RunRecord:
        if run.status == RunStatus.FAILED and run.error == error:
            return run
        failed = run.model_copy(
            update={
                "status": RunStatus.FAILED,
                "error": error,
                "version": run.version + 1,
                "updated_at": _now(),
                "completed_at": _now(),
            }
        )
        self.repository.save_run(failed)
        self._append_status(failed)
        return failed

    def _append_status(self, run: RunRecord) -> None:
        self.repository.append_message(
            conversation_id=run.conversation_id,
            run_id=run.run_id,
            role="system",
            kind="run_status",
            content=run.status.value,
            metadata={"termination_reason": run.termination_reason},
        )

    @classmethod
    def _transition(
        cls,
        run: RunRecord,
        status: RunStatus,
        **updates: Any,
    ) -> RunRecord:
        if status != run.status and status not in cls._ALLOWED_TRANSITIONS[run.status]:
            raise InvalidRunTransitionError(
                f"invalid run transition: {run.status.value} -> {status.value}"
            )
        return run.model_copy(
            update={
                "status": status,
                "version": run.version + 1,
                "updated_at": _now(),
                **updates,
            }
        )

    @staticmethod
    def _result_error(result: MultiAgentResearchResult) -> str | None:
        return (
            result.evaluation_error
            or result.revision_error
            or result.research_retry_error
        )
