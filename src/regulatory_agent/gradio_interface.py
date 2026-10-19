"""Gradio MVP backed exclusively by the conversation application service."""

from __future__ import annotations

import argparse
import atexit
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import logging
from threading import Lock
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

import gradio as gr
from pydantic import BaseModel, Field

from regulatory_agent.conversation_service import (
    ConversationMessage,
    ConversationRecord,
    ConversationService,
    ConversationServiceError,
    RunRecord,
    RunStatus,
    StaleReviewRequestError,
)
from regulatory_agent.multi_agent import HumanReviewAction, trajectory_logger
from regulatory_agent.persistence import StorageMode, open_conversation_service


logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = {
    RunStatus.CREATED,
    RunStatus.RUNNING,
    RunStatus.AWAITING_HUMAN_REVIEW,
}
_ACTION_LABELS: dict[HumanReviewAction, str] = {
    "approve": "Approve",
    "revise": "Revise Answer",
    "research_again": "Research Again",
    "edit_question": "Edit Question",
}


class ConversationServiceApi(Protocol):
    """The application-service surface used by the web layer."""

    def create_conversation(self) -> ConversationRecord: ...

    def start_run(self, conversation_id: UUID, question: str) -> RunRecord: ...

    def get_run(self, run_id: UUID) -> RunRecord: ...

    def list_conversations(self) -> list[ConversationRecord]: ...

    def list_conversation_messages(
        self, conversation_id: UUID
    ) -> list[ConversationMessage]: ...

    def submit_human_review(
        self,
        run_id: UUID,
        action: HumanReviewAction,
        feedback: str,
        idempotency_key: str,
        expected_review_request_id: UUID,
    ) -> RunRecord: ...

    def reconcile_run(self, run_id: UUID) -> RunRecord: ...


class WebSessionState(BaseModel):
    """Browser-session selection and pending-command state only."""

    conversation_id: UUID | None = None
    active_run_id: UUID | None = None
    pending_review_request_id: UUID | None = None
    review_submission_signature: str | None = None
    review_idempotency_key: str | None = None
    edit_question_draft: str = ""


class UiSnapshot(BaseModel):
    """Framework-neutral rendering result used by handlers and tests."""

    session: WebSessionState
    conversation_choices: list[tuple[str, str]] = Field(default_factory=list)
    chat_messages: list[dict[str, str]] = Field(default_factory=list)
    status_markdown: str = "Select a conversation or create a new one."
    error_markdown: str = ""
    review_markdown: str = ""
    review_visible: bool = False
    allowed_actions: list[HumanReviewAction] = Field(default_factory=list)
    question_value: str = ""
    question_interactive: bool = True


class GradioController:
    """Translate UI events into safe ``ConversationService`` operations."""

    def __init__(self, service: ConversationServiceApi) -> None:
        self.service = service
        self._execution_lock = Lock()
        self._executing_conversations: set[UUID] = set()

    def initialize(
        self, state: WebSessionState | dict[str, Any] | None = None
    ) -> UiSnapshot:
        session = self._state(state)
        try:
            conversations = self.service.list_conversations()
            known = {item.conversation_id for item in conversations}
            if session.conversation_id not in known:
                session.conversation_id = (
                    max(conversations, key=lambda item: item.updated_at).conversation_id
                    if conversations
                    else None
                )
            return self._snapshot(session, conversations=conversations)
        except Exception as error:
            return self._failure_snapshot(session, error)

    def create_conversation(
        self, state: WebSessionState | dict[str, Any] | None = None
    ) -> UiSnapshot:
        session = self._state(state)
        try:
            conversation = self.service.create_conversation()
            session = WebSessionState(conversation_id=conversation.conversation_id)
            return self._snapshot(session)
        except Exception as error:
            return self._failure_snapshot(session, error)

    def select_conversation(
        self,
        conversation_id: str | UUID | None,
        state: WebSessionState | dict[str, Any] | None = None,
    ) -> UiSnapshot:
        session = self._state(state)
        if not conversation_id:
            return self._snapshot(WebSessionState())
        try:
            session = WebSessionState(conversation_id=UUID(str(conversation_id)))
            return self._snapshot(session)
        except Exception as error:
            return self._failure_snapshot(session, error)

    def submit_question(
        self,
        question: str,
        state: WebSessionState | dict[str, Any] | None = None,
    ) -> UiSnapshot:
        session = self._state(state)
        normalized = question.strip()
        if not normalized:
            return self._snapshot(session, error="Enter a question.")
        if session.conversation_id is None:
            return self._snapshot(session, error="Create a conversation first.")

        conversation_id = session.conversation_id
        if not self._claim_conversation(conversation_id):
            return self._snapshot(
                session, error="Another request is already running in this conversation."
            )
        try:
            active = self._latest_active_run(conversation_id)
            if active is not None:
                session.active_run_id = active.run_id
                session.pending_review_request_id = (
                    active.pending_review.review_request_id
                    if active.pending_review is not None
                    else None
                )
                return self._snapshot(
                    session,
                    error="Complete the active run or human review first.",
                )
            run = self.service.start_run(conversation_id, normalized)
            session.active_run_id = (
                run.run_id if run.status in _ACTIVE_STATUSES else None
            )
            session.pending_review_request_id = (
                run.pending_review.review_request_id
                if run.pending_review is not None
                else None
            )
            session.edit_question_draft = ""
            return self._snapshot(session)
        except Exception as error:
            return self._failure_snapshot(session, error)
        finally:
            self._release_conversation(conversation_id)

    def submit_review(
        self,
        action: HumanReviewAction,
        feedback: str,
        state: WebSessionState | dict[str, Any] | None = None,
    ) -> UiSnapshot:
        session = self._state(state)
        if action not in _ACTION_LABELS:
            return self._snapshot(session, error="This review action is not supported.")
        if session.active_run_id is None or session.pending_review_request_id is None:
            return self._snapshot(session, error="There is no pending review request.")
        normalized_feedback = feedback.strip()
        if action != "approve" and not normalized_feedback:
            return self._snapshot(
                session, error="Feedback is required for non-approval actions."
            )

        run_id = session.active_run_id
        review_request_id = session.pending_review_request_id
        signature = "\x1f".join(
            (str(run_id), str(review_request_id), action, normalized_feedback)
        )
        if session.review_submission_signature != signature:
            session.review_submission_signature = signature
            session.review_idempotency_key = str(uuid4())
        idempotency_key = session.review_idempotency_key or str(uuid4())
        session.review_idempotency_key = idempotency_key

        try:
            run = self.service.submit_human_review(
                run_id=run_id,
                action=action,
                feedback=normalized_feedback,
                idempotency_key=idempotency_key,
                expected_review_request_id=review_request_id,
            )
            session.review_submission_signature = None
            session.review_idempotency_key = None
            session.active_run_id = (
                run.run_id if run.status in _ACTIVE_STATUSES else None
            )
            session.pending_review_request_id = (
                run.pending_review.review_request_id
                if run.pending_review is not None
                else None
            )
            if action == "edit_question" and run.status == RunStatus.CLOSED_FOR_EDIT:
                session.edit_question_draft = run.question
            return self._snapshot(session)
        except StaleReviewRequestError:
            refreshed = self._snapshot(
                session,
                error="The review request changed. Review the latest request.",
            )
            refreshed.session.review_submission_signature = None
            refreshed.session.review_idempotency_key = None
            return refreshed
        except Exception as error:
            # Preserve the key so retrying this exact command remains idempotent.
            return self._failure_snapshot(session, error)

    def _snapshot(
        self,
        session: WebSessionState,
        *,
        conversations: list[ConversationRecord] | None = None,
        error: str = "",
    ) -> UiSnapshot:
        conversations = conversations or self.service.list_conversations()
        choices = self._conversation_choices(conversations)
        if session.conversation_id is None:
            return UiSnapshot(
                session=session,
                conversation_choices=choices,
                error_markdown=self._error_markdown(error),
            )

        messages = self.service.list_conversation_messages(session.conversation_id)
        runs = self._runs_from_messages(messages)
        reconciled = False
        for run in list(runs.values()):
            if run.status in {RunStatus.CREATED, RunStatus.RUNNING}:
                try:
                    runs[run.run_id] = self.service.reconcile_run(run.run_id)
                    reconciled = True
                except ConversationServiceError:
                    # An active claim is not safe to replay. Leave it visibly running.
                    logger.info("Run %s is not currently reconcilable", run.run_id)
        if reconciled:
            messages = self.service.list_conversation_messages(
                session.conversation_id
            )
            runs = self._runs_from_messages(messages)

        active = self._active_from_runs(messages, runs)
        session.active_run_id = active.run_id if active is not None else None
        pending = active.pending_review if active is not None else None
        session.pending_review_request_id = (
            pending.review_request_id if pending is not None else None
        )
        latest = self._latest_run(messages, runs)
        allowed = list(pending.allowed_actions) if pending is not None else []
        question_value = session.edit_question_draft
        if latest is None or latest.status != RunStatus.CLOSED_FOR_EDIT:
            session.edit_question_draft = ""
            question_value = ""

        return UiSnapshot(
            session=session,
            conversation_choices=choices,
            chat_messages=self._render_messages(messages),
            status_markdown=self._render_status(latest),
            error_markdown=self._error_markdown(error),
            review_markdown=self._render_review(active),
            review_visible=pending is not None,
            allowed_actions=allowed,
            question_value=question_value,
            question_interactive=active is None,
        )

    def _failure_snapshot(
        self, session: WebSessionState, error: Exception
    ) -> UiSnapshot:
        # Exception messages can contain provider or database URLs. Keep UI and
        # default logs credential-safe; operators can correlate by exception type.
        logger.error("Gradio request failed (%s)", type(error).__name__)
        if isinstance(error, ConversationServiceError):
            message = self._domain_error_message(error)
        elif isinstance(error, ValueError):
            message = str(error)
        else:
            message = "The request could not be completed. Try again shortly."
        try:
            return self._snapshot(session, error=message)
        except Exception as refresh_error:
            logger.error(
                "Could not refresh the Gradio view after an error (%s)",
                type(refresh_error).__name__,
            )
            return UiSnapshot(
                session=session,
                error_markdown=self._error_markdown(message),
            )

    def _conversation_choices(
        self, conversations: list[ConversationRecord]
    ) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for conversation in sorted(
            conversations, key=lambda item: item.updated_at, reverse=True
        ):
            messages = self.service.list_conversation_messages(
                conversation.conversation_id
            )
            first_question = next(
                (item.content for item in messages if item.kind == "question"),
                "New conversation",
            )
            compact = " ".join(first_question.split())
            if len(compact) > 42:
                compact = f"{compact[:39]}..."
            stamp = conversation.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
            choices.append(
                (f"{stamp} · {compact}", str(conversation.conversation_id))
            )
        return choices

    def _runs_from_messages(
        self, messages: list[ConversationMessage]
    ) -> dict[UUID, RunRecord]:
        runs: dict[UUID, RunRecord] = {}
        for message in messages:
            if message.run_id not in runs:
                runs[message.run_id] = self.service.get_run(message.run_id)
        return runs

    @staticmethod
    def _ordered_run_ids(messages: list[ConversationMessage]) -> list[UUID]:
        return list(dict.fromkeys(item.run_id for item in messages))

    def _latest_active_run(self, conversation_id: UUID) -> RunRecord | None:
        messages = self.service.list_conversation_messages(conversation_id)
        runs = self._runs_from_messages(messages)
        return self._active_from_runs(messages, runs)

    def _active_from_runs(
        self,
        messages: list[ConversationMessage],
        runs: dict[UUID, RunRecord],
    ) -> RunRecord | None:
        for run_id in reversed(self._ordered_run_ids(messages)):
            run = runs[run_id]
            if run.status in _ACTIVE_STATUSES:
                return run
        return None

    def _latest_run(
        self,
        messages: list[ConversationMessage],
        runs: dict[UUID, RunRecord],
    ) -> RunRecord | None:
        ordered = self._ordered_run_ids(messages)
        return runs[ordered[-1]] if ordered else None

    @staticmethod
    def _render_messages(
        messages: list[ConversationMessage],
    ) -> list[dict[str, str]]:
        rendered: list[dict[str, str]] = []
        for message in messages:
            if message.kind == "question":
                rendered.append({"role": "user", "content": message.content})
            elif message.kind == "answer":
                citation_text = ""
                if message.citations:
                    citation_text = "\n\n**Citations**\n" + "\n".join(
                        f"- {citation}" for citation in message.citations
                    )
                rendered.append(
                    {"role": "assistant", "content": message.content + citation_text}
                )
            elif message.kind == "human_review_request":
                score = message.metadata.get("evaluation_total_score")
                score_text = f" (evaluation score: {score}/60)" if score is not None else ""
                rendered.append(
                    {
                        "role": "assistant",
                        "content": f"**Human review requested{score_text}**\n\n{message.content}",
                    }
                )
            elif message.kind == "human_feedback":
                action = message.metadata.get("action", "review")
                label = _ACTION_LABELS.get(action, str(action))
                rendered.append(
                    {
                        "role": "user",
                        "content": f"**Review action: {label}**\n\n{message.content}",
                    }
                )
            elif message.kind == "run_status":
                termination = message.metadata.get("termination_reason")
                detail = f" · termination: `{termination}`" if termination else ""
                rendered.append(
                    {
                        "role": "assistant",
                        "content": f"Run status: **{message.content}**{detail}",
                    }
                )
        return rendered

    @staticmethod
    def _render_status(run: RunRecord | None) -> str:
        if run is None:
            return "Enter a new question."
        parts = [f"**Run status:** `{run.status.value}`"]
        if run.termination_reason:
            parts.append(f"**Termination:** `{run.termination_reason}`")
        if run.source_count:
            source_types = ", ".join(run.source_types) or "unknown"
            parts.append(f"**Evidence:** {run.source_count} source(s) ({source_types})")
        if run.evaluation_total_score is not None:
            parts.append(f"**Evaluation:** {run.evaluation_total_score}/60")
        if run.status == RunStatus.FAILED:
            parts.append("The run did not complete. Its record is retained for audit.")
        if run.status == RunStatus.CLOSED_FOR_EDIT:
            parts.append("The run is closed. Edit the question below to start a new run.")
        return "  \n".join(parts)

    @staticmethod
    def _render_review(run: RunRecord | None) -> str:
        if run is None or run.pending_review is None:
            return ""
        pending = run.pending_review
        actions = ", ".join(_ACTION_LABELS[item] for item in pending.allowed_actions)
        score = (
            f"\n\nEvaluation score: **{pending.evaluation_total_score}/60**"
            if pending.evaluation_total_score is not None
            else ""
        )
        return f"**Review reason**\n\n{pending.reason}{score}\n\nAllowed actions: {actions}"

    @staticmethod
    def _error_markdown(message: str) -> str:
        return f"⚠️ {message}" if message else ""

    @staticmethod
    def _domain_error_message(error: ConversationServiceError) -> str:
        if isinstance(error, StaleReviewRequestError):
            return "The review request was already handled or has changed."
        messages = {
            "ConversationNotFoundError": "Conversation not found. Refresh the list.",
            "RunNotFoundError": "Run record not found.",
            "InvalidRunTransitionError": "This action is invalid for the current run state.",
            "InvalidReviewActionError": "The review action or feedback is invalid.",
            "IdempotencyConflictError": "This duplicate request differs from the original.",
            "ReviewClaimConflictError": "Another request is processing this review.",
            "RunExecutionError": "The research run failed. Review its run record.",
            "ReconciliationError": "The saved run could not be safely reconciled.",
        }
        return messages.get(type(error).__name__, "The request cannot be processed.")

    def _claim_conversation(self, conversation_id: UUID) -> bool:
        with self._execution_lock:
            if conversation_id in self._executing_conversations:
                return False
            self._executing_conversations.add(conversation_id)
            return True

    def _release_conversation(self, conversation_id: UUID) -> None:
        with self._execution_lock:
            self._executing_conversations.discard(conversation_id)

    @staticmethod
    def _state(
        state: WebSessionState | dict[str, Any] | None,
    ) -> WebSessionState:
        if isinstance(state, WebSessionState):
            return state.model_copy(deep=True)
        return WebSessionState.model_validate(state or {})


def _snapshot_outputs(snapshot: UiSnapshot) -> tuple[Any, ...]:
    """Convert the framework-neutral view to Gradio component updates."""

    selected = (
        str(snapshot.session.conversation_id)
        if snapshot.session.conversation_id is not None
        else None
    )
    enabled = set(snapshot.allowed_actions)
    return (
        gr.update(choices=snapshot.conversation_choices, value=selected),
        snapshot.chat_messages,
        snapshot.status_markdown,
        snapshot.error_markdown,
        gr.update(visible=snapshot.review_visible),
        snapshot.review_markdown,
        "",
        gr.update(interactive="approve" in enabled),
        gr.update(interactive="revise" in enabled),
        gr.update(interactive="research_again" in enabled),
        gr.update(interactive="edit_question" in enabled),
        gr.update(
            value=snapshot.question_value,
            interactive=snapshot.question_interactive,
        ),
        gr.update(interactive=snapshot.question_interactive),
        snapshot.session.model_dump(mode="json"),
    )


@dataclass
class GradioWebApp:
    """Own the Gradio UI and, optionally, its service context manager."""

    demo: gr.Blocks
    controller: GradioController
    _service_context: AbstractContextManager[ConversationService] | None = None
    _closed: bool = field(default=False, init=False)

    def launch(self, **kwargs: Any) -> Any:
        return self.demo.launch(**kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.demo.close(verbose=False)
        if self._service_context is not None:
            self._service_context.__exit__(None, None, None)


def create_gradio_app(
    *,
    service: ConversationServiceApi | None = None,
    storage_mode: StorageMode = "postgres",
    provider: str | None = None,
    model: str | None = None,
    human_review_policy: Literal[
        "never", "on_reviewer_request", "always"
    ] = "on_reviewer_request",
) -> GradioWebApp:
    """Build one service-backed Gradio application for the process lifetime."""

    service_context: AbstractContextManager[ConversationService] | None = None
    if service is None:
        service_context = open_conversation_service(
            storage_mode=storage_mode,
            provider=provider,
            model=model,
            human_review_policy=human_review_policy,
        )
        service = service_context.__enter__()
    controller = GradioController(service)

    with gr.Blocks(title="FDA Regulatory Research Assistant") as demo:
        session_state = gr.State(WebSessionState().model_dump(mode="json"))
        gr.Markdown(
            "# FDA Regulatory Research Assistant\n"
            "Research FDA regulations, guidance, and supplemental web evidence."
        )
        with gr.Row():
            conversation_selector = gr.Dropdown(
                label="Conversation", choices=[], interactive=True, scale=4
            )
            new_conversation = gr.Button("New Conversation", variant="secondary", scale=1)

        chatbot = gr.Chatbot(
            label="Conversation History",
            height=520,
            layout="bubble",
            buttons=["copy", "copy_all"],
        )
        status = gr.Markdown("Select a conversation or create a new one.")
        error = gr.Markdown("")

        with gr.Group(visible=False) as review_panel:
            gr.Markdown("## Human Review")
            review_context = gr.Markdown("")
            review_feedback = gr.Textbox(
                label="Feedback",
                placeholder="Feedback is required to revise, research again, or edit the question.",
                lines=3,
            )
            with gr.Row():
                approve_button = gr.Button("Approve", variant="primary")
                revise_button = gr.Button("Revise Answer")
                research_button = gr.Button("Research Again")
                edit_button = gr.Button("Edit Question")

        question = gr.Textbox(
            label="New Question",
            placeholder="Example: What must informed consent contain?",
            lines=3,
        )
        submit_button = gr.Button("Start Research", variant="primary")

        outputs = [
            conversation_selector,
            chatbot,
            status,
            error,
            review_panel,
            review_context,
            review_feedback,
            approve_button,
            revise_button,
            research_button,
            edit_button,
            question,
            submit_button,
            session_state,
        ]

        def render(snapshot: UiSnapshot) -> tuple[Any, ...]:
            return _snapshot_outputs(snapshot)

        demo.load(
            lambda state: render(controller.initialize(state)),
            inputs=session_state,
            outputs=outputs,
            show_progress="hidden",
        )
        new_conversation.click(
            lambda state: render(controller.create_conversation(state)),
            inputs=session_state,
            outputs=outputs,
            show_progress="hidden",
        )
        conversation_selector.change(
            lambda selected, state: render(
                controller.select_conversation(selected, state)
            ),
            inputs=[conversation_selector, session_state],
            outputs=outputs,
            show_progress="hidden",
        )

        def submit_question(value: str, state: dict[str, Any]) -> tuple[Any, ...]:
            return render(controller.submit_question(value, state))

        for trigger in (question.submit, submit_button.click):
            trigger(
                submit_question,
                inputs=[question, session_state],
                outputs=outputs,
                show_progress="full",
                concurrency_limit=4,
                concurrency_id="regulatory-question",
            )

        def bind_review(button: gr.Button, action: HumanReviewAction) -> None:
            button.click(
                lambda feedback, state, selected_action=action: render(
                    controller.submit_review(selected_action, feedback, state)
                ),
                inputs=[review_feedback, session_state],
                outputs=outputs,
                show_progress="full",
                concurrency_limit=4,
                concurrency_id="regulatory-review",
            )

        bind_review(approve_button, "approve")
        bind_review(revise_button, "revise")
        bind_review(research_button, "research_again")
        bind_review(edit_button, "edit_question")

    app = GradioWebApp(
        demo=demo,
        controller=controller,
        _service_context=service_context,
    )
    if service_context is not None:
        atexit.register(app.close)
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the regulatory Gradio MVP.")
    parser.add_argument(
        "--storage-mode", choices=("memory", "postgres"), default="postgres"
    )
    parser.add_argument(
        "--provider", choices=("ollama", "groq", "openai"), default=None
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--human-review-policy",
        choices=("never", "on_reviewer_request", "always"),
        default="on_reviewer_request",
    )
    return parser


def _configure_trajectory_logging() -> None:
    """Print graph-node progress without enabling noisy dependency logging."""

    if not trajectory_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [trajectory] %(message)s"))
        trajectory_logger.addHandler(handler)
    trajectory_logger.setLevel(logging.INFO)
    trajectory_logger.propagate = False


def main() -> None:
    args = _parser().parse_args()
    _configure_trajectory_logging()
    app = create_gradio_app(
        storage_mode=args.storage_mode,
        provider=args.provider,
        model=args.model,
        human_review_policy=args.human_review_policy,
    )
    try:
        app.demo.queue(default_concurrency_limit=4).launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=False,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
