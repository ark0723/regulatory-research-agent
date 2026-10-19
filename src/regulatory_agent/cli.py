"""Command-line composition for routing and the complete multi-agent graph."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Literal

from dotenv import load_dotenv

from regulatory_agent.evaluation import ToolCallingAnswerEvaluator
from regulatory_agent.multi_agent import EvaluationRetryPolicy, RegulatoryMultiAgent
from regulatory_agent.rerank import BGEReranker
from regulatory_agent.routing import RegulatoryQuestionRouter, RouteTarget
from regulatory_agent.runtime import create_chat_model, create_existing_vector_store
from regulatory_agent.specialist_agent import CorrectiveRegulatorySpecialist
from regulatory_agent.tools import create_regulatory_search_tools
from regulatory_agent.web_research import TavilyWebResearchAgent


def create_question_router(
    *, provider: str | None = None, model: str | None = None
) -> RegulatoryQuestionRouter:
    """Create only the lightweight structured question router."""

    return RegulatoryQuestionRouter(
        llm=create_chat_model(provider=provider, model=model)
    )


def create_multi_agent(
    *,
    provider: str | None = None,
    model: str | None = None,
    human_review_policy: Literal["never", "on_reviewer_request", "always"] = (
        "on_reviewer_request"
    ),
    retry_policy: EvaluationRetryPolicy | None = None,
    available_research_targets: set[RouteTarget] | None = None,
    checkpointer: Any | None = None,
) -> RegulatoryMultiAgent:
    """Compose regulation, guidance, and web nodes with final synthesis."""

    llm = create_chat_model(provider=provider, model=model)
    store = create_existing_vector_store()
    reranker = BGEReranker()
    regulation_agent = CorrectiveRegulatorySpecialist(
        llm=llm,
        document_type="regulation",
        store=store,
        reranker=reranker,
        candidate_k=15,
        top_n=5,
        max_rewrites=1,
    )
    guidance_agent = CorrectiveRegulatorySpecialist(
        llm=llm,
        document_type="guidance",
        store=store,
        reranker=reranker,
        candidate_k=15,
        top_n=5,
        max_rewrites=1,
    )
    web_agent = TavilyWebResearchAgent(
        reranker=reranker,
        candidate_k=10,
        extract_k=5,
        top_n=3,
    )
    evaluation_tools = [
        *create_regulatory_search_tools(
            store=store,
            reranker=reranker,
            candidate_k=15,
            top_n=5,
        ),
        web_agent.as_tool(),
    ]
    return RegulatoryMultiAgent(
        llm=llm,
        router=RegulatoryQuestionRouter(llm=llm),
        regulation_agent=regulation_agent,
        guidance_agent=guidance_agent,
        web_agent=web_agent,
        evaluator=ToolCallingAnswerEvaluator(llm=llm, tools=evaluation_tools),
        human_review_policy=human_review_policy,
        retry_policy=retry_policy,
        available_research_targets=available_research_targets,
        checkpointer=checkpointer,
    )


def create_research_coordinator(
    *,
    provider: str | None = None,
    model: str | None = None,
    human_review_policy: Literal["never", "on_reviewer_request", "always"] = (
        "on_reviewer_request"
    ),
    retry_policy: EvaluationRetryPolicy | None = None,
    available_research_targets: set[RouteTarget] | None = None,
    checkpointer: Any | None = None,
) -> RegulatoryMultiAgent:
    """Backward-compatible composition alias for the completed parent graph."""

    return create_multi_agent(
        provider=provider,
        model=model,
        human_review_policy=human_review_policy,
        retry_policy=retry_policy,
        available_research_targets=available_research_targets,
        checkpointer=checkpointer,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route or answer an FDA question with source-specialized agents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    route_parser = subparsers.add_parser("route")
    research_parser = subparsers.add_parser("research")
    for command_parser in (route_parser, research_parser):
        command_parser.add_argument("question")
        command_parser.add_argument(
            "--provider", choices=("ollama", "groq", "openai"), default=None
        )
        command_parser.add_argument("--model", default=None)
    research_parser.add_argument(
        "--human-review-policy",
        choices=("never", "on_reviewer_request", "always"),
        default="on_reviewer_request",
    )
    research_parser.add_argument(
        "--no-interactive-review",
        action="store_true",
        help="Return an awaiting_human_review result without prompting in the CLI.",
    )
    return parser


def _interactive_resume(agent: RegulatoryMultiAgent, result):
    """Collect terminal input outside graph nodes and resume the checkpoint."""

    request = result.human_review_request
    print("\n--- Human review required ---")
    print(request.final_answer)
    if request.evaluation:
        print(f"\nScore: {request.evaluation.total_score}/60")
        print(request.evaluation.brief_evaluation)
    approved = input("\nApprove this answer? (y/n): ").strip().lower() == "y"
    feedback = input("Feedback (required for rejection): ").strip()
    while not approved and not feedback:
        feedback = input("Please explain why the answer is rejected: ").strip()
    return agent.resume_human_review(
        thread_id=result.thread_id,
        approved=approved,
        feedback=feedback,
    )


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()
    if args.command == "route":
        result = create_question_router(
            provider=args.provider, model=args.model
        ).route(args.question)
    else:
        agent = create_multi_agent(
            provider=args.provider,
            model=args.model,
            human_review_policy=args.human_review_policy,
        )
        result = agent.ask(args.question)
        if (
            result.termination_reason == "awaiting_human_review"
            and not args.no_interactive_review
            and sys.stdin.isatty()
        ):
            result = _interactive_resume(agent, result)
    print(result.model_dump_json(indent=2))
