"""Structured question routing and parallel specialist-tool dispatch."""

from __future__ import annotations

from operator import add
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Protocol, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field


RouteTarget = Literal["regulation", "guidance", "web"]
ROUTE_TARGET_ORDER: tuple[RouteTarget, ...] = (
    "regulation",
    "guidance",
    "web",
)


class RouteChoice(BaseModel):
    """One specialist selected for the original user question."""

    target: RouteTarget
    reason: str = Field(min_length=1)


class RoutingDecision(BaseModel):
    """Structured multi-label routing result."""

    routes: list[RouteChoice] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class RoutedToolResult(BaseModel):
    """One specialist tool execution result."""

    target: RouteTarget
    tool_name: str
    success: bool
    content: str = ""
    error: str | None = None


class RoutedResearchResult(BaseModel):
    """Routing decision plus independently returned specialist evidence/answers."""

    question: str
    decision: RoutingDecision
    results: list[RoutedToolResult]


class InvokableResearchTool(Protocol):
    """Minimal interface for specialist tools used by the dispatcher."""

    name: str

    def invoke(self, input: dict[str, Any]) -> Any:
        """Invoke the tool with its question input."""


class RouterState(TypedDict, total=False):
    question: str
    decision: RoutingDecision
    results: Annotated[list[RoutedToolResult], add]


class ToolExecutionState(TypedDict):
    question: str
    choice: RouteChoice


ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You route FDA regulatory research questions to specialist tools.

## Available targets
- regulation: Binding 21 CFR requirements, prohibitions, definitions,
  applicability, exceptions, required elements, or precise regulatory citations.
- guidance: Nonbinding FDA recommendations, interpretation, implementation
  considerations, best practices, or questions explicitly about FDA guidance.
- web: Current or time-sensitive information, recent enforcement or safety
  activity, practical external context, industry implementation, or facts not
  expected to be contained in the indexed CFR and FDA guidance corpus.

## Selection rules
- Select every target materially needed to answer the whole question.
- A regulation question does not need to contain a CFR number. Words such as
  must, required, prohibited, permitted, exception, and applies commonly require
  regulation.
- Select both regulation and guidance when the user asks for binding duties and
  FDA's recommended implementation or interpretation.
- Select web only when freshness or external practical context adds material
  value. Do not use web merely because the user omitted a section number.
- Web information is supplemental and must not be treated as binding law.
- Do not answer the question and do not rewrite it.
- Keep each reason short and specific.

## Examples
- "What must informed consent contain?" → regulation
- "How does FDA recommend implementing electronic consent?" → guidance
- "What is required, and how does FDA recommend implementing it?"
  → regulation + guidance
- "What recent FDA warning letters discuss consent failures?" → web
- "What does the regulation require and what are current industry practices?"
  → regulation + web

Return the required RoutingDecision structured output.""",
        ),
        (
            "human",
            """## User question
{question}

Select the specialist targets needed to research this question.""",
        ),
    ]
)


def _normalize_decision(decision: RoutingDecision) -> RoutingDecision:
    """Deduplicate model output and impose stable architecture-level ordering."""

    choice_by_target: dict[RouteTarget, RouteChoice] = {}
    for choice in decision.routes:
        choice_by_target.setdefault(choice.target, choice)
    routes = [
        choice_by_target[target]
        for target in ROUTE_TARGET_ORDER
        if target in choice_by_target
    ]
    if not routes:
        raise ValueError("router must select at least one supported target")
    return decision.model_copy(update={"routes": routes})


class RegulatoryQuestionRouter:
    """LLM-based, structured multi-label router without tool execution."""

    def __init__(self, *, llm: BaseChatModel) -> None:
        self.llm = llm

    def route(self, question: str) -> RoutingDecision:
        if not question.strip():
            raise ValueError("question must not be empty")
        prompt = ROUTING_PROMPT.invoke({"question": question})
        structured_llm = self.llm.with_structured_output(RoutingDecision)
        decision = structured_llm.invoke(prompt)
        return _normalize_decision(decision)


class RoutedResearchCoordinator:
    """Route once and execute selected specialist tools in parallel.

    This is a bounded routing/dispatch slice, not the future answer synthesizer,
    reviewer, retry coordinator, or human-intervention supervisor.
    """

    def __init__(
        self,
        *,
        router: RegulatoryQuestionRouter,
        tools: Mapping[RouteTarget, InvokableResearchTool],
    ) -> None:
        missing = [target for target in ROUTE_TARGET_ORDER if target not in tools]
        if missing:
            raise ValueError(
                "tools must include regulation, guidance, and web; missing: "
                + ", ".join(missing)
            )
        self.router = router
        self.tools = dict(tools)
        self.graph = self._compile_graph()

    def route_question(self, state: RouterState) -> dict[str, Any]:
        return {"decision": self.router.route(state["question"])}

    def dispatch_tools(self, state: RouterState) -> list[Send]:
        return [
            Send(
                "execute_tool",
                {
                    "question": state["question"],
                    "choice": choice,
                },
            )
            for choice in state["decision"].routes
        ]

    def execute_tool(self, state: ToolExecutionState) -> dict[str, Any]:
        choice = state["choice"]
        selected_tool = self.tools[choice.target]
        try:
            response = selected_tool.invoke({"question": state["question"]})
            content = str(response).strip()
            if not content:
                raise ValueError("tool returned an empty response")
            result = RoutedToolResult(
                target=choice.target,
                tool_name=selected_tool.name,
                success=True,
                content=content,
            )
        except Exception as error:
            result = RoutedToolResult(
                target=choice.target,
                tool_name=selected_tool.name,
                success=False,
                error=str(error),
            )
        return {"results": [result]}

    def _compile_graph(self):
        workflow = StateGraph(RouterState)
        workflow.add_node("route_question", self.route_question)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_edge(START, "route_question")
        workflow.add_conditional_edges(
            "route_question",
            self.dispatch_tools,
            ["execute_tool"],
        )
        workflow.add_edge("execute_tool", END)
        return workflow.compile()

    def ask(self, question: str) -> RoutedResearchResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        state = self.graph.invoke({"question": question, "results": []})
        result_by_target = {result.target: result for result in state["results"]}
        ordered_results = [
            result_by_target[choice.target]
            for choice in state["decision"].routes
            if choice.target in result_by_target
        ]
        return RoutedResearchResult(
            question=question,
            decision=state["decision"],
            results=ordered_results,
        )

    def mermaid(self) -> str:
        """Return the compiled routing/dispatch graph as Mermaid source."""

        return self.graph.get_graph().draw_mermaid()

    def save_mermaid(self, output_path: Path | str) -> Path:
        """Save Mermaid source for local architecture review."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.mermaid() + "\n", encoding="utf-8")
        return path
