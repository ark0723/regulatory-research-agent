"""Tavily web research agent with deterministic BGE reranking.
→ URL 중복 제거
→ BGE snippet rerank
→ 상위 URL만 Tavily Extract
→ BGE extracted-content rerank
→ W1, W2, W3 evidence 반환

<Use Exaple>
from regulatory_agent.rerank import BGEReranker
from regulatory_agent.web_research import TavilyWebResearchAgent

web_agent = TavilyWebResearchAgent(
    reranker=BGEReranker(),
    candidate_k=10,
    extract_k=5,
    top_n=3,
    search_depth="advanced",
    extract_depth="advanced",
)

# 구조화된 결과
result = web_agent.ask(
    "FDA decentralized clinical trial practical implementation risks",
    include_domains=["fda.gov", "nih.gov"],
)

for source in result.sources:
    print(
        source.evidence_id,
        source.title,
        source.url,
        source.rerank_score,
    )

# 상위 Agent가 호출할 수 있는 Tool
web_research_tool = web_agent.as_tool()

output = web_research_tool.invoke(
    {
        "question": "FDA decentralized clinical trial practical implementation risks",
        "include_domains": ["fda.gov", "nih.gov"],
    }
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, TypedDict
from urllib.parse import urlsplit, urlunsplit

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilyExtract, TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from regulatory_agent.rerank import EvidenceReranker


TimeRange = Literal["day", "week", "month", "year"]


class InvokableTool(Protocol):
    """Minimal injected interface shared by Tavily tools and test doubles."""

    def invoke(self, input: dict[str, Any]) -> Any:
        """Invoke a tool with a dictionary payload."""


class WebResearchRequest(BaseModel):
    """Inputs exposed to parent agents through the web-research tool."""

    question: str = Field(min_length=1)
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None


class WebSearchCandidate(BaseModel):
    """One normalized, URL-deduplicated Tavily search result."""

    title: str
    url: str
    content: str
    search_rank: int
    tavily_score: float | None = None


class PreliminaryWebResult(WebSearchCandidate):
    """Search snippet after the first cross-encoder reranking pass."""

    initial_rerank_score: float
    initial_rerank_rank: int


class WebEvidence(PreliminaryWebResult):
    """Final web evidence returned to a calling agent."""

    evidence_id: str
    content: str
    extracted: bool
    rerank_score: float
    rerank_rank: int
    selection_rank: int


class WebResearchResult(BaseModel):
    """Structured result retained beneath the text-oriented tool wrapper."""

    question: str
    candidate_count: int
    initially_selected_count: int
    sources: list[WebEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    search_response_time: float | None = None
    extract_response_time: float | None = None


class WebResearchState(TypedDict, total=False):
    """LangGraph state for search, extraction, and two-stage reranking."""

    question: str
    include_domains: list[str]
    exclude_domains: list[str]
    time_range: TimeRange | None
    candidates: list[WebSearchCandidate]
    preliminary: list[PreliminaryWebResult]
    enriched: list[tuple[PreliminaryWebResult, str, bool]]
    sources: list[WebEvidence]
    warnings: list[str]
    search_response_time: float | None
    extract_response_time: float | None


def _url_key(url: str) -> str:
    parsed = urlsplit(url.strip())
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path.rstrip("/") or "/",
            parsed.query,
            "",
        )
    )


def _response_mapping(response: Any, *, stage: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError(f"Tavily {stage} returned a non-object response")
    if response.get("error"):
        raise ValueError(f"Tavily {stage} failed: {response['error']}")
    return response


def _score_documents(
    *,
    question: str,
    candidates: Sequence[WebSearchCandidate],
    contents: Sequence[str],
    reranker: EvidenceReranker,
) -> list[float]:
    documents = [
        Document(
            page_content=f"{candidate.title}\n{content}",
            metadata={"url": candidate.url},
        )
        for candidate, content in zip(candidates, contents, strict=True)
    ]
    scores = [float(score) for score in reranker.score(question, documents)]
    if len(scores) != len(candidates):
        raise ValueError(
            "reranker must return exactly one score for every web candidate"
        )
    return scores


def render_web_research_result(result: WebResearchResult) -> str:
    """Render traceable web evidence for a parent agent tool call."""

    if not result.sources:
        details = "; ".join(result.warnings) or "No relevant web evidence was found."
        return f"No reranked web evidence was found. {details}"

    sections = [
        "Web evidence scope: public web sources; verify authority and date before use.",
        f"Question: {result.question}",
        (
            "Diagnostics: "
            f"candidates={result.candidate_count}; "
            f"initially_selected={result.initially_selected_count}; "
            f"final_sources={len(result.sources)}"
        ),
    ]
    if result.warnings:
        sections.extend(["Warnings:", *[f"- {warning}" for warning in result.warnings]])
    for source in result.sources:
        sections.extend(
            [
                "",
                f"<web_evidence id=\"{source.evidence_id}\">",
                f"title: {source.title}",
                f"url: {source.url}",
                f"search_rank: {source.search_rank}",
                f"tavily_score: {source.tavily_score}",
                f"initial_rerank_rank: {source.initial_rerank_rank}",
                f"rerank_rank: {source.rerank_rank}",
                f"selection_rank: {source.selection_rank}",
                f"extracted: {source.extracted}",
                "--- BEGIN WEB EVIDENCE TEXT ---",
                source.content,
                "--- END WEB EVIDENCE TEXT ---",
                "</web_evidence>",
            ]
        )
    return "\n".join(sections)


class TavilyWebResearchAgent:
    """Search, extract selected pages, rerank, and expose results as a tool.

    Tavily's score is retained as retrieval metadata only. BGE cross-encoder
    scores determine both which URLs are extracted and which sources are finally
    returned. The class does not generate an answer or make legal conclusions.
    """

    def __init__(
        self,
        *,
        reranker: EvidenceReranker,
        search_tool: InvokableTool | None = None,
        extract_tool: InvokableTool | None = None,
        candidate_k: int = 10,
        extract_k: int = 5,
        top_n: int = 3,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "advanced",
        extract_depth: Literal["basic", "advanced"] = "advanced",
        extract_chunks_per_source: int = 3,
        max_content_chars: int = 6_000,
        use_extract: bool = True,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if extract_k <= 0 or extract_k > candidate_k:
            raise ValueError("extract_k must be between 1 and candidate_k")
        if top_n <= 0 or top_n > extract_k:
            raise ValueError("top_n must be between 1 and extract_k")
        if not 1 <= extract_chunks_per_source <= 5:
            raise ValueError("extract_chunks_per_source must be between 1 and 5")
        if max_content_chars <= 0:
            raise ValueError("max_content_chars must be positive")

        self.reranker = reranker
        self.candidate_k = candidate_k
        self.extract_k = extract_k
        self.top_n = top_n
        self.max_content_chars = max_content_chars
        self._search_tool = search_tool or TavilySearch(
            max_results=candidate_k,
            topic="general",
            search_depth=search_depth,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
        if extract_tool is not None:
            self._extract_tool: InvokableTool | None = extract_tool
        elif use_extract:
            self._extract_tool = TavilyExtract(
                extract_depth=extract_depth,
                include_images=False,
                format="markdown",
                chunks_per_source=extract_chunks_per_source,
            )
        else:
            self._extract_tool = None
        self.graph = self._compile_graph()

    def search_web(self, state: WebResearchState) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": state["question"]}
        if state.get("include_domains"):
            payload["include_domains"] = state["include_domains"]
        if state.get("exclude_domains"):
            payload["exclude_domains"] = state["exclude_domains"]
        if state.get("time_range"):
            payload["time_range"] = state["time_range"]

        warnings = list(state.get("warnings", []))
        try:
            response = _response_mapping(
                self._search_tool.invoke(payload), stage="search"
            )
        except Exception as error:
            return {
                "candidates": [],
                "warnings": [*warnings, str(error)],
            }

        candidates: list[WebSearchCandidate] = []
        seen_urls: set[str] = set()
        for search_rank, item in enumerate(response.get("results", []), start=1):
            if not isinstance(item, dict):
                warnings.append(f"Search result {search_rank} was not an object.")
                continue
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            if not url or not content:
                warnings.append(
                    f"Search result {search_rank} lacked a URL or content snippet."
                )
                continue
            key = _url_key(url)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            raw_score = item.get("score")
            candidates.append(
                WebSearchCandidate(
                    title=str(item.get("title") or url),
                    url=url,
                    content=content[: self.max_content_chars],
                    search_rank=search_rank,
                    tavily_score=(
                        float(raw_score) if raw_score is not None else None
                    ),
                )
            )
        return {
            "candidates": candidates,
            "warnings": warnings,
            "search_response_time": response.get("response_time"),
        }

    def rerank_search_results(self, state: WebResearchState) -> dict[str, Any]:
        candidates = state.get("candidates", [])
        if not candidates:
            return {"preliminary": []}
        scores = _score_documents(
            question=state["question"],
            candidates=candidates,
            contents=[candidate.content for candidate in candidates],
            reranker=self.reranker,
        )
        globally_ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        preliminary = [
            PreliminaryWebResult(
                **candidate.model_dump(),
                initial_rerank_score=score,
                initial_rerank_rank=rank,
            )
            for rank, (candidate, score) in enumerate(globally_ranked, start=1)
        ]
        return {"preliminary": preliminary[: self.extract_k]}

    def extract_selected_pages(self, state: WebResearchState) -> dict[str, Any]:
        preliminary = state.get("preliminary", [])
        if not preliminary:
            return {"enriched": []}
        if self._extract_tool is None:
            return {
                "enriched": [
                    (candidate, candidate.content, False)
                    for candidate in preliminary
                ]
            }

        warnings = list(state.get("warnings", []))
        urls = [candidate.url for candidate in preliminary]
        try:
            response = _response_mapping(
                self._extract_tool.invoke(
                    {"urls": urls, "query": state["question"]}
                ),
                stage="extract",
            )
        except Exception as error:
            return {
                "enriched": [
                    (candidate, candidate.content, False)
                    for candidate in preliminary
                ],
                "warnings": [
                    *warnings,
                    f"Extract failed; search snippets were retained. {error}",
                ],
            }

        extracted_by_url = {
            _url_key(str(item.get("url") or "")): str(
                item.get("raw_content") or ""
            ).strip()
            for item in response.get("results", [])
            if isinstance(item, dict) and item.get("url")
        }
        for failed in response.get("failed_results", []):
            if isinstance(failed, dict):
                warnings.append(
                    "Extract failed for "
                    f"{failed.get('url', 'an unknown URL')}: "
                    f"{failed.get('error', 'unknown error')}"
                )

        enriched: list[tuple[PreliminaryWebResult, str, bool]] = []
        for candidate in preliminary:
            extracted = extracted_by_url.get(_url_key(candidate.url), "")
            if extracted:
                enriched.append(
                    (candidate, extracted[: self.max_content_chars], True)
                )
            else:
                enriched.append((candidate, candidate.content, False))
                warnings.append(
                    f"No extracted content for {candidate.url}; search snippet retained."
                )
        return {
            "enriched": enriched,
            "warnings": list(dict.fromkeys(warnings)),
            "extract_response_time": response.get("response_time"),
        }

    def rerank_extracted_evidence(
        self, state: WebResearchState
    ) -> dict[str, Any]:
        enriched = state.get("enriched", [])
        if not enriched:
            return {"sources": []}
        candidates = [item[0] for item in enriched]
        contents = [item[1] for item in enriched]
        scores = _score_documents(
            question=state["question"],
            candidates=candidates,
            contents=contents,
            reranker=self.reranker,
        )
        globally_ranked = sorted(
            zip(enriched, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        sources: list[WebEvidence] = []
        for rerank_rank, ((candidate, content, extracted), score) in enumerate(
            globally_ranked, start=1
        ):
            if len(sources) == self.top_n:
                break
            selection_rank = len(sources) + 1
            sources.append(
                WebEvidence(
                    **candidate.model_dump(exclude={"content"}),
                    evidence_id=f"W{selection_rank}",
                    content=content,
                    extracted=extracted,
                    rerank_score=score,
                    rerank_rank=rerank_rank,
                    selection_rank=selection_rank,
                )
            )
        return {"sources": sources}

    def _compile_graph(self):
        workflow = StateGraph(WebResearchState)
        workflow.add_node("search_web", self.search_web)
        workflow.add_node("rerank_search_results", self.rerank_search_results)
        workflow.add_node("extract_selected_pages", self.extract_selected_pages)
        workflow.add_node("rerank_extracted_evidence", self.rerank_extracted_evidence)
        workflow.add_edge(START, "search_web")
        workflow.add_edge("search_web", "rerank_search_results")
        workflow.add_edge("rerank_search_results", "extract_selected_pages")
        workflow.add_edge("extract_selected_pages", "rerank_extracted_evidence")
        workflow.add_edge("rerank_extracted_evidence", END)
        return workflow.compile()

    def ask(
        self,
        question: str,
        *,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        time_range: TimeRange | None = None,
    ) -> WebResearchResult:
        """Run the bounded web-research graph and return structured evidence."""

        if not question.strip():
            raise ValueError("question must not be empty")
        request = WebResearchRequest(
            question=question,
            include_domains=list(include_domains or []),
            exclude_domains=list(exclude_domains or []),
            time_range=time_range,
        )
        state = self.graph.invoke(
            {
                **request.model_dump(),
                "warnings": [],
            }
        )
        return WebResearchResult(
            question=question,
            candidate_count=len(state.get("candidates", [])),
            initially_selected_count=len(state.get("preliminary", [])),
            sources=state.get("sources", []),
            warnings=state.get("warnings", []),
            search_response_time=state.get("search_response_time"),
            extract_response_time=state.get("extract_response_time"),
        )

    def as_tool(self) -> BaseTool:
        """Expose this bounded research agent as a parent-agent tool."""

        agent = self

        @tool("search_web_with_reranking", args_schema=WebResearchRequest)
        def search_web_with_reranking(
            question: str,
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
            time_range: TimeRange | None = None,
        ) -> str:
            """Search current public web sources and return BGE-reranked evidence.

            Use when the regulatory corpus alone cannot answer practical,
            contextual, current, or implementation-oriented parts of a question.
            Web evidence is supplemental and must not be presented as binding law.
            """

            result = agent.ask(
                question,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                time_range=time_range,
            )
            return render_web_research_result(result)

        return search_web_with_reranking

    def mermaid(self) -> str:
        """Return the web-research graph as Mermaid source."""

        return self.graph.get_graph().draw_mermaid()

    def save_mermaid(self, output_path: Path | str) -> Path:
        """Save Mermaid source for local architecture review."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.mermaid() + "\n", encoding="utf-8")
        return path
