# Regulatory Agent

FDA regulations and guidance are normalized into auditable section-level
records, stored in pgvector, and used by source-specialized LangGraph agents
with structured routing and citation-validated final synthesis.

## Project layout

```text
regulatory_agent/
├── src/regulatory_agent/
│   ├── ingestion/          # eCFR extraction and FDA/CFR normalization
│   ├── documents.py        # shared LangChain Document metadata contract
│   ├── chunking.py         # baseline and structure-aware retrieval inputs
│   ├── vector_db.py        # non-truncating embeddings and pgvector access
│   ├── retrieval.py        # scored retrieval and exact section expansion
│   ├── specialist_agent.py # corrective regulation/guidance RAG graph
│   ├── web_research.py     # Tavily search/extract plus BGE reranking
│   ├── routing.py          # structured multi-label question routing
│   ├── multi_agent.py      # explicit source nodes and final synthesis
│   ├── evaluation.py       # tool-assisted structured answer review
│   ├── conversation_service.py # run-isolated application lifecycle
│   ├── postgres_repository.py  # persistent application metadata
│   ├── persistence.py      # storage/checkpointer composition
│   ├── runtime.py          # model and existing-PGVector factories
│   ├── cli.py              # package command-line composition
│   └── paths.py            # repository-relative paths
├── data/
│   ├── raw/                # immutable source documents
│   └── processed/          # normalized JSON datasets
├── tests/                  # parser and Document regression tests
├── notebooks/              # exploratory work; run from the repository root
├── docs/experiments/       # validation reports and experiment notes
├── pyproject.toml
└── uv.lock
```

## Commands

```bash
uv sync
uv run extract-cfr
uv run normalize-cfr
uv run normalize-fda-guidance
uv run build-regulatory-documents
uv run analyze-chunking
# Validate all 330 chunks without touching PostgreSQL:
uv run regulatory-vector audit
# Audit, then upsert the chunks into the regulatory_chunks collection:
uv run regulatory-vector index
# Inspect the top five retrieved chunks, scores, and citations:
uv run regulatory-vector query "What are the basic elements of informed consent?" -k 5
# Inspect only the structured routing decision:
uv run regulatory-agent route "What are the eight basic elements of informed consent?"
# Run selected agents and citation-validated final synthesis:
uv run regulatory-agent research "What are the eight basic elements of informed consent?"
uv run regulatory-persistence setup
uv run python -m unittest discover -s tests -p 'test_*.py'
```

The retrieval corpus is produced only by
`build_chunked_documents(chunk_size=1500, chunk_overlap=200)`. Before indexing,
all chunks are sent to Ollama with `truncate=False`; any failure stops the run
before pgvector is opened. Indexing uses deterministic, collection-namespaced
IDs, so rerunning the command updates the same 330 records rather than adding
duplicates. Audit, indexing, and query embedding all use the same `bge-m3`
model options.

## Regulatory multi-agent graph

The regulation and guidance agents reuse the populated `regulatory_chunks`
collection. Each performs retrieval, BGE reranking, section expansion,
structured evidence evaluation, and a bounded query rewrite. The parent graph
routes the question to one or more explicit source nodes and synthesizes their
validated findings once. It does not rebuild or reindex data.

Configure one chat provider and model in `.env`:

```dotenv
# Use an installed Ollama chat model (bge-m3 is embedding-only):
REGULATORY_CHAT_PROVIDER=ollama
REGULATORY_CHAT_MODEL=<installed-chat-model>

# Or use an already configured provider dependency:
# REGULATORY_CHAT_PROVIDER=groq
# REGULATORY_CHAT_MODEL=<groq-model-name>
# REGULATORY_CHAT_PROVIDER=openai
# REGULATORY_CHAT_MODEL=<openai-model-name>
```

The existing database variables (`POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`) are reused. Set
`OLLAMA_BASE_URL` only when Ollama is not available at its default local URL.
The `regulatory_chunks` collection must already be populated with
`uv run regulatory-vector index`; the agent never reindexes it automatically.

## Persistent conversation service

Phase 4B stores user-visible metadata and LangGraph checkpoints in separate
PostgreSQL schemas. `regulatory_app` contains conversations, runs, messages,
human-review actions, and idempotency receipts. `regulatory_checkpoint`
contains only official LangGraph `PostgresSaver` tables. Existing public
PGVector tables and the `regulatory_chunks` collection are not migrated.

Apply the versioned application migrations and LangGraph migrations explicitly:

```bash
uv run regulatory-persistence setup
```

The setup account needs schema/table creation permission. A normal runtime
account needs DML access to both schemas and its existing read access to the
vector corpus. PostgreSQL mode fails closed if either schema is unavailable;
it never falls back to an in-memory checkpointer.

```python
from regulatory_agent.persistence import open_conversation_service

with open_conversation_service(
    storage_mode="postgres",
    provider="groq",
    model="<model-name>",
) as service:
    conversation = service.create_conversation()
    run = service.start_run(conversation.conversation_id, "Question")
```

Use `storage_mode="memory"` for deterministic tests and process-local
development. Set `REGULATORY_STORAGE_MODE=postgres` for the persistent MVP.
`REGULATORY_APP_DATABASE_URL` may provide a complete psycopg URI; otherwise
the existing `POSTGRES_*` values are used. Keep the context manager open for
the service lifetime because it owns both connection pools.

Human-review submissions use a short database claim. The transaction ends
before graph resume, search, or model calls. A second worker cannot resume the
same request concurrently. `ConversationService.reconcile_run()` compares an
expired claim with the latest checkpoint. It releases the claim only when the
checkpoint demonstrably did not advance, or projects the advanced checkpoint
into application metadata. Missing or inconsistent checkpoints become visible
failed runs and never trigger automatic research replay.

Exactly-once external model/tool execution cannot be guaranteed across a
process crash: an external call may finish before the next checkpoint. The
reconciler therefore prefers an explicit ambiguous failure over replay.

PostgreSQL integration tests require `REGULATORY_TEST_DATABASE_URL` pointing
to a dedicated database whose name contains `test`:

```bash
REGULATORY_TEST_DATABASE_URL=postgresql://.../regulatory_agent_test \
  LANGGRAPH_STRICT_MSGPACK=true \
  uv run python -m unittest tests.test_postgres_persistence -v
```

The provider and model can also be selected for one invocation:

```bash
uv run regulatory-agent research \
  "피험자 동의서에 포함되어야 하는 기본 요소는 무엇인가요?" \
  --provider groq \
  --model <groq-model-name>
```

The command prints structured JSON containing the routing decision, agent
contributions, synthesis sources, final answer, citations, and termination
reason. Binding CFR requirements, nonbinding FDA guidance, and supplemental web
context remain distinct throughout the graph.

## Source-specific retrieval tools

`regulatory_agent.tools` exposes separate CFR and FDA Guidance tools over the
same pgvector collection. Each tool retrieves vector candidates, reranks them,
then performs relevance-ordered, budgeted section expansion. CFR evidence uses
verified top-level paragraph boundaries when possible; otherwise the pipeline
falls back to source chunks and marks budget-limited evidence incomplete.

```python
from regulatory_agent.runtime import create_existing_vector_store
from regulatory_agent.rerank import BGEReranker
from regulatory_agent.tools import create_regulatory_search_tools

store = create_existing_vector_store()
reranker = BGEReranker()  # BAAI/bge-reranker-v2-m3
tools = create_regulatory_search_tools(
    store=store,
    reranker=reranker,
    candidate_k=15,
    top_n=5,
    context_char_budget=12_000,
    max_expanded_chunks=12,
)
```

`min_rerank_score` is optional and disabled by default. BGE scores are explicitly
sigmoid-transformed logits for ranking, not calibrated probabilities; choose any
filter only after evaluating representative queries. Tool diagnostics report
candidate and reranked counts, selected and excluded parents, chunk counts,
approximate context tokens, completeness warnings, and stage latency.
`retrieval_rank` is the original PGVector rank, `rerank_rank` is the global
cross-encoder rank before parent deduplication, and `selection_rank` is the
contiguous rank among selected unique parent sections.

## Corrective specialist graph

`CorrectiveRegulatorySpecialist` implements one bounded specialist workflow. It
retrieves and expands evidence, extracts and evaluates the evidence set through a
structured Pydantic response, then either generates an answer or rewrites the
query and retries. `SpecialistAgentState` is a `TypedDict`; LLM-controlled
payloads use `EvidenceAssessment`, `ExtractedEvidence`, and `QueryRewrite`
Pydantic models. Answer generation also uses a structured `GeneratedAnswer`:
the model emits request-local references such as `[[E1]]`, and deterministic
validation resolves them to citations from the supplied evidence metadata.
Only evidence referenced by validated extracted findings is passed to final
answer generation.

```python
from pathlib import Path

from regulatory_agent.runtime import (
    create_chat_model,
    create_existing_vector_store,
)
from regulatory_agent.rerank import BGEReranker
from regulatory_agent.specialist_agent import CorrectiveRegulatorySpecialist

store = create_existing_vector_store()
reranker = BGEReranker()
llm = create_chat_model(provider="ollama", model="<chat-model>")

regulation_agent = CorrectiveRegulatorySpecialist(
    llm=llm,
    document_type="regulation",
    store=store,
    reranker=reranker,
    candidate_k=15,
    top_n=5,
    max_rewrites=1,
)

result = regulation_agent.ask(
    "피험자 동의서에 반드시 포함되어야 하는 요소는 무엇인가?"
)
print(result["answer"])
print(result["citations"])
print(result["query_history"])

if result["termination_reason"] == "citation_validation_failed":
    for issue in result["citation_validation_issues"]:
        print(issue.code, issue.reference)

# Network-free Mermaid source generated from the compiled LangGraph.
print(regulation_agent.mermaid())
regulation_agent.save_mermaid(
    Path("docs/architecture/generated-regulation-specialist.mmd")
)
```

Use `document_type="guidance"` to create the corresponding FDA Guidance
specialist. The class remains independently usable below the parent graph and
does not implement cross-specialist review or human intervention itself.

`termination_reason="answered"` is returned only after all inline evidence IDs,
CFR references, paragraph specificity, and source URLs pass deterministic
provenance checks. Invalid or unresolved references return
`termination_reason="citation_validation_failed"`, an empty citation list, and
structured issues in `citation_validation_issues`; the invalid draft is not
shown to the user. `supporting_record_ids` remains available for compatible
inputs, but new extraction output should use `supporting_evidence_ids`. Synthetic
paragraph-scope IDs are evidence artifacts and are never accepted as original
database record IDs.

## Tavily web research tool-agent

`TavilyWebResearchAgent` is a supplemental public-web evidence tool. It does not
answer legal questions or reinterpret web pages as binding law. Its bounded
LangGraph performs:

```text
Tavily Search candidates
  → BGE snippet rerank
  → Tavily Extract for selected URLs
  → BGE extracted-content rerank
  → W1, W2, ... web evidence
```

Set the API key used by the official `langchain-tavily` integration:

```dotenv
TAVILY_API_KEY=<your-tavily-api-key>
```

```python
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

# Structured use during development or evaluation.
result = web_agent.ask(
    "FDA decentralized clinical trial practical implementation risks",
    include_domains=["fda.gov", "nih.gov"],
)
for source in result.sources:
    print(source.evidence_id, source.title, source.url, source.rerank_score)

# Text-oriented tool for a future parent agent or supervisor.
web_research_tool = web_agent.as_tool()
tool_output = web_research_tool.invoke(
    {
        "question": "FDA decentralized clinical trial practical implementation risks",
        "include_domains": ["fda.gov", "nih.gov"],
    }
)

print(web_agent.mermaid())
web_agent.save_mermaid("docs/architecture/web-research-agent.mmd")
```

Tavily's own result score is retained for diagnostics, but it does not determine
the returned order. BGE reranker scores select URLs for extraction and determine
the final order. If extraction fails for a selected page, its search snippet is
retained with an explicit warning. `search_depth="advanced"` favors precision
but consumes more Tavily credits than `"basic"`; both are configurable.

## Question routing, parallel research, and final synthesis

`RegulatoryQuestionRouter` produces a structured, multi-label decision over
three stable route keys:

- `regulation`: binding duties, definitions, applicability, exceptions, and CFR citations
- `guidance`: nonbinding FDA recommendations and implementation considerations
- `web`: current, practical, external, or time-sensitive information

The router does not require a user to name a CFR section. For example, “What
must informed consent contain?” still routes to `regulation`. Multiple targets
can be selected. `RegulatoryMultiAgent` runs selected explicit source nodes in
parallel, merges their structured evidence, and invokes `generate_answer` once.
The final model may cite only request-local `[[S1]]` markers; deterministic
validation resolves those markers to supplied legal citations or web URLs. A
tool-assisted `evaluate_answer` node then scores successful answers on six
dimensions and recommends the next orchestration action.

```python
from regulatory_agent.multi_agent import RegulatoryMultiAgent
from regulatory_agent.routing import RegulatoryQuestionRouter

agent = RegulatoryMultiAgent(
    llm=llm,
    router=RegulatoryQuestionRouter(llm=llm),
    regulation_agent=regulation_agent,
    guidance_agent=guidance_agent,
    web_agent=web_agent,
)

result = agent.ask(
    "What does 21 CFR require, how does FDA recommend implementing it, "
    "and what are current industry practices?"
)

print(result.final_answer)
print(result.citations)
print(result.termination_reason)
if result.evaluation:
    print(result.evaluation.total_score)
    print(result.evaluation.decision)
    print(result.evaluation.tools_used)

# Network-free visualization of the compiled parent graph.
print(agent.mermaid())
agent.save_mermaid("docs/architecture/regulatory-multi-agent.mmd")
```

The repository-level example provides separate commands so route evaluation
does not initialize PostgreSQL, BGE, or Tavily:

```bash
uv run python main.py route "What must informed consent contain?"
uv run python main.py research \
  "What is required and what are current implementation practices?"
```

The completed parent graph can be inspected in
`docs/architecture/regulatory-multi-agent.mmd`; the lower-level routing/dispatch
reference remains in `docs/architecture/question-routing-dispatch.mmd`.

## Tool-assisted answer evaluation

`ToolCallingAnswerEvaluator` uses LangChain's current `create_agent` API to run
a bounded tool-calling review. The tools supplied by the CLI are direct
`@tool` search interfaces, not the regulation or guidance specialist agents:

- `search_cfr_regulations`
- `search_fda_guidance`
- `search_web_with_reranking`

```python
from regulatory_agent.evaluation import ToolCallingAnswerEvaluator
from regulatory_agent.tools import create_regulatory_search_tools

evaluation_tools = [
    *create_regulatory_search_tools(store=store, reranker=reranker),
    web_agent.as_tool(),
]
evaluator = ToolCallingAnswerEvaluator(llm=llm, tools=evaluation_tools)

evaluation = evaluator.evaluate(
    question=result.question,
    answer=result.final_answer,
    citations=result.citations,
    sources=result.sources,
)
print(evaluation.model_dump_json(indent=2))
print(evaluator.mermaid())
```

The total is calculated in code from the six 1–10 scores. Decisions use stable
rules: unresolved high-stakes ambiguity becomes `human_review`, an evidence gap
becomes `research_again`, a sufficiently strong grounded answer becomes `pass`,
and other correctable answer-quality problems become `revise`. A `revise`
decision now starts a bounded answer-only revision using the existing evidence
and evaluation feedback. The revised `FinalAnswerDraft` must pass the same
deterministic citation validator as the initial answer, then returns to
evaluation. It does not run retrieval or call search tools. `research_again`
prepares and validates a targeted `ResearchRetryPlan`, executes only its listed
specialist/query pairs, merges new evidence with the existing catalog,
regenerates the answer, validates citations, and returns to evaluation.

Retry policy defaults are configurable without changing graph topology:

```python
from regulatory_agent.multi_agent import EvaluationRetryPolicy

agent = create_multi_agent(
    retry_policy=EvaluationRetryPolicy(
        max_total_automatic_retries=3,
        max_answer_revisions=2,
        max_research_retries=1,
        max_research_queries_per_plan=3,
        max_merged_synthesis_sources=20,
    )
)
```

The result exposes `evaluation_retry_count`, `revision_count`,
`research_retry_count`, and a compact `evaluation_history`. Counters represent
retry attempts that actually began. Entering `revise_answer` increments the
total and revision counters atomically and records `revision_started`; a blocked
attempt does not increment them. A research attempt increments the total and
research counters exactly once after deterministic plan validation and before
selective query dispatch. Planning failures do not increment either counter.

Inspect the executed plan and its compact per-query diagnostic directly:

```python
if result.research_retry_plan is not None:
    print(result.research_retry_plan.model_dump_json(indent=2))
    for diagnostic in result.research_retry_diagnostics:
        print(diagnostic.model_dump_json(indent=2))
if result.termination_reason == "research_retry_planning_failed":
    print(result.research_retry_error)
```

Each plan contains `evidence_gaps` and per-target `queries` with a specialist
target, focused query, and reason. The default policy permits at most three
queries per plan through `max_research_queries_per_plan`. Existing evidence is
preserved, stable record IDs or web URLs drive deduplication, and the merged
catalog is capped by `max_merged_synthesis_sources`. Partial or complete query
failures return `research_retry_partial_failure` or `research_retry_failed`
without broad fallback research.

## Human approval and resume

The parent graph uses LangGraph `interrupt()` rather than calling `input()`
inside a node. A checkpointer stores the paused graph state, and the caller
resumes it with the same `thread_id`:

```python
from regulatory_agent.cli import create_multi_agent

agent = create_multi_agent(human_review_policy="on_reviewer_request")
result = agent.ask(question, thread_id="review-case-001")

if result.termination_reason == "awaiting_human_review":
    request = result.human_review_request
    print(request.final_answer)
    print(request.evaluation.total_score)
    print(request.reason)

    result = agent.resume_human_review(
        thread_id=result.thread_id,
        action="approve",
        feedback="Applicability facts were confirmed.",
    )

print(result.termination_reason)  # human_approved
```

Available policies are:

- `on_reviewer_request`: pause only when evaluation returns `human_review`
- `always`: require approval for every successfully generated answer
- `never`: disable HITL pauses; a `human_review` decision ends explicitly as
  `human_review_required` rather than being reported as `answered`

The default in-memory checkpointer supports a single running Python process.
Applications that need resume after a process restart should inject a durable
LangGraph checkpointer through `create_multi_agent(checkpointer=...)`. A rejected
answer requires feedback and an explicit next action:

```python
# Revise only from the existing evidence catalog.
result = agent.resume_human_review(
    thread_id=result.thread_id,
    action="revise",
    feedback="State the applicability limitation before the conclusion.",
)

# Or execute a bounded, targeted Phase 3B research retry.
result = agent.resume_human_review(
    thread_id=result.thread_id,
    action="research_again",
    feedback="Find FDA guidance addressing electronic consent implementation.",
)

# End this run and let the UI collect a replacement question.
result = agent.resume_human_review(
    thread_id=result.thread_id,
    action="edit_question",
    feedback="The original question omitted the device-study context.",
)
```

The older `approved=True/False` API remains supported: `True` maps to `approve`
and `False` maps to `edit_question`. Human-requested revision and research use
the same retry budgets as evaluator-requested attempts. They do not rerun broad
question routing.

All commands resolve data paths from the repository location, so they do not depend on the shell's current working directory.
