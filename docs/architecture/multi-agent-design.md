# Multi-Agent Regulatory RAG — Architecture Vision

## 1. Project Goal

Build a multi-agent regulatory research and question-answering system focused initially on FDA regulations and guidance documents.

The system should produce evidence-grounded answers with traceable citations, distinguish binding regulatory requirements from guidance recommendations, and support iterative review and human intervention when necessary.

## 2. Target Architecture

User Question → Supervisor / Router → Specialist RAG Agents → Answer Synthesis → Reviewer → Final Answer / Retry / Human Review

This is the intended architecture, not a description of components already implemented.

## 3. Specialist Agents

### Regulation Agent

- Search relevant FDA regulations across CFR Parts.
- Identify requirements, exceptions, conditions, and applicability.
- Generate answers supported by precise regulatory citations.
- Use a dedicated regulation retrieval tool.

### Guidance Agent

- Search relevant FDA guidance documents.
- Explain recommendations, interpretations, and implementation considerations.
- Preserve document status, scope, and source provenance where available.
- Use a dedicated guidance retrieval tool.

Both agents should reuse the shared retrieval infrastructure, including PostgreSQL/PGVector, Cross-Encoder reranking, and context-aware evidence expansion.

The initial design uses one PGVector collection with metadata-based document filtering.

## 4. Supervisor and Routing

The Supervisor determines whether a question requires:

- Regulation Agent only
- Guidance Agent only
- Both agents

When both are needed, they should execute concurrently where practical.

The Supervisor also coordinates retries and human intervention.

## 5. Answer Synthesis

Combine specialist findings into a coherent response while:

- Preserving evidence and citations.
- Separating legal obligations from nonbinding recommendations.
- Avoiding unsupported conclusions.
- Identifying unresolved ambiguities or conflicting evidence.

## 6. Reviewer Agent

Evaluate generated answers for:

- Factual grounding in retrieved evidence.
- Citation validity and source attribution.
- Completeness relative to the question.
- Unsupported assumptions.
- Potentially misleading regulatory interpretations.

The reviewer should produce structured feedback.

Possible decisions:

- PASS
- RETRY
- HUMAN_REVIEW

Citation existence checks and semantic claim-support verification are separate responsibilities.

## 7. Retry Strategy

When review fails, the system may:

- Reformulate a query.
- Search specific CFR sections.
- Expand retrieval scope.
- Retrieve additional guidance.
- Revise an answer using existing evidence.

Retries must be bounded and should address specific identified deficiencies.

## 8. Human-in-the-Loop

Human intervention may be needed for:

- Missing user context.
- Ambiguous regulatory applicability.
- Unresolved evidence conflicts.
- Repeated unsuccessful retrieval or review.

LangGraph interrupt and checkpointing are potential implementation mechanisms.

Clarification from the user and expert regulatory review should be treated as distinct workflows.

## 9. Engineering Principles

- Modular, testable components.
- Clear separation of agent responsibilities.
- Shared retrieval infrastructure.
- Evidence-grounded generation.
- Reliable source provenance.
- Explicit state transitions.
- Measured latency and answer quality.
- Incremental development without premature complexity.

## 10. Implementation Status

This document describes the target architecture.

Actual implementation status must be verified against the repository. Do not assume that a component is implemented merely because it is described here.

The first bounded specialist slice is now implemented in
`regulatory_agent.specialist_agent.CorrectiveRegulatorySpecialist`:

```text
retrieve + rerank + expand
  → extract and evaluate evidence
  → generate answer
      or rewrite query → retrieve (bounded retry)
```

It can be configured for either `regulation` or `guidance`. The graph source can be generated with
`agent.mermaid()` or saved with `agent.save_mermaid(...)`; a readable reference
is stored in `docs/architecture/corrective-specialist-rag.mmd`.

A bounded question-routing and dispatch slice is now also implemented in
`regulatory_agent.routing`:

```text
structured multi-label route
  → regulation and/or guidance and/or web
  → selected tool-agents execute concurrently
  → independently attributed results
```

The top-level `regulatory_agent.multi_agent.RegulatoryMultiAgent` graph is now
also implemented:

```text
route_question
  → regulation_agent and/or guidance_agent and/or web_agent
  → merge structured contributions
  → generate_answer
  → deterministic synthesis-citation validation
  → evaluate_answer
  → route_after_evaluation
      → pass / controlled failure / human_review
      → revise_answer
          → validate_revision
          → evaluate_answer
      → prepare_research_retry
          → validate_retry_plan
          → execute_research_query (validated target/query pairs only)
          → merge_research_evidence
          → regenerate_answer
          → deterministic synthesis-citation validation
          → evaluate_answer
```

The three route targets are explicit graph nodes. Regulation and guidance nodes
consume structured `CorrectiveRegulatorySpecialist` results; the web node
consumes reranked `TavilyWebResearchAgent` evidence and remains supplemental.
Final source IDs (`S1`, `S2`, ...) are assigned deterministically after merge,
so generated citations can be checked and rendered from source metadata.

The `evaluate_answer` node is implemented with
`ToolCallingAnswerEvaluator`. It receives direct `@tool` search tools rather
than calling the specialist agents again. The reviewer can verify CFR,
guidance, or current web claims and returns six structured scores, a
deterministically calculated total, concrete deficiencies, tool usage, and one
of `pass`, `revise`, `research_again`, or `human_review`.

Evaluation routing is implemented as a deterministic policy node with no
additional routing LLM call. `pass` ends successfully only after citation
validation; `human_review` enters the existing interrupt flow. `revise` starts
a bounded revision that uses only the current answer, evaluator feedback, and
the existing `SynthesisSource` catalog. It cannot call retrieval or search
tools. The structured `FinalAnswerDraft` is checked by the same deterministic
`validate_and_render_synthesis` function used for the initial answer before it
can replace the published answer. A valid revision returns to evaluation.

`research_again` produces a bounded `ResearchRetryPlan` from evaluation
feedback, missing information, unsupported claims, compact source coverage, and
previous query history. The deterministic validator checks meaningful gaps,
target availability, query/reason content, duplicates, query limits, and false
claims that research already occurred. Only the validated target/query pairs
are dispatched; no broad routing pass is repeated. Multiple queries may target
one specialist and multi-target plans fan out through the same execution node.
Planning or validation failures terminate as `research_retry_planning_failed`
without consuming an attempt.

A validated plan is one research attempt. Its total and research counters are
incremented once immediately before dispatch. Successful contributions are
merged with existing evidence using persistent record/evidence IDs or web URLs;
display titles and citation strings alone are not treated as stable identity.
The merged catalog is bounded and receives fresh contiguous `S1`, `S2`, ...
IDs before answer regeneration. The regenerated draft passes the same citation
validator and evaluator as the initial answer. Partial or complete specialist
failure is reported explicitly and does not trigger unplanned fallback search.

Missing or invalid evaluations terminate as `evaluation_failed`, never as
`answered`. Revision model/schema failures terminate as `revision_failed`;
citation failures terminate as `synthesis_validation_failed` while retaining
the last validated answer rather than publishing the invalid draft.

`EvaluationRetryPolicy` defaults to three total automatic retries, two answer
revisions, one research retry, three queries per research plan, and twenty
merged synthesis sources. State retains the total and per-kind counters plus
compact evaluation and research-attempt diagnostics. Revision counters begin at
revision execution; research counters begin only after plan validation. Blocked
or invalid plans do not consume an attempt. Aggregate policy checks consider the
sum of per-kind counters so alternating retry types cannot bypass the total
limit.

The parent graph now supports checkpointed human approval through LangGraph
`interrupt()` and `Command(resume=...)`. By default, it pauses only for an
evaluation decision of `human_review`; callers can instead choose `always` or
`never`. The interrupt exposes a typed request containing the answer,
citations, evaluation, reason, and allowed actions. Approval and rejection are
persisted in a compact human-review history, and every rejection requires
written feedback. `revise` performs a citation-validated revision using the
existing source catalog; `research_again` sends the feedback through the
existing bounded Phase 3B plan and execution path; `edit_question` ends the
current run so the UI can collect a new question. A rejection never jumps
directly back to broad question routing.

## 11. Development Ownership

The project owner retains responsibility for architectural decisions, component boundaries, and implementation priorities.

Codex supports the implementation of individually assigned tasks.

**Do not implement the entire target architecture without explicit authorization.**
