# Codex Project Instructions

## Project Overview

This project is a multi-agent regulatory RAG system focused initially on FDA regulations and guidance documents.

The long-term architecture is documented in `docs/architecture/multi-agent-design.md`.

Read that document before proposing or implementing architectural changes.

## Development Responsibilities

- The project owner makes high-level architecture and design decisions.
- Codex assists with implementation, testing, debugging, and code review.
- Implement only the explicitly requested scope.
- Do not proactively implement future architectural components.
- Do not perform broad refactoring without approval.
- When requirements are ambiguous, explain the tradeoffs and ask for a decision.

## Architecture Principles

- Use LangGraph for agent orchestration.
- Separate regulation retrieval from guidance retrieval.
- Reuse the existing PostgreSQL/PGVector retrieval pipeline.
- Preserve source provenance and citation traceability.
- Prefer maintained libraries and avoid deprecated APIs.
- Use structured outputs and testable component boundaries.
- Keep retry loops bounded.
- Prioritize regulatory accuracy over answer fluency.

## Implementation Workflow

Before implementation:
1. Inspect relevant existing code.
2. Explain the proposed changes and their scope.
3. Identify potential compatibility issues.

During implementation:
1. Preserve existing behavior unless changes are requested.
2. Keep modifications focused.
3. Add or update relevant tests.

After implementation:
1. Run relevant tests.
2. Summarize modified files.
3. Report remaining limitations and risks.

## Important

The architecture document describes the intended future system, not a request to implement everything immediately.

Do not begin work on unimplemented architectural components unless explicitly instructed.