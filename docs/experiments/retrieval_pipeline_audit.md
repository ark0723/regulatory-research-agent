# Retrieval pipeline audit

## Existing behavior

- Every unique parent in the reranked top-N was expanded to all sibling chunks.
- Duplicate parents were grouped and expanded once, and chunks were restored in
  `chunk_index` order.
- Known character overlap was removed only while rendering context.
- Expansion had no character or chunk budget.
- `section_complete` compared only the number of unique chunks with
  `chunk_count`; it did not verify exact IDs, indexes, or continuous offsets.
- CFR `paragraph_context` identifies ancestry at a chunk boundary, but it is not
  sufficient by itself for exact paragraph reconstruction. For example, the
  second § 50.25 chunk begins in (a)(6) and ends in (b)(1).

The installed Sentence Transformers `CrossEncoder.predict()` uses the model's
single-label sigmoid activation. The resulting scores are useful for ordering,
but are not calibrated probabilities.

## Changes

- Exact sibling IDs, chunk indexes, and provenance coverage now determine parent
  completeness.
- Expansion follows reranker order and supports character and expanded-chunk
  budgets.
- A later section that cannot fit is excluded with an explicit warning instead
  of consuming the remaining budget with a large low-priority expansion. The
  first/highest-priority section may fall back to a bounded chunk neighborhood
  and is then marked incomplete.
- CFR top-level paragraph extraction is attempted only after reconstructing a
  complete parent and verifying the paragraph markers in the source text.
- Stable IDs, citations, paragraph context, URLs, and other source metadata are
  retained. Rendering distinguishes complete parent sections, complete scoped
  paragraphs, and incomplete budgeted chunks.
- Structured diagnostics cover counts, selected/excluded parents, approximate
  context tokens, completeness, warnings, and retrieval/reranking/expansion
  latency.
- An absolute `min_rerank_score` is available but defaults to `None`; no threshold
  was selected from one query.

## Informed-consent regression

Query: `피험자 동의서의 기본 요소는 무엇인가?`

The observed reranker order supplied for this audit was §§ 50.25, 50.27, 50.20,
50.22, and 50.23, with scores 0.6094, 0.0082, 0.0066, 0.0046, and 0.0030. An
offline replay used that order with the repository's real chunk corpus.

| Measure | Previous expansion | Budgeted paragraph-aware expansion |
| --- | ---: | ---: |
| Source chunks expanded | 25 | 9 |
| Rendered evidence items | 25 | 4 |
| Context characters | 24,217 | 6,652 |
| Approximate tokens (chars / 4) | 6,055 | 1,663 |
| All eight § 50.25(a) elements present | Yes | Yes |
| § 50.25(b) additional elements present | Yes | No |
| § 50.23 expansion | All 16 chunks | Excluded by remaining chunk budget |

The live pgvector replay could not be repeated during this change because the
configured Ollama server was not running. The paragraph-completeness result is
also covered by a deterministic regression test using the real indexed corpus.

## Remaining limitations

- Paragraph-aware scoping currently handles verifiable top-level CFR markers.
  Nested-only questions and FDA Guidance sections fall back to chunk or full
  parent evidence.
- A single best chunk determines the paragraph scope for each selected parent.
  Questions requiring several top-level paragraphs in the same section may need
  multi-scope selection in a later iteration.
- Approximate tokens use a transparent `characters / 4` estimate rather than a
  provider-specific tokenizer.
- No default rerank threshold is enabled. A representative evaluation set is
  still needed before choosing one.
