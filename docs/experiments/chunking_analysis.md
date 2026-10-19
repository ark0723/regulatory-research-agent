# Section-aware chunking analysis

## Experiment configuration

- Baseline A: section-level Documents without splitting
- Experiment B: structure-aware splitting for long sections
- Target chunk size: 1500 characters, including contextual header
- Requested overlap: 200 characters; actual overlap is boundary-aligned
- Unlinked FDA footnote 1: excluded from both retrieval strategies and retained in source review data

## Document counts

- Baseline A: 173
- Experiment B: 330
- Section Documents split: 52
- Section Documents unchanged: 28
- Linked footnotes unchanged: 93

### Baseline by type

- guidance/footnote: 93
- guidance/section: 65
- regulation/section: 15

### Experiment B by type

- guidance/footnote: 93
- guidance/section: 177
- regulation/section: 60

## Chunk length statistics

### Baseline A

- min: 158
- median: 762
- p95: 4181
- max: 13458

### Experiment B

- min: 158
- median: 1099.0
- p95: 1460
- max: 1500

- Chunks exceeding target: 0
- Total intentional overlap characters: 42243
- Split boundaries: {'paragraph': 117, 'sentence': 38, 'hard_limit': 2, 'document_end': 52}
- Unavoidable hard-limit splits: ['fda-informed-consent-2023:section:III.A.1 chunk 0', 'fda-informed-consent-2023:section:V.14 chunk 0']

## Representative sections

- regulation `50.23`: 16 chunks; lengths [1048, 1048, 1413, 1418, 1416, 1357, 1334, 1427, 1336, 1425, 1121, 1194, 981, 1287, 1424, 1148]; boundaries [paragraph, paragraph, sentence, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, document_end]
- regulation `50.3`: 10 chunks; lengths [1364, 1267, 1367, 1316, 1335, 1355, 1379, 1318, 1407, 584]; boundaries [paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, document_end]
- regulation `50.24`: 11 chunks; lengths [1384, 1410, 958, 1354, 1136, 958, 1291, 1186, 953, 1236, 1222]; boundaries [paragraph, paragraph, paragraph, paragraph, paragraph, paragraph, sentence, paragraph, paragraph, paragraph, document_end]
- guidance `V.5`: 9 chunks; lengths [1316, 1464, 1417, 1460, 1263, 1317, 1470, 994, 1373]; boundaries [paragraph, paragraph, paragraph, sentence, paragraph, sentence, paragraph, paragraph, document_end]
- guidance `V.1`: 11 chunks; lengths [1295, 983, 1279, 1276, 896, 1027, 1216, 1125, 950, 1356, 1400]; boundaries [paragraph, paragraph, paragraph, sentence, paragraph, sentence, paragraph, paragraph, paragraph, sentence, document_end]
- guidance `III.A.2`: 7 chunks; lengths [1117, 1101, 1237, 1233, 1466, 1329, 947]; boundaries [sentence, paragraph, paragraph, paragraph, paragraph, sentence, document_end]

## Integrity and provenance

- CFR normalized records contain usable paragraph identifiers such as `(a)`, `(1)`, `(i)`, and combined identifiers such as `(d)(1)`.
- CFR chunk ends prefer normalized paragraph boundaries. Oversized paragraphs fall back to sentence boundaries and only then to a hard character limit.
- FDA Guidance chunks prefer paragraph boundaries, then sentence boundaries, and never cross a Section boundary.
- Linked footnotes remain individual Documents and are never split.
- All original metadata and citations are retained. Split chunks receive deterministic IDs and keep their original ID in `parent_record_id`.
- Full parent bodies were reconstructed from chunk offsets, with overlaps checked for character equality. No source body text was lost, rewritten, or introduced in this experiment.

## Known limitations

- Chunk page ranges retain the parent Section page range because the normalized source does not provide exact page offsets for each paragraph.
- Boundary-aligned overlap may be smaller or larger than 200 characters.
- CFR paragraph ancestry is inferred only from identifiers present in the normalized XML. Missing identifiers are not invented.
- Contextual title, Section path, and CFR paragraph-context headers repeat across chunks; these headers are excluded from body reconstruction checks.
- Chunk size is character-based rather than tokenizer-specific.
- Two Guidance paragraphs (`III.A.1` chunk 0 and `V.14` chunk 0) contain no usable paragraph or sentence boundary near the target; those chunks use the documented hard-limit fallback.

## Recommendation

Keep Baseline A and Experiment B as separate, versioned retrieval inputs. Compare them later with the same queries, embedding model, filters, and retrieval metrics. This preparation analysis demonstrates structural and text integrity, not retrieval-quality improvement.
