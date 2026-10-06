# 0031 — arXiv-only full-text fetch for high-relevance nodes in paper-graph

## Status

Accepted (2026-08-31)

## Context

[[0007]]–[[0011]] establish paper-graph's recursive traversal, scoring candidates by
relevance to a seed paper via blended SPECTER2 + nomic embedding. Candidates are stored in the
`paper-knowledge` collection with their abstract as the document text.

The traversal identifies papers worth including (score ≥ `relevance_floor`, default 0.65) but
doesn't distinguish between *highly relevant* nodes (which might merit deeper analysis) and
*barely included* nodes at the floor. When using these papers downstream (e.g., in `/compare`
or synthesis workflows), abstract-only storage limits analysis — full text unlocks finer-grained
claims, methodology, and counterarguments.

Fetching full text for *every* node is expensive (API calls, storage, embedding complexity).
Fetching it only for *high-relevance* nodes (a stricter threshold than inclusion) trades cost for
benefit: the papers most likely to matter get full-text treatment.

Sci-Hub was initially considered as a fallback source for DOI-only papers, but commit c3982a1
(2026-08-29) deliberately removed Sci-Hub from the project's source list across all skills,
with this reasoning: *"Sci-Hub is a shadow library that bypasses publisher paywalls to serve
copyrighted papers without authorization, not an open-access or society-partnered source."*
Implementing an automated Sci-Hub scraper as a permanent pipeline feature would contradict that
decision and complicate the system without a clear policy change to support it.

## Decision

**Full-text fetch happens for high-relevance nodes with arXiv IDs only.** Specifically:

1. **Two-tier relevance thresholds:**
   - `relevance_floor` (default 0.65): minimum score for *inclusion* in the graph
   - `full_text_threshold` (default 0.85): minimum score for *full-text fetch*
   
   Below-threshold nodes never trigger a fetch, even if they have arXiv IDs.

2. **arXiv as the sole full-text source:**
   - Nodes with arXiv IDs scoring ≥ `full_text_threshold` have their PDF fetched from
     `https://arxiv.org/pdf/{arxiv_id}` and full text extracted via the same pipeline as
     `ingest_paper.py` (importing `ingest_paper.extract_pdf()`).
   - Nodes scoring ≥ `full_text_threshold` without arXiv IDs (DOI-only) store their abstract — no
     fallback source, no Sci-Hub scraper.
   - Fetch failures (404, network error, PDF parse failure) gracefully degrade to abstract-only
     storage; they do not fail the traversal.

3. **Metadata stamping:**
   - `record_paper()` stores `full_text: bool` in ChromaDB metadata, making downstream code aware
     whether a document is full-text or abstract-only.

## Consequences

- **Pro:** High-relevance papers now have full text available for downstream analysis. The
  threshold split (inclusion vs. full-text) is explicit, testable, and tunable without rebuilding
  the whole traversal.
- **Pro:** arXiv-only scope is defensible and consistent with the project's existing stance on
  open-access sources (see c3982a1).
- **Pro:** Graceful degradation on fetch failure means one bad PDF or network hiccup doesn't
  break the traversal.
- **Con:** DOI-only high-relevance papers (journal articles without preprints) stay abstract-only.
  If a significant proportion of relevant papers are DOI-only, full-text coverage will be
  incomplete. This is a legitimate "no open-access source available" outcome, not a gap to
  solve with unauthorized mirror sites.
- **Con:** Chroma's default embedding function has a max input length; storing full text as
  documents may get truncated on embedding even though the full text itself is preserved.
  This is noted as a follow-up ticket on chunking if search quality suffers.
- **Acceptance criterion:** AC1 (full-text fetched and run through extraction for high-relevance
  arXiv nodes), AC2 (same extraction pipeline as ingest_paper), and AC3 (no fetch for
  below-threshold nodes, no Sci-Hub fallback) are all met and tested.

## Related ADRs

- [[0007]] — greedy best-first citation traversal (the traversal heuristic itself)
- [[0011]] — blended SPECTER2 + nomic embedding (relevance scoring)
