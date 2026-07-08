# 0008 — Bidirectional traversal (references + citations), asymmetric caps, shared relevance floor with a result-intent bypass

## Status

Accepted (2026-07-07)

## Context

The roadmap's original wording ("follow its bibliography") implied references-only (backward:
papers the seed cites) — this terminates cleanly toward foundational sources and was the initial
recommendation. Counter-argument, and the one that won: §K's own already-scoped downstream
synthesis tools require both directions to function at all — the *continuity checker* needs later
papers that may contradict the seed (forward/citations), and the *argument mapper* needs
foundational lineage (backward/references). References-only would have silently broken tools
already committed to in the roadmap.

Following both directions reopens a real scale asymmetry: a paper's reference list is naturally
bounded (tens, ~its own bibliography size); citations are not (a landmark paper can have
thousands of citing papers) — unbounded citation-following risks one popular node exploding the
frontier and the API budget in a single hop.

Separately: Gil raised a real edge case — a rebuttal paper could rank below whatever top-K cutoff
is chosen and get silently dropped, even though it's exactly what the "competing-ideas surface"
tool needs to find. Checked Semantic Scholar's API directly rather than assume: citations expose
an `intents` field, but only three coarse categories (background/method/result), not a dedicated
"contradicts" or "compare_contrast" label, and only when S2 has full-text access.

## Decision

- Traverse both directions (references and citations).
- Each direction gets its own top-K cap per hop (e.g. references_top_k=10, citations_top_k=5) —
  same relevance-ranking mechanism, two different K values reflecting the asymmetric role:
  references are the primary-source backbone (worth a more generous pull), citations are
  situational context (lighter pull).
- The relevance floor (minimum similarity to bother expanding) is shared across both directions —
  no structural reason for "relevant" to mean something different depending on which edge led
  there.
- Citations tagged with `result`-intent (substantive engagement with the seed's actual findings,
  per Semantic Scholar's classification) bypass the top-K cutoff entirely and are always pulled in
  regardless of rank.

## Consequences

- Two tunable numbers (per-direction top-K) instead of one, but same mechanism — not two
  separately-tuned systems.
- The result-intent bypass is an imperfect proxy for "don't drop rebuttals" — `result`-intent
  citations include papers that confirm the seed's findings just as often as ones that dispute
  them, since Semantic Scholar has no dedicated contradiction/rebuttal label. Real gap, not fully
  solvable with currently-available data — closing it properly needs the fallacy/logic-validation
  capability logged as a longer-term vision in §K, not this design pass.
- `intents` data is only available when Semantic Scholar has full-text access to the citing paper
  — the bypass rule has silent blind spots for papers where that data doesn't exist.
