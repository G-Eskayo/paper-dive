# 0009 — Store first-discovery edge metadata only, not all paths

## Status

Accepted (2026-07-07)

## Context

The downstream synthesis tools already scoped in §K (argument mapper, continuity checker) need
to know not just that a paper is *in* the knowledge base, but *how it connects* — which parent
paper led to it and via which edge type (reference or citation). Without this, those tools would
have nothing to build a claim-chain from and would need to re-derive relationships from scratch.

A paper can be reached multiple ways: as a reference of node A in one traversal, and later as a
citation of node B in a different investigation entirely (enabled by cross-investigation reuse,
[[0008]]). Two options: keep only the edge metadata from whichever traversal discovered it first,
or accumulate every path ever found to it.

## Decision

Each paper record stores `discovered_via: {parent_doi, edge_type: reference|citation,
hop_depth}` from its *first* discovery only. Later re-encounters via a different edge are not
recorded.

## Consequences

- Simple, one fixed record per paper — no growing list of incoming edges to maintain or query.
- A paper's stored relationship reflects whichever investigation found it first, not necessarily
  the most contextually relevant connection for a later investigation reusing it. E.g. a paper
  first discovered as a passing reference from an unrelated seed might later be genuinely
  central to a new seed's argument, but the metadata will only show the original, unrelated link.
- If the synthesis tools, once built, turn out to need multi-path history, this is the first place
  to revisit — deliberately deferred rather than built speculatively now.
