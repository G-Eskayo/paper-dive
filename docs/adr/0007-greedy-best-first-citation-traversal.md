# 0007 — Greedy best-first search, not depth-penalized A*, for citation-graph traversal

## Status

Accepted (2026-07-07)

## Context

§K of `marvin-roadmap.md` scopes recursive bibliography traversal for the `paper-dive` citation-graph
mode, currently just a vague BFS/DFS sketch with an ad-hoc "depth limit OR relevance floor,
whichever triggers first" tie-break. The actual design need is turning this into informed search:
a priority queue ordered by relevance-to-seed (embedding similarity) rather than naive
breadth-first or depth-first expansion order.

Two shapes considered: true A* (blend accumulated path-cost, e.g. depth, into the priority score
itself, so a highly-relevant paper many hops deep ranks behind a moderately-relevant paper one hop
deep) vs. plain greedy best-first search (priority = pure relevance-to-seed; depth is a separate
hard cap, not part of the ranking).

## Decision

Plain greedy best-first search. The priority queue orders candidate papers purely by embedding
similarity to the seed paper. Depth remains a separate hard stopping condition, not blended into
the priority score.

## Consequences

- A highly relevant paper several hops from the seed is not penalized for distance — matches the
  actual goal (surface the most relevant primary sources near a topic), where "far but relevant"
  is exactly the kind of foundational source this feature exists to find.
- Simpler than true A*: one scoring dimension (relevance) instead of two blended together.
- Depth limit and relevance floor remain two independent stopping conditions, as originally
  sketched in the roadmap — this decision fixes what the priority *ordering* optimizes for, not
  the stopping conditions themselves (see [[0008]] for how those gates work).
