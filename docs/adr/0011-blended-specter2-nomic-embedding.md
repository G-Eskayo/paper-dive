# 0011 — Blend SPECTER2 with nomic-embed for relevance scoring, not SPECTER2 alone

## Status

Accepted (2026-07-07)

## Context

SPECTER2 was chosen ([[0007]]'s priority-queue heuristic needs an embedding source) because it's
trained via citation-based contrastive learning — its entire training signal is "papers that cite
each other should embed close together," a strong match for a citation-graph traversal.

That same property is also its documented weakness: a citation-community audit ("Topic Is Not
Agenda") found citation-trained embeddings partly encode *which research community/citation-clique
a paper belongs to*, not purely *what it's about*. Concretely: a paper topically identical to the
seed but from a different, non-citing research tradition (independently-derived competing work,
or a rival camp) could embed as less similar than its actual topical relevance warrants — because
SPECTER2 learned citation-adjacency, not semantic-adjacency.

This collides directly with [[0008]]'s concern about not silently dropping rebuttals/competing
ideas — SPECTER2 alone could systematically under-rank exactly the cross-camp competing papers
the traversal (and the downstream "competing-ideas surface" synthesis tool) most needs to
surface. The `result`-intent bypass rule in [[0008]] already partially mitigates this (a
cross-camp rebuttal often still carries a `result`-intent tag and gets rescued regardless of
rank), but that's one layer of defense, not a reason to skip a second, cheaper one.

## Decision

Relevance scores blend two embedding sources: SPECTER2 (citation-graph-aware relevance) and
nomic-embed via Ollama (pure topical similarity, already running for `qa-knowledge`/
`research-feed`, independent of citation structure). A weighted combination, not SPECTER2 alone.

## Consequences

- Hedges against SPECTER2's documented citation-community bias without giving up its
  domain-specific strength — nomic-embed catches topically-relevant papers SPECTER2's
  citation-shaped space might under-rank.
- Two embedding calls per candidate paper instead of one — roughly doubles embedding compute
  cost per node, though this is small relative to the Semantic Scholar API round-trip per node.
- The blend weighting (how much each source contributes) isn't tuned yet — starting point,
  needs real traversal runs to sanity-check rather than a value picked in the abstract.
- Combines with [[0008]]'s result-intent bypass as a second, independent layer of defense against
  silently dropping competing ideas — neither mechanism alone was judged sufficient.
