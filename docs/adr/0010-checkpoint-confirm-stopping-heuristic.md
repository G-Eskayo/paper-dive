# 0010 — Checkpoint-and-confirm with a diminishing-returns signal, not a flat hard node cap

## Status

Accepted (2026-07-07)

## Context

Even with depth limit, shared relevance floor, and per-direction top-K caps ([[0007]], [[0008]]),
there's no hard mathematical ceiling on total nodes/API calls for a broad enough seed topic — a
real cost/time/rate-limit risk. Initial proposal was a flat total-node cap (e.g. `--max-nodes
150`) as a fourth independent stopping condition.

Gil pushed back with a real concern: a flat cap can't distinguish "this topic is genuinely
exhausted" from "we just hit an arbitrary number" — risking silent, invisible quality loss on a
genuinely rich topic that still had high-relevance sources left to find.

## Decision

Two mechanisms instead of one flat cap:
1. **Diminishing-returns stopping heuristic** — track the relevance-score trend of recently
   processed nodes; treat the topic as naturally exhausted when recent nodes consistently
   approach the relevance floor, not before.
2. **Checkpoint-and-confirm at a cost ceiling** — if the diminishing-returns signal hasn't fired
   by some ceiling (e.g. 100 nodes / N API calls), pause and report what's still queued (count,
   relevance range) and ask whether to continue, extend, or stop. Fits `paper-dive`'s existing
   conversational interaction model (the L0-L5 ladder is already a live back-and-forth, not a
   silent background job).

A narrow topic self-terminates early via (1) and never reaches the checkpoint. A genuinely broad
topic reaches the checkpoint and gets a deliberate, informed decision rather than a silent cutoff.

## Consequences

- Running out of budget becomes a visible, deliberate choice (continue/extend/stop) instead of
  an invisible truncation — directly answers the "am I losing quality without knowing it" worry.
- More moving parts than a flat cap: needs the relevance-trend tracking logic, plus a
  conversational checkpoint/resume mechanism, neither of which a flat cap would have required.
- Assumes `paper-graph` mode runs in a live conversational context where pausing to ask is
  natural — would need rethinking if this mode were ever invoked headlessly/unattended (e.g.
  from a cron job), where there'd be no one to answer the checkpoint.
