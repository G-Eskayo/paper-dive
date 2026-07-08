# 0012 — Semantic Scholar rate-limit handling: exponential backoff, no API key (yet)

## Status

Accepted (2026-07-07)

## Context

The real end-to-end smoke test against an unpublished paper (`fetch_neighbors_by_search`, the
`/paper/search` endpoint) hit a 429 that a first retry attempt (1s/2s/4s backoff) didn't survive.
Checked Semantic Scholar's own API release notes (`allenai/s2-folks/API_RELEASE_NOTES.md`) rather
than guess further:

- **Unauthenticated** (what we currently use): 5,000 requests per 5 minutes, shared globally
  across every unauthenticated caller of the API — not a per-user quota. Explains the volatility:
  contention from unrelated users elsewhere can throttle us even at modest usage on our end.
- **Authenticated (API key)**: endpoint-specific. `/paper/search`, `/paper/batch`,
  `/recommendations` are capped at 1 request/second even *with* a key — this is exactly the
  endpoint `fetch_neighbors_by_search` (the unpublished-seed path) depends on. Other endpoints get
  10/second.
- **Since March 2024, exponential backoff is a required policy**, not optional — validates
  `_get_with_retry`'s approach directly.
- **Since August 2024, Semantic Scholar rejects new API key applications from free email domains
  and third-party applications.** A Gmail-based application could plausibly be rejected. "Just get
  a key" is not a guaranteed reliability fix.

## Decision

Keep using unauthenticated access with exponential backoff (`_get_with_retry`), not an API key,
for now. Do not treat applying for a key as the obvious next step — the search endpoint's 1 RPS
cap would apply even with one, and approval itself is uncertain given the free-email-domain
policy.

## Consequences

- The unpublished-seed path (`fetch_neighbors_by_search`) is the more rate-limit-fragile of the
  two fetch paths, since it hits `/paper/search` specifically — worth being conservative about
  how often it's invoked (it's only called once per traversal, for the seed's own first hop, not
  per-node, which keeps this manageable as currently scoped).
- If reliability becomes a real recurring problem in practice, applying for an API key is still
  worth attempting (existing keys aren't affected by the rejection policy, only new
  applications) — but go in with the expectation it might be rejected, not as a sure fix.
- `_get_with_retry`'s current backoff (1s/2s/4s/8s across 4 attempts) may still be too short for
  a sustained rate-limit window during heavy global usage — if 429s recur often in practice,
  lengthen the backoff schedule rather than add more retries at the same short intervals.
