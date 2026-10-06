#!/usr/bin/env python3
"""
Semantic Scholar API client with rate-limit handling.

Provides retry logic for transient 429 (rate-limit) responses and ID namespace detection.
Reused by paper-dive and research skills. See docs/adr/0012 for rate-limit strategy.
"""
import os
import random
import re
import time
from typing import Any
from urllib.parse import urlparse

import requests

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

S2_PAPER_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_HOST = "api.semanticscholar.org"


def s2_id(identifier: str) -> str:
    """Semantic Scholar's paper endpoint accepts several ID namespaces (DOI:,
    ARXIV:, CorpusId:...) under the same /paper/{id} path. Most of this
    project's own related-work citations are arXiv preprints (e.g.
    "2503.03704") with no formal DOI at all — recent AI-safety papers
    especially — so assuming DOI: unconditionally silently fails or (worse,
    see _shape_and_score) silently drops exactly the population this paper
    cites most. Detects the bare arXiv YYMM.NNNNN[vN] shape and prefixes
    accordingly instead of assuming DOI; passes through an already-prefixed
    identifier unchanged."""
    if identifier.startswith(("DOI:", "ARXIV:", "CorpusId:")):
        return identifier
    if _ARXIV_ID_RE.match(identifier):
        return f"ARXIV:{identifier}"
    return f"DOI:{identifier}"


def get_with_retry(
    url: str,
    params: dict,
    timeout: int,
    max_retries: int = 8,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """HTTP GET with exponential backoff on 429 rate-limit responses.

    Unauthenticated S2 access is a 1000 req/s pool shared across every anonymous caller on the
    internet, not a per-user quota — a 429 here is transient global contention, not us exceeding
    anything. An API key's introductory tier (1 RPS) isn't meaningfully better than this pool for
    our actual volume (~1-2 calls per seed paper), so riding out contention with longer backoff is
    the right fix, not chasing a key. S2_API_KEY is still honored if set, for whenever a real one
    with a higher approved tier exists.

    Args:
        url: Request URL
        params: Query parameters
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts (default 8)
        headers: Optional HTTP headers (S2_API_KEY will be added if set in env)

    Returns:
        Response object on success

    Raises:
        requests.exceptions.HTTPError: On non-429 errors, or 429 after max_retries exhausted
    """
    if headers is None:
        headers = {}
    else:
        headers = dict(headers)  # Copy to avoid mutating caller's dict

    # Only ever to Semantic Scholar's own host: this function is shared with the
    # research skill's arXiv search, and attaching the key to every URL would
    # send it to export.arxiv.org. Exact hostname match, not a substring, so a
    # lookalike such as api.semanticscholar.org.evil.example is not trusted.
    api_key = os.environ.get("S2_API_KEY")
    if api_key and urlparse(url).hostname == S2_HOST:
        headers["x-api-key"] = api_key

    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        if attempt < max_retries - 1:
            backoff = min(2 ** attempt, 60)  # 1s, 2s, 4s, ... capped at 60s
            time.sleep(backoff + random.uniform(0, 1))  # jitter — avoid lockstep retries against a shared pool
    resp.raise_for_status()  # exhausted retries — surface the final 429 as an error
