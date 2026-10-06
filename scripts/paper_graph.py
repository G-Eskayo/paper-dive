#!/usr/bin/env python3
"""
paper_graph.py — recursive citation-graph traversal for paper-dive.

Design record: ~/.agents/docs/adr/0007-0011.

Dependency note: SPECTER2 requires the `adapters` library, which pins
transformers~=4.57.6 — this is BELOW mlx-lm's declared transformers>=5.0.0
requirement. Verified empirically (real model load + generate) that mlx-lm
works fine under 4.57.6 despite that declared constraint; the pin is kept at
4.57.6 deliberately, not a mistake if you see it flagged by pip.
"""
from __future__ import annotations
import heapq
import sys
import tempfile
from pathlib import Path

# Helpers live next to this file in the standalone repo (G-Eskayo/paper-dive) and in ~/.agents/lib inside MARVIN.
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path.home() / ".agents" / "lib")]
from s2_client import get_with_retry as _get_with_retry
from s2_client import s2_id as _s2_id
from s2_client import S2_PAPER_BASE


def select_candidates(candidates: list[dict], top_k: int, relevance_floor: float) -> list[dict]:
    bypassed = [c for c in candidates if c.get("intent") == "result"]
    above_floor = [c for c in candidates if c["score"] >= relevance_floor and c.get("intent") != "result"]
    ranked = sorted(above_floor, key=lambda c: c["score"], reverse=True)
    return ranked[:top_k] + bypassed


def is_known(doi: str, collection) -> bool:
    result = collection.get(ids=[doi])
    return len(result["ids"]) > 0


def get_discovery(collection, doi: str) -> dict | None:
    """Returns the first-discovery record ({parent_doi, edge_type, hop_depth}) for doi, or None if doi is unknown or is a seed (no parent)."""
    result = collection.get(ids=[doi], include=["metadatas"])
    if not result["ids"]:
        return None
    meta = result["metadatas"][0]
    if "parent_doi" not in meta:
        return None
    return {"parent_doi": meta["parent_doi"], "edge_type": meta["edge_type"], "hop_depth": meta["hop_depth"]}


def record_paper(collection, doi: str, abstract: str, discovered_via: dict | None, full_text: bool = False) -> None:
    import time

    # created_epoch (not created_at) matches dump_collection.py's --since
    # filter, which needs a numeric field for ChromaDB's $gt comparator
    # (ISO-8601 strings aren't supported) -- same field name qa-knowledge
    # uses, so cross_machine_merge.py's incremental sync pattern works here
    # unmodified.
    metadata = {"doi": doi, "created_epoch": time.time(), "full_text": full_text}
    if discovered_via is not None:
        metadata["parent_doi"] = discovered_via["parent_doi"]
        metadata["edge_type"] = discovered_via["edge_type"]
        metadata["hop_depth"] = discovered_via["hop_depth"]
    collection.add(ids=[doi], documents=[abstract], metadatas=[metadata])


def fetch_arxiv_pdf_text(arxiv_id: str) -> str | None:
    """Fetch full PDF text from arXiv for a given arXiv ID.

    Returns the extracted text on success, or None on 404/network failure.
    Uses ingest_paper.extract_pdf() for parsing, so behavior is consistent
    with paper-dive's main ingestion path.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        resp = _get_with_retry(url, params={}, timeout=30, max_retries=3)
    except Exception:
        # network error, 404, etc. — gracefully degrade to abstract-only
        return None

    if resp.status_code == 404:
        return None

    try:
        resp.raise_for_status()
    except Exception:
        return None

    # Write PDF bytes to temp file, extract via ingest_paper
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        # Import ingest_paper to use its extract_pdf function
        import sys
        scripts_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts_dir))
        import ingest_paper

        text = ingest_paper.extract_pdf(Path(tmp_path))
        return text
    except Exception:
        # PDF parse failed (scanned, corrupted, etc.) — gracefully degrade
        return None
    finally:
        # clean up temp file
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


def fetch_full_text(node: dict) -> str | None:
    """Fetch full text for a high-relevance node if an arXiv ID is available.

    Currently arXiv-only; returns None if no arXiv ID present or fetch fails.
    A future legitimate open-access DOI resolver could be added here.
    """
    arxiv_id = node.get("arxiv_id")
    if not arxiv_id:
        return None
    return fetch_arxiv_pdf_text(arxiv_id)


_NEAR_FLOOR_MARGIN = 0.05


def diminishing_returns(recent_scores: list[float], relevance_floor: float, window: int) -> bool:
    if len(recent_scores) < window:
        return False
    last_n = recent_scores[-window:]
    return all(score <= relevance_floor + _NEAR_FLOOR_MARGIN for score in last_n)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def blended_score(seed_embeddings: dict, candidate_embeddings: dict, specter2_weight: float = 0.5) -> float:
    """Blend SPECTER2 and nomic embeddings to hedge against citation-clique bias. See ADR 0011."""
    specter2_sim = _cosine_similarity(seed_embeddings["specter2"], candidate_embeddings["specter2"])
    nomic_sim = _cosine_similarity(seed_embeddings["nomic"], candidate_embeddings["nomic"])
    return specter2_weight * specter2_sim + (1 - specter2_weight) * nomic_sim


_specter2_model = None
_specter2_tokenizer = None


def _check_huggingface_reachable() -> None:
    """Fails fast with an actionable message instead of a ~60s timeout
    followed by a raw traceback -- found live 2026-07-13 when
    huggingface.co was TLS-reset on Gil's work network (SNI-level
    filtering: DNS + TCP succeed, handshake gets reset) while
    api.semanticscholar.org worked fine from the same machine. Reuses
    recorded history first so a second attempt on a known-bad network
    doesn't even need to re-check live."""
    import sys

    sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path.home() / ".agents" / "lib")]
    from network_reachability import known_status, check_and_record, current_network_id

    status = known_status("huggingface.co")
    if status is None:
        status = check_and_record("huggingface.co")
    if not status:
        raise RuntimeError(
            "huggingface.co is unreachable on this network "
            f"({current_network_id()}) -- SPECTER2 embeddings need it to "
            "download model weights. This looks network-specific, not "
            "machine-specific (see network_reachability.py's docstring for "
            "the 2026-07-13 incident this guards against) -- try a "
            "different network, e.g. run this on gils-mac-mini."
        )


def _load_specter2():
    global _specter2_model, _specter2_tokenizer
    if _specter2_model is None:
        _check_huggingface_reachable()
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        _specter2_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        _specter2_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        _specter2_model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
        # NOTE: load_adapter() logs "There are adapters available but none are activated for the
        # forward pass" during its own internal execution. This is a false positive in the
        # `adapters` library (verified: a direct before/after embedding comparison shows the
        # adapter IS active and correctly affecting output on every subsequent forward pass) —
        # see adapter-hub/adapters#815. Safe to ignore; not evidence anything here is broken.
    return _specter2_model, _specter2_tokenizer


def _real_specter2_embed(text: str) -> list[float]:
    import torch

    model, tokenizer = _load_specter2()
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().tolist()  # CLS-token pooling, SPECTER2 convention


def _real_nomic_embed(text: str) -> list[float]:
    import requests

    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_paper(text: str, specter2_fn=None, nomic_fn=None) -> dict:
    specter2_fn = specter2_fn or _real_specter2_embed
    nomic_fn = nomic_fn or _real_nomic_embed
    return {"specter2": specter2_fn(text), "nomic": nomic_fn(text)}


def _shape_and_score(papers: list[dict], seed_embeddings: dict, embed_fn, is_citation: bool, is_known_fn=None) -> list[dict]:
    candidates = []
    for p in papers:
        external_ids = p.get("externalIds") or {}
        # DOI preferred (more universal/stable), but most AI-safety papers this
        # project actually cites are arXiv-only preprints with no formal DOI —
        # falling back to ArXiv here (rather than skipping) is what makes
        # traversal actually cover that population instead of silently
        # dropping it. See _s2_id() for how this gets re-prefixed on lookup.
        doi = external_ids.get("DOI") or external_ids.get("ArXiv")
        arxiv_id = external_ids.get("ArXiv")  # preserve ArXiv ID separately for full-text fetch
        abstract = p.get("abstract") or p.get("title") or ""
        if not doi or not abstract:
            continue
        if is_known_fn is not None and is_known_fn(doi):
            continue
        candidate_embeddings = embed_fn(abstract)
        score = blended_score(seed_embeddings, candidate_embeddings)
        intent = "result" if is_citation and "result" in (p.get("intents") or []) else None
        candidates.append({"doi": doi, "arxiv_id": arxiv_id, "score": score, "intent": intent, "abstract": abstract})
    return candidates


def fetch_neighbors_from_s2(doi: str, seed_embeddings: dict, embed_fn=None, is_known_fn=None) -> dict:
    import time
    import requests

    embed_fn = embed_fn or embed_paper
    fields = (
        "references.title,references.abstract,references.externalIds,"
        "citations.title,citations.abstract,citations.externalIds,citations.intents"
    )
    resp = _get_with_retry(f"{S2_PAPER_BASE}/{_s2_id(doi)}", params={"fields": fields}, timeout=15)
    data = resp.json()
    time.sleep(0.5)  # rate-limit courtesy, same pattern as fetch_related.py

    return {
        "references": _shape_and_score(data.get("references") or [], seed_embeddings, embed_fn, is_citation=False, is_known_fn=is_known_fn),
        "citations": _shape_and_score(data.get("citations") or [], seed_embeddings, embed_fn, is_citation=True, is_known_fn=is_known_fn),
    }


def fetch_neighbors_by_search(query_text: str, seed_embeddings: dict, embed_fn=None, is_known_fn=None) -> dict:
    """For an unpublished/non-indexed seed: S2 has no record to fetch 'its' references/citations
    from, so this finds candidate related papers via full-text search instead. Only feeds the
    references bucket — there's no way to discover citations of a paper S2 doesn't know exists."""
    import requests

    embed_fn = embed_fn or embed_paper
    resp = _get_with_retry(
        f"{S2_PAPER_BASE}/search",
        params={"query": query_text, "limit": 15, "fields": "title,abstract,externalIds"},
        timeout=15,
    )
    data = resp.json()
    references = _shape_and_score(data.get("data") or [], seed_embeddings, embed_fn, is_citation=False, is_known_fn=is_known_fn)
    return {"references": references, "citations": []}


def run_paper_graph(
    seed_doi: str | None,
    seed_abstract: str,
    collection,
    seed_slug: str | None = None,
    seed_search_query: str | None = None,
    max_depth: int = 2,
    references_top_k: int = 10,
    citations_top_k: int = 5,
    relevance_floor: float = 0.65,
    full_text_threshold: float = 0.85,
    diminishing_returns_window: int | None = 5,
    cost_ceiling: int | None = 100,
    on_checkpoint=None,
    embed_fn=None,
    fetch_fn=None,
    search_fetch_fn=None,
) -> list[dict]:
    is_unpublished = seed_doi is None
    seed_id = seed_doi if seed_doi is not None else seed_slug

    needs_real_embeddings = fetch_fn is None or (is_unpublished and search_fetch_fn is None)
    seed_embeddings = None
    if needs_real_embeddings:
        embed_fn = embed_fn or embed_paper
        seed_embeddings = embed_fn(seed_abstract)

    is_known_fn = None
    if fetch_fn is None:
        is_known_fn = lambda doi: is_known(doi, collection)

        def fetch_fn(doi):
            return fetch_neighbors_from_s2(doi, seed_embeddings, embed_fn=embed_fn, is_known_fn=is_known_fn)

    if is_unpublished:
        base_fetch_fn = fetch_fn
        do_search = search_fetch_fn or fetch_neighbors_by_search

        def fetch_fn(doi):  # noqa: F811 — deliberate shadow: dispatches seed vs. everything else
            if doi == seed_id:
                if search_fetch_fn is None:
                    return do_search(seed_search_query, seed_embeddings, embed_fn, is_known_fn=is_known_fn)
                else:
                    return do_search(seed_search_query, seed_embeddings, embed_fn)
            return base_fetch_fn(doi)

    results = traverse(
        seed_doi=seed_id,
        fetch_fn=fetch_fn,
        max_depth=max_depth,
        references_top_k=references_top_k,
        citations_top_k=citations_top_k,
        relevance_floor=relevance_floor,
        cost_ceiling=cost_ceiling,
        on_checkpoint=on_checkpoint,
        diminishing_returns_window=diminishing_returns_window,
    )

    for node in results:
        if is_known(node["doi"], collection):
            continue
        abstract = node.get("abstract") or (seed_abstract if node["doi"] == seed_id else "")

        # Attempt full-text fetch for high-relevance nodes with arXiv IDs
        full_text = False
        document_text = abstract
        if node["score"] >= full_text_threshold and node.get("arxiv_id"):
            fetched_text = fetch_full_text(node)
            if fetched_text:
                document_text = fetched_text
                full_text = True

        record_paper(collection, doi=node["doi"], abstract=document_text, discovered_via=node["discovered_via"], full_text=full_text)

    return results


_EDGE_TYPE_LABEL = {"references": "reference", "citations": "citation"}


def traverse(
    seed_doi: str,
    fetch_fn,
    max_depth: int,
    references_top_k: int,
    citations_top_k: int,
    relevance_floor: float,
    cost_ceiling: int | None = None,
    on_checkpoint=None,
    diminishing_returns_window: int | None = None,
) -> list[dict]:
    results = [{"doi": seed_doi, "score": 1.0, "discovered_via": None}]
    visited = {seed_doi}
    queue: list[tuple[float, str, int]] = [(-1.0, seed_doi, 0)]
    stopped = False
    per_direction_top_k = {"references": references_top_k, "citations": citations_top_k}
    recent_scores: list[float] = []

    def ceiling_hit() -> bool:
        """ADR 0010: report queue state at the checkpoint (not just a bare
        pause) so continue/stop is an informed decision, not a guess."""
        if cost_ceiling is None or on_checkpoint is None:
            return False
        if len(results) < cost_ceiling + 1:
            return False
        decision = on_checkpoint({"nodes_so_far": len(results), "queue_size": len(queue)})
        return decision == "stop"

    def exhausted() -> bool:
        if diminishing_returns_window is None:
            return False
        return diminishing_returns(recent_scores, relevance_floor, diminishing_returns_window)

    while queue and not stopped:
        _, doi, depth = heapq.heappop(queue)
        if depth >= max_depth:
            continue

        neighbors = fetch_fn(doi)

        for edge_type, top_k in per_direction_top_k.items():
            candidates = neighbors.get(edge_type, [])
            selected = select_candidates(candidates, top_k=top_k, relevance_floor=relevance_floor)
            for cand in selected:
                if cand["doi"] in visited:
                    continue
                if ceiling_hit() or exhausted():
                    stopped = True
                    break
                visited.add(cand["doi"])
                recent_scores.append(cand["score"])
                results.append({
                    "doi": cand["doi"],
                    "score": cand["score"],
                    "abstract": cand.get("abstract", ""),
                    "arxiv_id": cand.get("arxiv_id"),
                    "discovered_via": {
                        "parent_doi": doi,
                        "edge_type": _EDGE_TYPE_LABEL[edge_type],
                        "hop_depth": depth + 1,
                    },
                })
                heapq.heappush(queue, (-cand["score"], cand["doi"], depth + 1))
            if stopped:
                break

    return results


PAPER_KNOWLEDGE_COLLECTION = "paper-knowledge"
CHROMA_PATH = Path.home() / ".claude" / "chroma"
SESSION_DIR = Path.home() / ".claude" / "paper-sessions"


def _fetch_seed_abstract(doi: str) -> str:
    resp = _get_with_retry(f"{S2_PAPER_BASE}/{_s2_id(doi)}", params={"fields": "title,abstract"}, timeout=15)
    data = resp.json()
    return data.get("abstract") or data.get("title") or ""


def _confirm_checkpoint(state: dict) -> str:
    print(
        f"\n[paper-graph] {state['nodes_so_far']} papers found so far, "
        f"{state['queue_size']} more still queued and above the relevance floor."
    )
    answer = input("Continue traversing? [y/N/more]: ").strip().lower()
    return "continue" if answer in ("y", "yes", "more") else "stop"


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Recursive citation-graph traversal from a seed paper (paper-dive).")
    ap.add_argument("--doi", help="DOI of the seed paper (skips session/slug lookup entirely)")
    ap.add_argument("--slug", help="Session slug — reads DOI/title/text from that session's state.json + raw.md")
    ap.add_argument("--show-discovery", help="Query first-discovery metadata for a DOI and exit")
    ap.add_argument("--depth", type=int, default=2, help="Max hop depth (default: 2)")
    ap.add_argument("--references-top-k", type=int, default=10)
    ap.add_argument("--citations-top-k", type=int, default=5)
    ap.add_argument("--relevance-floor", type=float, default=0.65)
    ap.add_argument("--full-text-threshold", type=float, default=0.85, help="Score threshold for full-text fetch from arXiv (default: 0.85)")
    ap.add_argument("--cost-ceiling", type=int, default=100)
    ap.add_argument("--diminishing-returns-window", type=int, default=5)
    args = ap.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(PAPER_KNOWLEDGE_COLLECTION)

    if args.show_discovery:
        discovery = get_discovery(collection, args.show_discovery)
        if discovery is None:
            print(f"No discovery record for {args.show_discovery}")
        else:
            print(f"First discovery of {args.show_discovery}:")
            print(f"  parent_doi: {discovery['parent_doi']}")
            print(f"  edge_type: {discovery['edge_type']}")
            print(f"  hop_depth: {discovery['hop_depth']}")
        raise SystemExit(0)

    doi = args.doi
    slug = args.slug
    title = None
    seed_abstract = None

    if not doi and slug:
        session = SESSION_DIR / slug
        state_path = session / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            doi = state.get("doi")
            title = state.get("title")
        raw_path = session / "raw.md"
        if raw_path.exists():
            seed_abstract = raw_path.read_text(encoding="utf-8")[:3000]

    if not doi and not slug:
        print("ERROR: provide --doi, or --slug (works for both published and unpublished/no-DOI sessions)")
        raise SystemExit(1)

    if doi:
        seed_abstract = seed_abstract or _fetch_seed_abstract(doi)
        seed_id = doi
        run_kwargs = {"seed_doi": doi}
    else:
        # unpublished/no-DOI: search-seed the first hop from title, traverse normally from there
        seed_id = slug
        run_kwargs = {"seed_doi": None, "seed_slug": slug, "seed_search_query": title or seed_abstract[:200]}

    results = run_paper_graph(
        seed_abstract=seed_abstract or "",
        collection=collection,
        max_depth=args.depth,
        references_top_k=args.references_top_k,
        citations_top_k=args.citations_top_k,
        relevance_floor=args.relevance_floor,
        full_text_threshold=args.full_text_threshold,
        diminishing_returns_window=args.diminishing_returns_window,
        cost_ceiling=args.cost_ceiling,
        on_checkpoint=_confirm_checkpoint,
        **run_kwargs,
    )

    print(f"\n# Citation graph for {seed_id}\n")
    print(f"{len(results)} papers total (including seed).\n")
    for node in sorted(results, key=lambda n: n["score"], reverse=True):
        via = node["discovered_via"]
        if via is None:
            print(f"- **{node['doi']}** (seed)")
        else:
            print(f"- **{node['doi']}** — score {node['score']:.2f}, {via['edge_type']} of {via['parent_doi']}, hop {via['hop_depth']}")


if __name__ == "__main__":
    main()
