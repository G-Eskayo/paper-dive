#!/usr/bin/env python3
"""
argument_mapper.py — extracts each seed paper's core claim and maps which
discovered reference papers it structurally builds on (paper-dive synthesis
tool, see marvin-roadmap.md §K: "Argument mapper — extract the core claim
chain: which paper's finding is the foundation each subsequent paper builds
on.").

Scope (v1, 2026-07-13): core-claim extraction runs only for hand-picked
seed papers, not the auto-discovered references pulled in by embedding-
similarity search — those haven't been vetted for relevance/quality the
way a hand-picked seed has, and running an LLM call per seed only (not
per seed x every discovered neighbor) keeps this cheap. Discovered
references are shown by title only, structurally: the citation-graph
edge (parent_doi == seed) already IS the "builds on" relationship,
paper_graph.py's traversal only ever discovers *references* here (never
*citations* — build_bibliography.py's OpenAlex search-seeding hardcodes
citations: []), which is exactly the backward/foundational direction this
tool needs.

Uses the local qwen2.5:3b model via Ollama for claim extraction — zero API
cost, matches this project's other local-model usage (see
feedback-low-cost-experimentation) and harness_v0.py's ollama_chat pattern.

Title backfill: record_paper() never stored a 'title' field (only
doi/abstract/edge info) — this fetches it from OpenAlex for any record
missing one and writes it back to the collection, so future synthesis
tools built on paper-knowledge don't hit the same gap.
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

CHROMA_PATH = Path.home() / ".claude" / "chroma"
COLLECTION_NAME = "paper-knowledge"
OLLAMA_URL = "http://localhost:11434/api/chat"
CLAIM_MODEL = "qwen2.5:3b"
OPENALEX_WORKS_BASE = "https://api.openalex.org/works"


def ollama_chat(model: str, messages: list[dict], timeout: int = 60) -> str:
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


CLAIM_PROMPT = """Below is an academic paper's title and abstract. State its single core claim or finding in ONE crisp sentence -- what did they actually show or argue, not what topic they cover. No preamble, no "This paper...", just the claim itself.

Title: {title}

Abstract: {abstract}

Core claim (one sentence):"""


def extract_core_claim(title: str, abstract: str, chat_fn=None) -> str:
    chat_fn = chat_fn or (lambda messages: ollama_chat(CLAIM_MODEL, messages))
    prompt = CLAIM_PROMPT.format(title=title, abstract=abstract)
    response = chat_fn([{"role": "user", "content": prompt}])
    return response.strip().strip('"')


def _openalex_work_id(doi: str) -> str:
    """OpenAlex's /works/{id} path accepts either its own W-prefixed id
    (already a full URL from a search result) or a bare DOI, which must be
    explicitly prefixed with 'doi:' to be recognized as one."""
    if doi.startswith("https://openalex.org/"):
        return doi.rsplit("/", 1)[-1]
    return f"doi:{doi}"


def fetch_title(doi: str, get_fn=None) -> str | None:
    import requests

    get_fn = get_fn or requests.get
    work_id = _openalex_work_id(doi)
    resp = get_fn(f"{OPENALEX_WORKS_BASE}/{work_id}", timeout=15)
    if resp.status_code != 200:
        return None
    return resp.json().get("title")


def enrich_missing_titles(collection, get_fn=None) -> int:
    data = collection.get(include=["metadatas"])
    updated = 0
    for doi, meta in zip(data["ids"], data["metadatas"]):
        if meta.get("title"):
            continue
        title = fetch_title(doi, get_fn=get_fn)
        if not title:
            continue
        collection.update(ids=[doi], metadatas=[{**meta, "title": title}])
        updated += 1
    return updated


def build_argument_map(
    collection,
    seed_slugs: list[str],
    seed_titles: dict[str, str] | None = None,
    chat_fn=None,
) -> list[dict]:
    seed_titles = seed_titles or {}
    data = collection.get(include=["documents", "metadatas"])
    by_id = {i: (doc, meta) for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])}

    references_by_seed: dict[str, list[dict]] = {}
    for doi, (doc, meta) in by_id.items():
        parent = meta.get("parent_doi")
        if parent in seed_slugs:
            references_by_seed.setdefault(parent, []).append({
                "doi": doi,
                "title": meta.get("title") or doc[:80],
            })

    result = []
    for slug in seed_slugs:
        if slug not in by_id:
            continue
        doc, meta = by_id[slug]
        title = seed_titles.get(slug) or meta.get("title") or slug
        claim = extract_core_claim(title, doc, chat_fn=chat_fn)
        result.append({
            "slug": slug,
            "title": title,
            "core_claim": claim,
            "builds_on": sorted(references_by_seed.get(slug, []), key=lambda r: r["title"]),
        })
    return result


def render_markdown(argument_map: list[dict]) -> str:
    lines = [
        "# Argument Map — Related Work Claim Chain",
        "",
        "Generated by `argument_mapper.py`. Each seed's core claim (LLM-extracted, "
        "qwen2.5:3b) plus the discovered reference papers it structurally builds on "
        "(via OpenAlex citation-graph traversal, hop 1).",
        "",
    ]
    for entry in argument_map:
        lines.append(f"## {entry['title']}")
        lines.append("")
        lines.append(f"**Core claim**: {entry['core_claim']}")
        lines.append("")
        if entry["builds_on"]:
            lines.append("**Builds on**:")
            for ref in entry["builds_on"]:
                lines.append(f"- {ref['title']}")
        else:
            lines.append("**Builds on**: (no discovered references)")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build an argument/claim-chain map from paper-knowledge.")
    ap.add_argument("--seeds-json", required=True, help="Path to a JSON file: {slug: title, ...}")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-markdown", required=True)
    ap.add_argument("--skip-title-enrichment", action="store_true")
    args = ap.parse_args()

    import chromadb

    seed_titles = json.loads(Path(args.seeds_json).read_text())
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if not args.skip_title_enrichment:
        updated = enrich_missing_titles(collection)
        print(f"[argument-mapper] backfilled {updated} missing title(s)", file=sys.stderr)

    argument_map = build_argument_map(collection, seed_slugs=list(seed_titles), seed_titles=seed_titles)

    Path(args.out_json).write_text(json.dumps(argument_map, indent=2))
    Path(args.out_markdown).write_text(render_markdown(argument_map))
    print(f"[argument-mapper] wrote {len(argument_map)} seed(s) to {args.out_json} and {args.out_markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
