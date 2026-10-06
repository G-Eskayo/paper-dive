#!/usr/bin/env python3
"""
competing_ideas.py — surfaces papers that support or refute a given hypothesis,
revealing disagreement patterns in the literature (paper-dive synthesis tool,
see issue #26).

v1 scope (2026-08-31): semantic search over paper-knowledge collection to find
relevant papers, then LLM-classify each as supporting, refuting, mixed, or
unrelated to the hypothesis. The interesting output is when both supports and
refutes are non-empty — that's evidence of active disagreement in the field.
Empty-conflict (all papers support or all refute) is a valid, non-error result.

Model choice (unvalidated on real abstracts, noted in docstring): qwen2.5:7b
for stance classification, same judg-grade tier as logic_auditor's extraction
task; smaller than the 14b classification model since this is simpler than
paper-type classification.

Uses ChromaDB's default semantic search (embedder unspecified in prior tools,
so we reuse whatever the collection already has) to find papers. Every
side-effecting call (search_fn, chat_fn) is injected as a parameter so tests
never need a real embedder or model.
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

CHROMA_PATH = Path.home() / ".claude" / "chroma"
COLLECTION_NAME = "paper-knowledge"
OLLAMA_URL = "http://localhost:11434/api/chat"
STANCE_MODEL = "qwen2.5:7b"

STANCES = {"supports", "refutes", "mixed", "unrelated"}


def ollama_chat(model: str, messages: list[dict], timeout: int = 60) -> str:
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def semantic_search(collection, hypothesis: str, n_results: int = 10, search_fn=None) -> list[dict]:
    """Semantic search over the collection for papers related to the hypothesis.
    Returns list of {doi, title, abstract} from the N highest-scoring results.
    search_fn defaults to collection.query() if not injected (for testing)."""
    if search_fn is None:
        search_fn = lambda query_texts, n_results: collection.query(query_texts=query_texts, n_results=n_results)

    results = search_fn(query_texts=[hypothesis], n_results=n_results)

    papers = []
    # results structure: {"ids": [[id1, id2, ...]], "documents": [[doc1, doc2, ...]]}
    ids_list = results.get("ids", [[]])[0] if results.get("ids") else []
    docs_list = results.get("documents", [[]])[0] if results.get("documents") else []

    for doi, doc in zip(ids_list, docs_list):
        if not doi:
            continue
        meta = {}
        # Try to get metadata if available
        try:
            data = collection.get(ids=[doi], include=["metadatas"])
            if data["metadatas"]:
                meta = data["metadatas"][0]
        except Exception:
            pass

        papers.append({
            "doi": doi,
            "title": meta.get("title") or doi,
            "abstract": doc,
        })

    return papers


STANCE_PROMPT = """Below is a hypothesis and an academic paper's title and abstract. Classify the paper's stance toward the hypothesis:

- supports: The paper provides evidence, arguments, or findings that support the hypothesis.
- refutes: The paper provides evidence, arguments, or findings that contradict or undermine the hypothesis.
- mixed: The paper provides both supporting and refuting evidence, or is ambiguous about the hypothesis.
- unrelated: The paper doesn't meaningfully engage with the hypothesis at all.

Hypothesis: {hypothesis}

Title: {title}

Abstract: {abstract}

Respond in exactly this format:
STANCE: <one of: supports, refutes, mixed, unrelated>
RATIONALE: <one sentence explaining how the paper relates to the hypothesis>"""


def classify_stance(hypothesis: str, title: str, abstract: str, chat_fn=None) -> dict[str, str]:
    """Classifies a paper's stance toward the hypothesis.
    Returns {"stance": one of STANCES, "rationale": str}."""
    chat_fn = chat_fn or (lambda messages: ollama_chat(STANCE_MODEL, messages))
    prompt = STANCE_PROMPT.format(hypothesis=hypothesis, title=title, abstract=abstract)
    response = chat_fn([{"role": "user", "content": prompt}])
    return _parse_stance_result(response)


def _parse_stance_result(response: str) -> dict[str, str]:
    """Parses the STANCE and RATIONALE lines from stance response."""
    result = {}
    for line in response.splitlines():
        if line.strip().upper().startswith("STANCE:"):
            stance_text = line.strip()[len("STANCE:"):].strip().lower()
            # Match whole stance words, not substrings
            import re
            found = [s for s in STANCES if re.search(r'\b' + re.escape(s) + r'\b', stance_text)]
            result["stance"] = found[0] if len(found) == 1 else "unknown"
        elif line.strip().upper().startswith("RATIONALE:"):
            result["rationale"] = line.strip()[len("RATIONALE:"):].strip()
    return result


def build_competing_ideas_map(
    collection,
    hypothesis: str,
    n_results: int = 10,
    search_fn=None,
    chat_fn=None,
) -> dict[str, list[dict]]:
    """For a hypothesis, searches paper-knowledge, classifies stance, and groups
    papers by stance. Drops 'unrelated' papers. Returns {"supports": [...],
    "refutes": [...], "mixed": [...]}, each entry {doi, title, abstract, rationale}."""
    papers = semantic_search(collection, hypothesis, n_results=n_results, search_fn=search_fn)

    result = {"supports": [], "refutes": [], "mixed": []}

    for paper in papers:
        stance_result = classify_stance(hypothesis, paper["title"], paper["abstract"], chat_fn=chat_fn)
        stance = stance_result.get("stance", "unknown")
        rationale = stance_result.get("rationale", "")

        if stance in result:
            result[stance].append({
                "doi": paper["doi"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "rationale": rationale,
            })

    return result


def render_markdown(hypothesis: str, map_result: dict[str, list[dict]]) -> str:
    lines = [
        "# Competing Ideas — Hypothesis Analysis",
        "",
        "Generated by `competing_ideas.py`. Semantic search + stance classification "
        "(qwen2.5:7b, unvalidated on real abstracts) to surface supporting, refuting, "
        "and mixed evidence on a hypothesis. Papers classified as unrelated are excluded.",
        "",
        f"**Hypothesis**: {hypothesis}",
        "",
    ]

    # Show supports
    supports = map_result.get("supports", [])
    lines.append(f"## Supporting Evidence ({len(supports)} paper{'s' if len(supports) != 1 else ''})")
    if supports:
        for paper in supports:
            lines.append(f"### {paper['title']}")
            lines.append(f"**DOI**: {paper['doi']}")
            lines.append(f"**Rationale**: {paper['rationale']}")
            lines.append("")
    else:
        lines.append("No papers support this hypothesis in the collection.")
        lines.append("")

    # Show refutes
    refutes = map_result.get("refutes", [])
    lines.append(f"## Refuting Evidence ({len(refutes)} paper{'s' if len(refutes) != 1 else ''})")
    if refutes:
        for paper in refutes:
            lines.append(f"### ⚠️ {paper['title']}")
            lines.append(f"**DOI**: {paper['doi']}")
            lines.append(f"**Rationale**: {paper['rationale']}")
            lines.append("")
    else:
        lines.append("No papers refute this hypothesis in the collection.")
        lines.append("")

    # Show mixed
    mixed = map_result.get("mixed", [])
    if mixed:
        lines.append(f"## Mixed Evidence ({len(mixed)} paper{'s' if len(mixed) != 1 else ''})")
        for paper in mixed:
            lines.append(f"### {paper['title']}")
            lines.append(f"**DOI**: {paper['doi']}")
            lines.append(f"**Rationale**: {paper['rationale']}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Surface competing ideas on a hypothesis from paper-knowledge.")
    ap.add_argument("--hypothesis", required=True, help="The hypothesis to investigate")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-markdown", required=True)
    ap.add_argument("--n-results", type=int, default=10, help="Number of papers to search for (default 10)")
    args = ap.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    map_result = build_competing_ideas_map(collection, args.hypothesis, n_results=args.n_results)

    Path(args.out_json).write_text(json.dumps(map_result, indent=2))
    Path(args.out_markdown).write_text(render_markdown(args.hypothesis, map_result))

    total = sum(len(v) for v in map_result.values())
    print(f"[competing-ideas] analyzed {total} paper(s) and wrote to {args.out_json} and {args.out_markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
