#!/usr/bin/env python3
"""
ingest_paper.py — extract text from a scientific paper (PDF, URL, or text paste).

Usage:
    python ingest_paper.py <filepath_or_url> [--slug my-paper-slug]

Outputs structured Markdown to stdout. Saves raw.md + state.json to
~/.claude/paper-sessions/<slug>/ if --slug is provided.
Exits non-zero on unrecoverable failure.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


SESSION_DIR = Path.home() / ".claude" / "paper-sessions"


def extract_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print("ERROR: missing pdfminer.six — run: ~/.agents/venv/bin/pip install pdfminer.six", file=sys.stderr)
        sys.exit(2)
    text = extract_text(str(path))
    if not text or not text.strip():
        print(f"ERROR: no text extracted from {path.name} — may be a scanned/image PDF", file=sys.stderr)
        sys.exit(1)
    return text.strip()


def extract_url(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("ERROR: missing requests/beautifulsoup4 — run: ~/.agents/venv/bin/pip install requests beautifulsoup4", file=sys.stderr)
        sys.exit(2)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: failed to fetch {url}: {e}", file=sys.stderr)
        sys.exit(1)

    content_type = resp.headers.get("content-type", "")
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        tmp = Path("/tmp/_paper_ingest.pdf")
        tmp.write_bytes(resp.content)
        return extract_pdf(tmp)

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()


def parse_metadata(text: str) -> dict:
    """Best-effort extraction of title, authors, DOI from raw text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    doi_match = re.search(r'\b(10\.\d{4,}/\S+)', text)
    return {
        "title": lines[0] if lines else "Unknown",
        "doi": doi_match.group(1).rstrip(".,)") if doi_match else None,
        "first_lines": lines[:5],
    }


def slugify(title: str) -> str:
    words = re.sub(r'[^a-z0-9\s]', '', title.lower()).split()
    return "-".join(words[:6]) or "paper"


def save_session(slug: str, raw_text: str, meta: dict) -> Path:
    session = SESSION_DIR / slug
    session.mkdir(parents=True, exist_ok=True)

    (session / "raw.md").write_text(f"# {meta['title']}\n\n{raw_text}", encoding="utf-8")

    state_path = session / "state.json"
    existing = {}
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text())
        except Exception:
            pass
    state = {**existing, "slug": slug, "title": meta["title"], "doi": meta.get("doi"),
             "ladder_level": existing.get("ladder_level", 0), "lexicon_terms": existing.get("lexicon_terms", [])}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="PDF file path or URL")
    ap.add_argument("--slug", help="session slug (auto-derived from title if omitted)")
    args = ap.parse_args()

    source = args.source

    if source.startswith("http://") or source.startswith("https://"):
        raw = extract_url(source)
    else:
        p = Path(source).expanduser().resolve()
        if not p.exists():
            print(f"ERROR: file not found: {source}", file=sys.stderr)
            sys.exit(1)
        if p.suffix.lower() == ".pdf":
            raw = extract_pdf(p)
        else:
            raw = p.read_text(encoding="utf-8", errors="replace")

    meta = parse_metadata(raw)
    slug = args.slug or slugify(meta["title"])
    session_dir = save_session(slug, raw, meta)

    print(f"## Ingested: {meta['title']}")
    print(f"DOI: {meta['doi'] or 'not detected'}")
    print(f"Session: {session_dir}")
    print(f"Slug: {slug}")
    print()
    print("--- BEGIN RAW TEXT (first 3000 chars) ---")
    print(raw[:3000])
    print("--- END PREVIEW ---")


if __name__ == "__main__":
    main()
