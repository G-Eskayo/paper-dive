"""Tests for argument_mapper.py. Run via:
    ~/.agents/venv/bin/python -m pytest scripts/tests/test_argument_mapper.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import chromadb
import pytest

from argument_mapper import (
    extract_core_claim,
    _openalex_work_id,
    fetch_title,
    enrich_missing_titles,
    build_argument_map,
    render_markdown,
)


# ── extract_core_claim ──────────────────────────────────────────────────────

def test_extract_core_claim_strips_whitespace_and_quotes():
    def fake_chat(messages):
        return '  "Models weight context best at the start or end."  '

    claim = extract_core_claim("Lost in the Middle", "some abstract", chat_fn=fake_chat)
    assert claim == "Models weight context best at the start or end."


def test_extract_core_claim_includes_title_and_abstract_in_prompt():
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "a claim"

    extract_core_claim("My Title", "My abstract text", chat_fn=fake_chat)
    assert "My Title" in captured["prompt"]
    assert "My abstract text" in captured["prompt"]


# ── _openalex_work_id ────────────────────────────────────────────────────────

def test_openalex_work_id_extracts_w_id_from_full_url():
    assert _openalex_work_id("https://openalex.org/W7151672805") == "W7151672805"


def test_openalex_work_id_prefixes_bare_doi():
    assert _openalex_work_id("10.48550/arxiv.2307.03172") == "doi:10.48550/arxiv.2307.03172"


# ── fetch_title ──────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_fetch_title_returns_title_on_success():
    def fake_get(url, timeout=None):
        assert url.endswith("/works/doi:10.1/x")
        return _FakeResp(200, {"title": "A Real Paper Title"})

    assert fetch_title("10.1/x", get_fn=fake_get) == "A Real Paper Title"


def test_fetch_title_returns_none_on_non_200():
    def fake_get(url, timeout=None):
        return _FakeResp(404)

    assert fetch_title("10.1/missing", get_fn=fake_get) is None


# ── enrich_missing_titles ────────────────────────────────────────────────────

def test_enrich_missing_titles_backfills_only_records_without_one(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection("paper-knowledge")
    col.add(
        ids=["has-title", "no-title"],
        documents=["abstract a", "abstract b"],
        metadatas=[{"doi": "has-title", "title": "Already Known"}, {"doi": "no-title"}],
    )

    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return _FakeResp(200, {"title": "Fetched Title"})

    updated = enrich_missing_titles(col, get_fn=fake_get)

    assert updated == 1
    assert len(calls) == 1  # only the record missing a title triggered a lookup
    result = col.get(ids=["no-title"])
    assert result["metadatas"][0]["title"] == "Fetched Title"


def test_enrich_missing_titles_skips_records_where_lookup_fails(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection("paper-knowledge")
    col.add(ids=["no-title"], documents=["abstract"], metadatas=[{"doi": "no-title"}])

    updated = enrich_missing_titles(col, get_fn=lambda url, timeout=None: _FakeResp(404))

    assert updated == 0
    result = col.get(ids=["no-title"])
    assert "title" not in result["metadatas"][0]


# ── build_argument_map ───────────────────────────────────────────────────────

def _seed_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection("paper-knowledge")
    col.add(
        ids=["seed-a", "ref-1", "ref-2", "unrelated-seed", "helical-engine"],
        documents=[
            "Seed A's full abstract text.",
            "Reference 1's abstract.",
            "Reference 2's abstract.",
            "A different seed's abstract, not requested in this map.",
            "Unrelated leftover test data from a different project entirely.",
        ],
        metadatas=[
            {"doi": "seed-a"},
            {"doi": "ref-1", "parent_doi": "seed-a", "edge_type": "reference", "hop_depth": 1, "title": "Reference One"},
            {"doi": "ref-2", "parent_doi": "seed-a", "edge_type": "reference", "hop_depth": 1, "title": "Reference Two"},
            {"doi": "unrelated-seed"},
            {"doi": "helical-engine"},
        ],
    )
    return col


def test_build_argument_map_extracts_claim_only_for_requested_seeds(tmp_path):
    col = _seed_collection(tmp_path)
    calls = []

    def fake_chat(messages):
        calls.append(messages[0]["content"])
        return "Seed A's core claim."

    result = build_argument_map(
        col, seed_slugs=["seed-a"], seed_titles={"seed-a": "Seed A Title"}, chat_fn=fake_chat,
    )

    assert len(calls) == 1  # not called for ref-1, ref-2, unrelated-seed, or helical-engine
    assert len(result) == 1
    assert result[0]["slug"] == "seed-a"
    assert result[0]["title"] == "Seed A Title"
    assert result[0]["core_claim"] == "Seed A's core claim."


def test_build_argument_map_groups_discovered_references_under_their_seed(tmp_path):
    col = _seed_collection(tmp_path)

    result = build_argument_map(
        col, seed_slugs=["seed-a"], seed_titles={"seed-a": "Seed A Title"},
        chat_fn=lambda messages: "claim",
    )

    titles = {r["title"] for r in result[0]["builds_on"]}
    assert titles == {"Reference One", "Reference Two"}


def test_build_argument_map_excludes_unrelated_papers_from_builds_on(tmp_path):
    col = _seed_collection(tmp_path)

    result = build_argument_map(
        col, seed_slugs=["seed-a"], seed_titles={"seed-a": "Seed A Title"},
        chat_fn=lambda messages: "claim",
    )

    all_dois_shown = {r["doi"] for r in result[0]["builds_on"]}
    assert "unrelated-seed" not in all_dois_shown
    assert "helical-engine" not in all_dois_shown


def test_build_argument_map_falls_back_to_slug_when_no_title_given(tmp_path):
    col = _seed_collection(tmp_path)

    result = build_argument_map(
        col, seed_slugs=["seed-a"], seed_titles={}, chat_fn=lambda messages: "claim",
    )

    assert result[0]["title"] == "seed-a"


def test_build_argument_map_skips_seed_slugs_not_in_the_collection(tmp_path):
    col = _seed_collection(tmp_path)

    result = build_argument_map(
        col, seed_slugs=["seed-a", "never-seeded"], seed_titles={"seed-a": "Seed A"},
        chat_fn=lambda messages: "claim",
    )

    assert [r["slug"] for r in result] == ["seed-a"]


# ── render_markdown ──────────────────────────────────────────────────────────

def test_render_markdown_includes_title_claim_and_references():
    argument_map = [
        {
            "slug": "seed-a",
            "title": "Seed A Title",
            "core_claim": "Seed A's core claim.",
            "builds_on": [{"doi": "ref-1", "title": "Reference One"}],
        }
    ]
    md = render_markdown(argument_map)
    assert "Seed A Title" in md
    assert "Seed A's core claim." in md
    assert "Reference One" in md


def test_render_markdown_handles_no_references_gracefully():
    argument_map = [
        {"slug": "seed-a", "title": "Seed A Title", "core_claim": "A claim.", "builds_on": []}
    ]
    md = render_markdown(argument_map)
    assert "no discovered references" in md.lower()
