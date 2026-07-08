"""Tests for paper-dive's paper_graph module. Run via:
    ~/.agents/venv/bin/python -m pytest scripts/tests/test_paper_graph.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import chromadb
import pytest

from paper_graph import (
    select_candidates,
    is_known,
    traverse,
    record_paper,
    diminishing_returns,
    blended_score,
    embed_paper,
    fetch_neighbors_from_s2,
    fetch_neighbors_by_search,
    run_paper_graph,
    _get_with_retry,
)


# ── select_candidates ────────────────────────────────────────────────────────

def test_select_candidates_returns_top_k_by_score():
    candidates = [
        {"doi": "A", "score": 0.9, "intent": None},
        {"doi": "B", "score": 0.8, "intent": None},
        {"doi": "C", "score": 0.7, "intent": None},
    ]
    result = select_candidates(candidates, top_k=2, relevance_floor=0.0)
    assert [c["doi"] for c in result] == ["A", "B"]


def test_select_candidates_excludes_below_relevance_floor():
    candidates = [
        {"doi": "A", "score": 0.9, "intent": None},
        {"doi": "B", "score": 0.6, "intent": None},  # below floor
    ]
    result = select_candidates(candidates, top_k=5, relevance_floor=0.65)
    assert [c["doi"] for c in result] == ["A"]


def test_select_candidates_bypasses_cap_for_result_intent():
    candidates = [
        {"doi": "A", "score": 0.9, "intent": None},
        {"doi": "B", "score": 0.8, "intent": None},
        {"doi": "C", "score": 0.3, "intent": "result"},  # below floor, but result-intent
    ]
    result = select_candidates(candidates, top_k=2, relevance_floor=0.65)
    assert {c["doi"] for c in result} == {"A", "B", "C"}


# ── is_known (dedup-before-fetch) ────────────────────────────────────────────

def test_is_known_distinguishes_known_and_unknown_dois(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")
    collection.add(
        ids=["10.1234/known"],
        documents=["abstract text"],
        metadatas=[{"doi": "10.1234/known"}],
    )

    assert is_known("10.1234/known", collection) is True
    assert is_known("10.1234/unknown", collection) is False


def test_record_paper_makes_it_known(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    assert is_known("10.5678/new", collection) is False

    record_paper(
        collection,
        doi="10.5678/new",
        abstract="an abstract about something",
        discovered_via={"parent_doi": "seed", "edge_type": "reference", "hop_depth": 1},
    )

    assert is_known("10.5678/new", collection) is True


# ── diminishing_returns ───────────────────────────────────────────────────────

def test_diminishing_returns_false_when_recent_scores_are_comfortably_above_floor():
    recent_scores = [0.85, 0.82, 0.88, 0.9, 0.87]
    assert diminishing_returns(recent_scores, relevance_floor=0.65, window=5) is False


def test_diminishing_returns_true_when_recent_scores_hover_near_floor():
    recent_scores = [0.68, 0.66, 0.67, 0.69, 0.65]
    assert diminishing_returns(recent_scores, relevance_floor=0.65, window=5) is True


def test_diminishing_returns_false_when_not_enough_history_yet():
    recent_scores = [0.66, 0.67]  # fewer than `window` data points
    assert diminishing_returns(recent_scores, relevance_floor=0.65, window=5) is False


# ── traverse ──────────────────────────────────────────────────────────────────

def test_traverse_depth_1_keeps_surviving_reference_and_drops_the_rest():
    graph = {
        "seed": {
            "references": [
                {"doi": "ref-high", "score": 0.9, "intent": None},
                {"doi": "ref-low", "score": 0.3, "intent": None},  # below floor
            ],
            "citations": [],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.65,
    )

    dois = {node["doi"] for node in result}
    assert dois == {"seed", "ref-high"}

    ref_high = next(n for n in result if n["doi"] == "ref-high")
    assert ref_high["discovered_via"] == {
        "parent_doi": "seed",
        "edge_type": "reference",
        "hop_depth": 1,
    }


def test_traverse_stops_exactly_at_max_depth():
    graph = {
        "seed": {"references": [{"doi": "hop1", "score": 0.9, "intent": None}], "citations": []},
        "hop1": {"references": [{"doi": "hop2", "score": 0.9, "intent": None}], "citations": []},
        "hop2": {"references": [{"doi": "hop3", "score": 0.9, "intent": None}], "citations": []},
    }

    def fake_fetch(doi):
        return graph[doi]

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=2,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.65,
    )

    dois = {node["doi"] for node in result}
    assert dois == {"seed", "hop1", "hop2"}  # hop3 is one hop past the limit


# ── checkpoint-and-confirm stopping ──────────────────────────────────────────

def test_traverse_stops_at_cost_ceiling_when_checkpoint_says_stop():
    graph = {
        "seed": {
            "references": [
                {"doi": "n1", "score": 0.9, "intent": None},
                {"doi": "n2", "score": 0.85, "intent": None},
                {"doi": "n3", "score": 0.8, "intent": None},
            ],
            "citations": [],
        },
        "n1": {"references": [], "citations": []},
        "n2": {"references": [], "citations": []},
        "n3": {"references": [], "citations": []},
    }

    def fake_fetch(doi):
        return graph[doi]

    def stop_checkpoint(state):
        return "stop"

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=2,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        cost_ceiling=2,
        on_checkpoint=stop_checkpoint,
    )

    # seed + at most cost_ceiling more nodes — not all 3 references
    assert len(result) <= 1 + 2


def test_traverse_continues_past_ceiling_when_checkpoint_says_continue():
    graph = {
        "seed": {
            "references": [
                {"doi": "n1", "score": 0.9, "intent": None},
                {"doi": "n2", "score": 0.85, "intent": None},
                {"doi": "n3", "score": 0.8, "intent": None},
            ],
            "citations": [],
        },
        "n1": {"references": [], "citations": []},
        "n2": {"references": [], "citations": []},
        "n3": {"references": [], "citations": []},
    }

    def fake_fetch(doi):
        return graph[doi]

    def continue_checkpoint(state):
        return "continue"

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=2,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        cost_ceiling=2,
        on_checkpoint=continue_checkpoint,
    )

    # all 3 references present — the ceiling didn't stop anything once "continue" was chosen
    assert {n["doi"] for n in result} == {"seed", "n1", "n2", "n3"}


def test_traverse_stops_early_via_diminishing_returns_without_hitting_ceiling():
    # scores hover right at the floor from the very first hop — should self-terminate
    # long before the (deliberately generous) cost_ceiling would ever matter.
    graph = {
        "seed": {
            "references": [
                {"doi": "n1", "score": 0.66, "intent": None},
                {"doi": "n2", "score": 0.67, "intent": None},
                {"doi": "n3", "score": 0.65, "intent": None},
            ],
            "citations": [],
        },
        "n1": {"references": [{"doi": "n1-child", "score": 0.66, "intent": None}], "citations": []},
        "n2": {"references": [{"doi": "n2-child", "score": 0.67, "intent": None}], "citations": []},
        "n3": {"references": [{"doi": "n3-child", "score": 0.65, "intent": None}], "citations": []},
    }

    def fake_fetch(doi):
        return graph[doi]

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=5,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.65,
        diminishing_returns_window=3,
        cost_ceiling=100,  # generous — should never actually be reached
    )

    # stopped once 3 near-floor scores accumulated, well before all 6 descendants existed
    assert len(result) < 7


# ── blended_score (SPECTER2 + nomic-embed) ───────────────────────────────────

def test_blended_score_is_one_for_identical_embeddings():
    seed = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}
    candidate = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}
    assert blended_score(seed, candidate, specter2_weight=0.5) == pytest.approx(1.0)


def test_blended_score_actually_blends_both_sources():
    # specter2 says "identical" (cosine=1.0), nomic says "orthogonal" (cosine=0.0)
    seed = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}
    candidate = {"specter2": [1.0, 0.0], "nomic": [0.0, 1.0]}
    # weight 0.5 each → blended should land at 0.5, not collapse to either source alone
    assert blended_score(seed, candidate, specter2_weight=0.5) == pytest.approx(0.5)


# ── embed_paper (injected backends — no real model calls in the fast suite) ─

def test_embed_paper_wires_injected_backends_into_expected_shape():
    result = embed_paper(
        "some paper text",
        specter2_fn=lambda text: [1.0, 0.0],
        nomic_fn=lambda text: [0.0, 1.0],
    )
    assert result == {"specter2": [1.0, 0.0], "nomic": [0.0, 1.0]}


# ── fetch_neighbors_from_s2 (real API boundary, mocked at the HTTP layer) ───

class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_neighbors_from_s2_shapes_references_and_citations(monkeypatch):
    s2_payload = {
        "references": [
            {"title": "Foundational Paper", "abstract": "an early paper", "externalIds": {"DOI": "10.1/ref-a"}},
            {"title": "No DOI Paper", "abstract": "skip me", "externalIds": {}},  # no DOI — should be skipped
        ],
        "citations": [
            {
                "title": "Rebuttal Paper",
                "abstract": "disputes the seed's findings",
                "externalIds": {"DOI": "10.1/cite-a"},
                "intents": ["result"],
            },
            {
                "title": "Passing Mention",
                "abstract": "briefly cites the seed",
                "externalIds": {"DOI": "10.1/cite-b"},
                "intents": ["background"],
            },
        ],
    }

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(s2_payload)

    monkeypatch.setattr("requests.get", fake_get)

    seed_embeddings = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    def fake_embed_paper(text, specter2_fn=None, nomic_fn=None):
        return {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}  # identical to seed → score 1.0

    result = fetch_neighbors_from_s2(
        doi="10.1/seed",
        seed_embeddings=seed_embeddings,
        embed_fn=fake_embed_paper,
    )

    ref_dois = {c["doi"] for c in result["references"]}
    assert ref_dois == {"10.1/ref-a"}  # no-DOI paper skipped

    cite_by_doi = {c["doi"]: c for c in result["citations"]}
    assert cite_by_doi["10.1/cite-a"]["intent"] == "result"
    assert cite_by_doi["10.1/cite-b"]["intent"] is None  # "background" is not "result"
    assert cite_by_doi["10.1/cite-a"]["score"] == pytest.approx(1.0)


# ── fetch_neighbors_by_search (unpublished/non-DOI seed papers) ─────────────

def test_fetch_neighbors_by_search_shapes_results_as_references_only(monkeypatch):
    s2_payload = {
        "data": [
            {"title": "A Related Paper", "abstract": "on the same topic", "externalIds": {"DOI": "10.1/found-a"}},
            {"title": "No DOI Result", "abstract": "no doi here", "externalIds": {}},  # skipped
        ]
    }

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(s2_payload)

    monkeypatch.setattr("requests.get", fake_get)

    seed_embeddings = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    def fake_embed_paper(text, specter2_fn=None, nomic_fn=None):
        return {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    result = fetch_neighbors_by_search(
        query_text="an unpublished draft about some topic",
        seed_embeddings=seed_embeddings,
        embed_fn=fake_embed_paper,
    )

    assert {c["doi"] for c in result["references"]} == {"10.1/found-a"}
    assert result["citations"] == []  # an unindexed seed has no discoverable citations


def test_run_paper_graph_uses_search_for_unpublished_seed(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    # after the search-seeded hop, "found-a" is a real, DOI'd paper — its own
    # expansion should go through the normal per-DOI fetch, not search again.
    def fake_search_fetch(query_text, seed_embeddings, embed_fn):
        return {
            "references": [{"doi": "found-a", "score": 0.9, "intent": None, "abstract": "related"}],
            "citations": [],
        }

    def fake_doi_fetch(doi):
        assert doi == "found-a"
        return {"references": [], "citations": []}

    results = run_paper_graph(
        seed_doi=None,  # no DOI — this is an unpublished draft
        seed_abstract="my unpublished draft's abstract",
        seed_slug="my-unpublished-draft",
        seed_search_query="my unpublished draft topic keywords",
        collection=collection,
        max_depth=2,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        search_fetch_fn=fake_search_fetch,
        fetch_fn=fake_doi_fetch,
    )

    assert {n["doi"] for n in results} == {"my-unpublished-draft", "found-a"}


# ── _get_with_retry (429 rate-limit handling) ────────────────────────────────

def test_get_with_retry_retries_on_429_then_succeeds(monkeypatch):
    import requests

    calls = {"count": 0}

    class FakeResp:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code == 429:
                raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return {"ok": True}

    def fake_get(url, params=None, timeout=None):
        calls["count"] += 1
        return FakeResp(429) if calls["count"] == 1 else FakeResp(200)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    resp = _get_with_retry("http://example.com", params={}, timeout=15, max_retries=3)

    assert resp.json() == {"ok": True}
    assert calls["count"] == 2


def test_get_with_retry_gives_up_after_max_retries(monkeypatch):
    import requests

    class FakeResp:
        status_code = 429

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    monkeypatch.setattr("requests.get", lambda url, params=None, timeout=None: FakeResp())
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(requests.exceptions.HTTPError):
        _get_with_retry("http://example.com", params={}, timeout=15, max_retries=2)


# ── run_paper_graph (top-level orchestration) ────────────────────────────────

def test_run_paper_graph_records_newly_discovered_papers(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    graph = {
        "seed": {
            "references": [{"doi": "ref-a", "score": 0.9, "intent": None, "abstract": "an abstract"}],
            "citations": [],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    results = run_paper_graph(
        seed_doi="seed",
        seed_abstract="the seed paper's abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        fetch_fn=fake_fetch,
    )

    assert {n["doi"] for n in results} == {"seed", "ref-a"}
    assert is_known("seed", collection) is True
    assert is_known("ref-a", collection) is True


def test_run_paper_graph_skips_recording_already_known_papers(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")
    record_paper(collection, doi="ref-a", abstract="already have this one", discovered_via=None)

    graph = {
        "seed": {
            "references": [{"doi": "ref-a", "score": 0.9, "intent": None, "abstract": "an abstract"}],
            "citations": [],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    # should not raise (e.g. a duplicate-ID error from re-adding to the collection)
    run_paper_graph(
        seed_doi="seed",
        seed_abstract="the seed paper's abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        fetch_fn=fake_fetch,
    )
