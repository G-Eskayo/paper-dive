"""Tests for paper-dive's paper_graph module. Run via:
    ~/.agents/venv/bin/python -m pytest scripts/tests/test_paper_graph.py -v
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import chromadb
import pytest

from paper_graph import (
    select_candidates,
    is_known,
    get_discovery,
    traverse,
    record_paper,
    diminishing_returns,
    blended_score,
    embed_paper,
    fetch_neighbors_from_s2,
    fetch_neighbors_by_search,
    run_paper_graph,
    _get_with_retry,
    _s2_id,
    _check_huggingface_reachable,
    _cosine_similarity,
    _shape_and_score,
    fetch_arxiv_pdf_text,
    fetch_full_text,
)

LIB = Path.home() / ".agents" / "lib"
sys.path.insert(0, str(LIB))
import network_reachability  # noqa: E402


# ── _s2_id (DOI vs arXiv ID namespace detection) ────────────────────────────

def test_s2_id_prefixes_bare_doi():
    assert _s2_id("10.1234/example") == "DOI:10.1234/example"


def test_s2_id_prefixes_bare_arxiv_id():
    assert _s2_id("2503.03704") == "ARXIV:2503.03704"


def test_s2_id_prefixes_bare_arxiv_id_with_version_suffix():
    assert _s2_id("2503.03704v2") == "ARXIV:2503.03704v2"


def test_s2_id_passes_through_already_prefixed_identifiers():
    assert _s2_id("DOI:10.1234/example") == "DOI:10.1234/example"
    assert _s2_id("ARXIV:2503.03704") == "ARXIV:2503.03704"


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


def test_record_paper_stamps_created_epoch(tmp_path):
    # cross_machine_merge.py's incremental sync filters on this field via
    # dump_collection.py's --since -- without it, every sync re-transfers
    # the whole collection instead of only what's new.
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    before = time.time()
    record_paper(collection, doi="10.5678/timed", abstract="abstract", discovered_via=None)
    after = time.time()

    meta = collection.get(ids=["10.5678/timed"])["metadatas"][0]
    assert before <= meta["created_epoch"] <= after


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


def test_traverse_checkpoint_receives_accurate_queue_state():
    # ADR 0010 requires the checkpoint to report queue state (not only pause silently) so
    # continue/stop is an informed decision -- this pins down exactly what gets reported.
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

    seen_states = []

    def spy_checkpoint(state):
        seen_states.append(state)
        return "stop"

    traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=2,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        cost_ceiling=1,
        on_checkpoint=spy_checkpoint,
    )

    # ceiling=1 is hit right after n1 is accepted (seed + n1 = 2 nodes so far), with n1
    # itself still sitting in the queue unexpanded, and n2 not yet checked against the ceiling
    assert seen_states == [{"nodes_so_far": 2, "queue_size": 1}]


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


def test_traverse_follows_both_directions_with_independent_caps():
    # Ticket #21 AC: asymmetric caps apply independently to references and citations.
    # Seed has 3 above-floor references and 3 above-floor citations, with
    # references_top_k=2 and citations_top_k=1 — only the top 2 refs and top 1 cite survive.
    graph = {
        "seed": {
            "references": [
                {"doi": "ref-1", "score": 0.9, "intent": None},
                {"doi": "ref-2", "score": 0.85, "intent": None},
                {"doi": "ref-3", "score": 0.75, "intent": None},  # above floor (0.65) but will be dropped by top_k=2
            ],
            "citations": [
                {"doi": "cite-1", "score": 0.88, "intent": None},
                {"doi": "cite-2", "score": 0.8, "intent": None},   # above floor but will be dropped by top_k=1
                {"doi": "cite-3", "score": 0.7, "intent": None},   # above floor but will be dropped by top_k=1
            ],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=1,
        references_top_k=2,
        citations_top_k=1,
        relevance_floor=0.65,
    )

    dois = {node["doi"] for node in result}
    # seed + 2 references (top-2 of 3 above-floor) + 1 citation (top-1 of 3 above-floor)
    assert dois == {"seed", "ref-1", "ref-2", "cite-1"}

    # Verify edge types are recorded correctly
    by_doi = {node["doi"]: node for node in result}
    assert by_doi["ref-1"]["discovered_via"]["edge_type"] == "reference"
    assert by_doi["ref-2"]["discovered_via"]["edge_type"] == "reference"
    assert by_doi["cite-1"]["discovered_via"]["edge_type"] == "citation"


def test_traverse_shared_floor_applies_to_both_directions():
    # Ticket #21 AC: shared relevance_floor rejects below-floor candidates in both directions.
    graph = {
        "seed": {
            "references": [
                {"doi": "ref-ok", "score": 0.75, "intent": None},
                {"doi": "ref-below", "score": 0.6, "intent": None},  # below 0.65 floor
            ],
            "citations": [
                {"doi": "cite-ok", "score": 0.8, "intent": None},
                {"doi": "cite-below", "score": 0.62, "intent": None},  # below 0.65 floor
            ],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=1,
        references_top_k=10,
        citations_top_k=10,
        relevance_floor=0.65,
    )

    dois = {node["doi"] for node in result}
    # seed + only the above-floor candidates from both directions
    assert dois == {"seed", "ref-ok", "cite-ok"}
    assert "ref-below" not in dois
    assert "cite-below" not in dois


def test_traverse_result_intent_citation_survives_full_traversal_over_cap():
    # Ticket #21 AC: result-intent citations bypass the top_k cap at the traverse() level,
    # not only in select_candidates() unit tests.
    # citations_top_k=1 with 2 ranked-above citations + 1 low-score result-intent citation.
    # All 3 should make it through the full traverse() call.
    graph = {
        "seed": {
            "references": [],
            "citations": [
                {"doi": "cite-high-1", "score": 0.9, "intent": None},
                {"doi": "cite-high-2", "score": 0.85, "intent": None},
                {"doi": "cite-result", "score": 0.3, "intent": "result"},  # below floor, but result-intent
            ],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    result = traverse(
        seed_doi="seed",
        fetch_fn=fake_fetch,
        max_depth=1,
        references_top_k=10,
        citations_top_k=1,  # normally would limit to 1, but result-intent bypasses
        relevance_floor=0.65,
    )

    dois = {node["doi"] for node in result}
    # seed + the top-1 ranked citation + the result-intent citation (bypassed)
    # NOT cite-high-2 (it's ranked 2nd and not result-intent)
    assert dois == {"seed", "cite-high-1", "cite-result"}


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
            # No DOI, arXiv-only preprint — falls back to ArXiv id rather than
            # being skipped, since most of this project's own citations are
            # exactly this shape (recent AI-safety arXiv preprints).
            {"title": "ArXiv-Only Paper", "abstract": "no DOI, arxiv only", "externalIds": {"ArXiv": "2503.03704"}},
            {"title": "No Identifier Paper", "abstract": "skip me", "externalIds": {}},  # neither — still skipped
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

    def fake_get(url, params=None, timeout=None, headers=None):
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
    assert ref_dois == {"10.1/ref-a", "2503.03704"}  # arXiv-only kept, no-identifier paper skipped

    cite_by_doi = {c["doi"]: c for c in result["citations"]}
    assert cite_by_doi["10.1/cite-a"]["intent"] == "result"
    assert cite_by_doi["10.1/cite-b"]["intent"] is None  # "background" is not "result"
    assert cite_by_doi["10.1/cite-a"]["score"] == pytest.approx(1.0)


# ── fetch_neighbors_by_search (unpublished/non-DOI seed papers) ─────────────

def test_fetch_neighbors_by_search_shapes_results_as_references_only(monkeypatch):
    s2_payload = {
        "data": [
            {"title": "A Related Paper", "abstract": "on the same topic", "externalIds": {"DOI": "10.1/found-a"}},
            {"title": "ArXiv-Only Result", "abstract": "no DOI, arxiv only", "externalIds": {"ArXiv": "2605.15338"}},
            {"title": "No Identifier Result", "abstract": "no id here", "externalIds": {}},  # skipped
        ]
    }

    def fake_get(url, params=None, timeout=None, headers=None):
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

    assert {c["doi"] for c in result["references"]} == {"10.1/found-a", "2605.15338"}
    assert result["citations"] == []  # an unindexed seed has no discoverable citations


def test_fetch_neighbors_from_s2_uses_arxiv_prefix_for_arxiv_shaped_seed(monkeypatch):
    captured_urls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        captured_urls.append(url)
        return _FakeResponse({"references": [], "citations": []})

    monkeypatch.setattr("requests.get", fake_get)

    def fake_embed_paper(text, specter2_fn=None, nomic_fn=None):
        return {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    fetch_neighbors_from_s2(
        doi="2503.03704",
        seed_embeddings={"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]},
        embed_fn=fake_embed_paper,
    )

    assert captured_urls == ["https://api.semanticscholar.org/graph/v1/paper/ARXIV:2503.03704"]


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

    def fake_get(url, params=None, timeout=None, headers=None):
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

    monkeypatch.setattr("requests.get", lambda url, params=None, timeout=None, headers=None: FakeResp())
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


# ── _check_huggingface_reachable (fail fast on a known-bad network) ────────

def test_check_huggingface_reachable_passes_silently_when_known_reachable(monkeypatch):
    monkeypatch.setattr(network_reachability, "known_status", lambda domain: True)
    monkeypatch.setattr(
        network_reachability, "check_and_record",
        lambda domain: pytest.fail("should not live-check when history already says reachable"),
    )
    _check_huggingface_reachable()  # should not raise


def test_check_huggingface_reachable_raises_when_known_blocked(monkeypatch):
    monkeypatch.setattr(network_reachability, "known_status", lambda domain: False)
    monkeypatch.setattr(network_reachability, "current_network_id", lambda: "work-net")
    with pytest.raises(RuntimeError, match="huggingface.co is unreachable"):
        _check_huggingface_reachable()


def test_check_huggingface_reachable_live_checks_when_no_history_yet(monkeypatch):
    monkeypatch.setattr(network_reachability, "known_status", lambda domain: None)
    monkeypatch.setattr(network_reachability, "check_and_record", lambda domain: True)
    _check_huggingface_reachable()  # should not raise


def test_check_huggingface_reachable_raises_when_live_check_fails(monkeypatch):
    monkeypatch.setattr(network_reachability, "known_status", lambda domain: None)
    monkeypatch.setattr(network_reachability, "check_and_record", lambda domain: False)
    monkeypatch.setattr(network_reachability, "current_network_id", lambda: "work-net")
    with pytest.raises(RuntimeError, match="huggingface.co is unreachable"):
        _check_huggingface_reachable()


def test_blended_score_rescues_cross_camp_rebuttal_specter2_alone_would_drop():
    # Rebuttal uses different terminology/style from seed (orthogonal in SPECTER2 space),
    # but same topic in nomic space. Blend rescues it; specter2 alone would drop it.
    seed = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}
    rebuttal = {"specter2": [0.0, 1.0], "nomic": [1.0, 0.0]}  # orthogonal SPECTER2, identical nomic
    specter2_only = _cosine_similarity(seed["specter2"], rebuttal["specter2"])
    nomic_only = _cosine_similarity(seed["nomic"], rebuttal["nomic"])
    assert specter2_only == pytest.approx(0.0)
    assert nomic_only == pytest.approx(1.0)
    blended = blended_score(seed, rebuttal, specter2_weight=0.5)
    assert blended == pytest.approx(0.5)
    assert blended > specter2_only


def test_shape_and_score_selects_cross_camp_rebuttal_that_specter2_alone_would_cap_out():
    # Verify that _shape_and_score uses blended_score (not specter2 alone) and thus
    # includes a high-nomic-similarity rebuttal that a specter2-only scorer would drop.
    seed_embeddings = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}
    papers = [
        {
            "title": "Rebuttal",
            "abstract": "disputes findings",
            "externalIds": {"DOI": "10.1/rebuttal"},
            "intents": ["result"],
        }
    ]

    def fake_embed(text):
        # Rebuttal: orthogonal in SPECTER2, identical in nomic
        return {"specter2": [0.0, 1.0], "nomic": [1.0, 0.0]}

    candidates = _shape_and_score(
        papers, seed_embeddings, embed_fn=fake_embed, is_citation=True
    )
    assert len(candidates) == 1
    assert candidates[0]["doi"] == "10.1/rebuttal"
    assert candidates[0]["intent"] == "result"
    assert candidates[0]["score"] == pytest.approx(0.5)  # blended score, not 0.0


# ── Full-text fetch for high-relevance nodes ─────────────────────────────────

def test_shape_and_score_preserves_arxiv_id_separate_from_doi():
    # AC1: arxiv_id must be distinct from doi for full-text fetch routing
    seed_embeddings = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}
    papers = [
        # DOI + ArXiv both present
        {
            "title": "Has Both",
            "abstract": "doi and arxiv",
            "externalIds": {"DOI": "10.1/example", "ArXiv": "2503.03704"},
        },
        # ArXiv-only (no DOI)
        {
            "title": "ArXiv Only",
            "abstract": "no doi preprint",
            "externalIds": {"ArXiv": "2605.15338"},
        },
    ]

    def fake_embed(text):
        return {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    candidates = _shape_and_score(papers, seed_embeddings, embed_fn=fake_embed, is_citation=False)

    assert len(candidates) == 2
    both_present = candidates[0]
    assert both_present["doi"] == "10.1/example"  # DOI picked as primary identifier
    assert both_present["arxiv_id"] == "2503.03704"  # ArXiv preserved separately

    arxiv_only = candidates[1]
    assert arxiv_only["doi"] == "2605.15338"  # ArXiv used as fallback for doi
    assert arxiv_only["arxiv_id"] == "2605.15338"  # ArXiv also available for fetch


def test_fetch_arxiv_pdf_text_constructs_correct_url_and_extracts(monkeypatch):
    # AC2: fetch_arxiv_pdf_text builds the right URL and uses ingest_paper.extract_pdf
    captured_urls = []

    class FakePdfResponse:
        status_code = 200

        def __init__(self):
            self.content = b"PDF binary content"

        def raise_for_status(self):
            pass

    def fake_get_with_retry(url, params, timeout, max_retries=8):
        captured_urls.append(url)
        return FakePdfResponse()

    monkeypatch.setattr("paper_graph._get_with_retry", fake_get_with_retry)

    # Mock ingest_paper.extract_pdf to return test text
    def fake_extract_pdf(path):
        return "extracted paper text from arXiv"

    # Patch the import inside fetch_arxiv_pdf_text
    import sys
    sys.modules["ingest_paper"] = type(sys)("ingest_paper")
    sys.modules["ingest_paper"].extract_pdf = fake_extract_pdf

    result = fetch_arxiv_pdf_text("2503.03704")

    assert result == "extracted paper text from arXiv"
    assert "https://arxiv.org/pdf/2503.03704" in captured_urls


def test_fetch_arxiv_pdf_text_returns_none_on_404(monkeypatch):
    # AC3: graceful degradation when PDF not found
    class FakeResponse:
        status_code = 404

        def raise_for_status(self):
            pass

    def fake_get_with_retry(url, params, timeout, max_retries=8):
        return FakeResponse()

    monkeypatch.setattr("paper_graph._get_with_retry", fake_get_with_retry)

    result = fetch_arxiv_pdf_text("2503.03704")

    assert result is None


def test_fetch_arxiv_pdf_text_returns_none_on_network_error(monkeypatch):
    # AC3: graceful degradation on network failure
    def fake_get_with_retry(url, params, timeout, max_retries=8):
        raise Exception("Network error")

    monkeypatch.setattr("paper_graph._get_with_retry", fake_get_with_retry)

    result = fetch_arxiv_pdf_text("2503.03704")

    assert result is None


def test_fetch_full_text_returns_none_when_no_arxiv_id():
    # AC3: no Sci-Hub fallback — DOI-only papers stay abstract-only
    node = {"doi": "10.1/example", "arxiv_id": None, "score": 0.9}

    result = fetch_full_text(node)

    assert result is None


def test_record_paper_stamps_full_text_metadata(tmp_path):
    # AC1: record_paper must store full_text flag in metadata
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge-test")

    record_paper(
        collection,
        doi="10.1/with-full-text",
        abstract="full text content here",
        discovered_via=None,
        full_text=True,
    )

    meta = collection.get(ids=["10.1/with-full-text"])["metadatas"][0]
    assert meta["full_text"] is True

    # Also test default (abstract-only)
    record_paper(
        collection,
        doi="10.1/abstract-only",
        abstract="abstract content",
        discovered_via=None,
        full_text=False,
    )

    meta2 = collection.get(ids=["10.1/abstract-only"])["metadatas"][0]
    assert meta2["full_text"] is False


def test_run_paper_graph_fetches_full_text_for_high_relevance_nodes_with_arxiv_id(
    tmp_path, monkeypatch
):
    # AC1 & AC2: high-relevance node with arXiv ID → full text fetched and stored
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    graph = {
        "seed": {
            "references": [
                {
                    "doi": "ref-high-arxiv",
                    "score": 0.88,
                    "intent": None,
                    "abstract": "abstract text",
                    "arxiv_id": "2503.03704",
                }
            ],
            "citations": [],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    # Mock full-text fetch to return extracted text
    def fake_fetch_full_text(node):
        if node.get("arxiv_id") == "2503.03704":
            return "This is the extracted full text from the PDF."
        return None

    monkeypatch.setattr("paper_graph.fetch_full_text", fake_fetch_full_text)

    results = run_paper_graph(
        seed_doi="seed",
        seed_abstract="seed abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        full_text_threshold=0.85,  # ref-high-arxiv at 0.88 exceeds this
        fetch_fn=fake_fetch,
    )

    # Verify the full text was stored
    stored = collection.get(ids=["ref-high-arxiv"])
    assert len(stored["ids"]) == 1
    assert stored["documents"][0] == "This is the extracted full text from the PDF."
    assert stored["metadatas"][0]["full_text"] is True


def test_run_paper_graph_stores_abstract_for_high_relevance_node_without_arxiv_id(
    tmp_path, monkeypatch
):
    # AC1 & AC3: high-relevance DOI-only node → abstract stored, full_text=False, fetch not called
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    graph = {
        "seed": {
            "references": [
                {
                    "doi": "ref-doi-only",
                    "score": 0.88,
                    "intent": None,
                    "abstract": "abstract of doi-only paper",
                    "arxiv_id": None,  # no ArXiv ID available
                }
            ],
            "citations": [],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    fetch_calls = []

    def spy_fetch_full_text(node):
        fetch_calls.append(node["doi"])
        return None

    monkeypatch.setattr("paper_graph.fetch_full_text", spy_fetch_full_text)

    results = run_paper_graph(
        seed_doi="seed",
        seed_abstract="seed abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        full_text_threshold=0.85,
        fetch_fn=fake_fetch,
    )

    # Verify the abstract was stored without attempting a fetch
    stored = collection.get(ids=["ref-doi-only"])
    assert stored["documents"][0] == "abstract of doi-only paper"
    assert stored["metadatas"][0]["full_text"] is False
    assert "ref-doi-only" not in fetch_calls  # fetch never called


def test_run_paper_graph_never_fetches_for_below_threshold_nodes(tmp_path, monkeypatch):
    # AC3: nodes below full_text_threshold must never trigger a fetch, even with arXiv ID
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    graph = {
        "seed": {
            "references": [
                {
                    "doi": "ref-below-threshold",
                    "score": 0.75,  # below 0.85 threshold
                    "intent": None,
                    "abstract": "abstract of below-threshold paper",
                    "arxiv_id": "2503.03704",  # has ArXiv ID but shouldn't be fetched
                }
            ],
            "citations": [],
        },
    }

    def fake_fetch(doi):
        return graph[doi]

    fetch_calls = []

    def spy_fetch_full_text(node):
        fetch_calls.append(node["doi"])
        return "full text"

    monkeypatch.setattr("paper_graph.fetch_full_text", spy_fetch_full_text)

    results = run_paper_graph(
        seed_doi="seed",
        seed_abstract="seed abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        full_text_threshold=0.85,  # threshold is 0.85, node score is 0.75
        fetch_fn=fake_fetch,
    )

    # Verify fetch was NEVER called for below-threshold node
    assert "ref-below-threshold" not in fetch_calls
    # Verify abstract stored with full_text=False
    stored = collection.get(ids=["ref-below-threshold"])
    assert stored["documents"][0] == "abstract of below-threshold paper"
    assert stored["metadatas"][0]["full_text"] is False


# ── AC2: skip already-known papers (embedding + fetch) ───────────────────────

def test_shape_and_score_skips_already_known_candidates_without_embedding():
    # AC2: Known DOIs should skip embedding entirely, not only skip re-recording.
    seed_embeddings = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}
    papers = [
        {"title": "Known", "abstract": "already in collection", "externalIds": {"DOI": "10.1/known"}},
        {"title": "Unknown", "abstract": "new paper", "externalIds": {"DOI": "10.1/unknown"}},
    ]

    embed_calls = []

    def spy_embed(text):
        embed_calls.append(text)
        return {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    def fake_is_known(doi):
        return doi == "10.1/known"

    candidates = _shape_and_score(
        papers,
        seed_embeddings,
        embed_fn=spy_embed,
        is_citation=False,
        is_known_fn=fake_is_known,
    )

    # only unknown paper should be embedded and returned
    assert len(candidates) == 1
    assert candidates[0]["doi"] == "10.1/unknown"
    assert len(embed_calls) == 1
    assert embed_calls[0] == "new paper"


def test_fetch_neighbors_from_s2_filters_out_known_candidates(monkeypatch):
    # AC2: fetch_neighbors_from_s2 should skip embedding and exclude known DOIs from results.
    s2_payload = {
        "references": [
            {"title": "Known Ref", "abstract": "already known", "externalIds": {"DOI": "10.1/ref-known"}},
            {"title": "New Ref", "abstract": "new reference", "externalIds": {"DOI": "10.1/ref-new"}},
        ],
        "citations": [
            {"title": "Known Cite", "abstract": "already known", "externalIds": {"DOI": "10.1/cite-known"}, "intents": []},
            {"title": "New Cite", "abstract": "new citation", "externalIds": {"DOI": "10.1/cite-new"}, "intents": []},
        ],
    }

    def fake_get(url, params=None, timeout=None, headers=None):
        return _FakeResponse(s2_payload)

    monkeypatch.setattr("requests.get", fake_get)

    seed_embeddings = {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    embed_calls = []

    def spy_embed(text, specter2_fn=None, nomic_fn=None):
        embed_calls.append(text)
        return {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    def fake_is_known(doi):
        return doi.endswith("-known")

    result = fetch_neighbors_from_s2(
        doi="10.1/seed",
        seed_embeddings=seed_embeddings,
        embed_fn=spy_embed,
        is_known_fn=fake_is_known,
    )

    # only new papers should be in results and embedded
    ref_dois = {c["doi"] for c in result["references"]}
    cite_dois = {c["doi"] for c in result["citations"]}
    assert ref_dois == {"10.1/ref-new"}
    assert cite_dois == {"10.1/cite-new"}
    assert len(embed_calls) == 2
    assert "new reference" in embed_calls
    assert "new citation" in embed_calls


def test_run_paper_graph_skips_embedding_already_known_candidates_end_to_end(monkeypatch, tmp_path):
    # AC2 end-to-end: pre-seed the collection with a candidate DOI, verify it's
    # never embedded and never explored in the traversal.
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    # Pre-record a candidate DOI so it's "known"
    record_paper(
        collection,
        doi="10.1/known-candidate",
        abstract="this was found before",
        discovered_via=None,
    )

    embed_calls = []

    def spy_embed(text, specter2_fn=None, nomic_fn=None):
        embed_calls.append(text)
        return {"specter2": [1.0, 0.0], "nomic": [1.0, 0.0]}

    # Mock S2 API to return our test data
    s2_payload = {
        "references": [
            {"title": "Known", "abstract": "known abstract", "externalIds": {"DOI": "10.1/known-candidate"}},
            {"title": "New", "abstract": "new abstract", "externalIds": {"DOI": "10.1/new-candidate"}},
        ],
        "citations": [],
    }

    def fake_get(url, params=None, timeout=None, headers=None):
        return _FakeResponse(s2_payload)

    monkeypatch.setattr("requests.get", fake_get)

    results = run_paper_graph(
        seed_doi="10.1/seed",
        seed_abstract="the seed paper's abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        embed_fn=spy_embed,
    )

    # AC2: known-candidate should be filtered out during traversal entirely (never added to results,
    # never embedded, never queued for further exploration). Only seed and new-candidate should be present.
    result_dois = {n["doi"] for n in results}
    assert result_dois == {"10.1/seed", "10.1/new-candidate"}

    # only seed and new-candidate should be embedded; known-candidate never touched
    embedded_texts = embed_calls
    assert "the seed paper's abstract" in embedded_texts
    assert "new abstract" in embedded_texts
    assert "known abstract" not in embedded_texts  # never embedded


# ── get_discovery (queryable first-discovery metadata) ────────────────────────

def test_get_discovery_returns_parent_edge_and_hop_for_recorded_paper(tmp_path):
    # AC2: get_discovery should return the first-discovery record for a non-seed paper.
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    record_paper(
        collection,
        doi="10.1/paper",
        abstract="some abstract",
        discovered_via={"parent_doi": "10.1/seed", "edge_type": "reference", "hop_depth": 1},
    )

    discovery = get_discovery(collection, "10.1/paper")
    assert discovery is not None
    assert discovery["parent_doi"] == "10.1/seed"
    assert discovery["edge_type"] == "reference"
    assert discovery["hop_depth"] == 1


def test_get_discovery_returns_none_for_seed_paper(tmp_path):
    # AC2: a seed paper (discovered_via=None) should return None from get_discovery.
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    record_paper(
        collection,
        doi="10.1/seed",
        abstract="seed abstract",
        discovered_via=None,
    )

    discovery = get_discovery(collection, "10.1/seed")
    assert discovery is None


def test_get_discovery_returns_none_for_unknown_doi(tmp_path):
    # AC2: an unknown DOI should return None.
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    discovery = get_discovery(collection, "10.1/unknown")
    assert discovery is None


def test_run_paper_graph_preserves_first_discovery_only_on_rediscovery(tmp_path):
    # ADR 0009 regression: "first discovery wins" — if a paper is discovered via one parent/edge
    # in the first run and then re-encountered via a different parent/edge in a second run, the
    # original first-discovery metadata should be preserved.
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("paper-knowledge")

    # First traversal: find "10.1/child" via "10.1/seed1" as a reference
    graph1 = {
        "10.1/seed1": {
            "references": [{"doi": "10.1/child", "score": 0.9, "intent": None, "abstract": "child abstract"}],
            "citations": [],
        },
    }

    def fake_fetch1(doi):
        return graph1[doi]

    results1 = run_paper_graph(
        seed_doi="10.1/seed1",
        seed_abstract="seed1 abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        fetch_fn=fake_fetch1,
    )

    # Verify first discovery recorded
    first_discovery = get_discovery(collection, "10.1/child")
    assert first_discovery is not None
    assert first_discovery["parent_doi"] == "10.1/seed1"
    assert first_discovery["edge_type"] == "reference"

    # Second traversal: if we somehow discover "10.1/child" again via a different parent,
    # the first-discovery metadata should not be overwritten (it's already known)
    # (In reality, is_known() filters it out before re-recording, but let's verify the
    # behavior explicitly by checking that a second run doesn't change the metadata)
    graph2 = {
        "10.1/seed2": {
            "references": [{"doi": "10.1/child", "score": 0.85, "intent": None, "abstract": "child abstract"}],
            "citations": [],
        },
    }

    def fake_fetch2(doi):
        return graph2[doi]

    results2 = run_paper_graph(
        seed_doi="10.1/seed2",
        seed_abstract="seed2 abstract",
        collection=collection,
        max_depth=1,
        references_top_k=10,
        citations_top_k=5,
        relevance_floor=0.5,
        fetch_fn=fake_fetch2,
    )

    # Verify first-discovery metadata was NOT changed (still points to original parent/edge)
    still_first_discovery = get_discovery(collection, "10.1/child")
    assert still_first_discovery is not None
    assert still_first_discovery["parent_doi"] == "10.1/seed1"  # unchanged
    assert still_first_discovery["edge_type"] == "reference"  # unchanged
