"""Tests for competing_ideas.py. Run via:
    ~/.agents/venv/bin/python -m pytest scripts/tests/test_competing_ideas.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import chromadb
import pytest

from competing_ideas import (
    semantic_search,
    classify_stance,
    _parse_stance_result,
    build_competing_ideas_map,
    render_markdown,
)


# ── _parse_stance_result ───────────────────────────────────────────────────

def test_parse_stance_result_extracts_stance_and_rationale():
    response = """STANCE: supports
RATIONALE: The paper provides empirical evidence for the hypothesis."""
    result = _parse_stance_result(response)
    assert result["stance"] == "supports"
    assert "empirical" in result["rationale"]


def test_parse_stance_result_recognizes_all_stances():
    for stance in ["supports", "refutes", "mixed", "unrelated"]:
        response = f"STANCE: {stance}\nRATIONALE: test rationale"
        result = _parse_stance_result(response)
        assert result["stance"] == stance


def test_parse_stance_result_is_case_insensitive():
    response = """stance: SUPPORTS
Rationale: Empirical evidence."""
    result = _parse_stance_result(response)
    assert result["stance"] == "supports"


def test_parse_stance_result_returns_unknown_on_unrecognized_stance():
    response = """STANCE: maybe_supports
RATIONALE: Hard to classify."""
    result = _parse_stance_result(response)
    assert result["stance"] == "unknown"


# ── classify_stance ────────────────────────────────────────────────────────

def test_classify_stance_returns_stance_and_rationale():
    def fake_chat(messages):
        return "STANCE: supports\nRATIONALE: The paper validates the hypothesis with new data."

    result = classify_stance(
        "Models benefit from larger training datasets.",
        "Scaling Laws for Neural Language Models",
        "We study how performance scales with data size...",
        chat_fn=fake_chat,
    )
    assert result["stance"] == "supports"
    assert "validates" in result["rationale"]


def test_classify_stance_includes_hypothesis_title_abstract_in_prompt():
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "STANCE: supports\nRATIONALE: test"

    classify_stance(
        "My Hypothesis",
        "Paper Title",
        "Paper abstract text",
        chat_fn=fake_chat,
    )
    assert "My Hypothesis" in captured["prompt"]
    assert "Paper Title" in captured["prompt"]
    assert "Paper abstract text" in captured["prompt"]


# ── semantic_search ────────────────────────────────────────────────────────

def test_semantic_search_returns_papers_from_query():
    client = chromadb.PersistentClient(path=str(Path.home() / ".test_chroma_competing"))
    col = client.get_or_create_collection("test-paper-knowledge")
    col.add(
        ids=["paper-1", "paper-2"],
        documents=["Document about scaling and training data.", "Unrelated document."],
        metadatas=[
            {"doi": "paper-1", "title": "Scaling Laws"},
            {"doi": "paper-2", "title": "Other Topic"},
        ],
    )

    def fake_search(query_texts, n_results):
        # Mock: return papers that match
        return {
            "ids": [["paper-1", "paper-2"]],
            "documents": [["Document about scaling and training data.", "Unrelated document."]],
        }

    papers = semantic_search(col, "model scaling", search_fn=fake_search)
    assert len(papers) == 2
    assert papers[0]["doi"] == "paper-1"
    assert papers[0]["title"] == "Scaling Laws"


def test_semantic_search_retrieves_metadata():
    client = chromadb.PersistentClient(path=str(Path.home() / ".test_chroma_competing2"))
    col = client.get_or_create_collection("test-paper-knowledge2")
    col.add(
        ids=["paper-1"],
        documents=["Document about scaling."],
        metadatas=[{"doi": "paper-1", "title": "A Scaling Paper"}],
    )

    def fake_search(query_texts, n_results):
        return {"ids": [["paper-1"]], "documents": [["Document about scaling."]]}

    papers = semantic_search(col, "scaling", n_results=1, search_fn=fake_search)
    assert papers[0]["title"] == "A Scaling Paper"


def test_semantic_search_handles_missing_title_metadata():
    client = chromadb.PersistentClient(path=str(Path.home() / ".test_chroma_competing3"))
    col = client.get_or_create_collection("test-paper-knowledge3")
    col.add(
        ids=["paper-1"],
        documents=["Some document."],
        metadatas=[{"doi": "paper-1"}],  # No title
    )

    def fake_search(query_texts, n_results):
        return {"ids": [["paper-1"]], "documents": [["Some document."]]}

    papers = semantic_search(col, "something", search_fn=fake_search)
    # Should fall back to DOI if title is missing
    assert papers[0]["title"] == "paper-1"


# ── build_competing_ideas_map ──────────────────────────────────────────────

def test_build_competing_ideas_map_groups_by_stance(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection("paper-knowledge")
    col.add(
        ids=["support-1", "refute-1", "mixed-1", "unrelated-1"],
        documents=["supporting", "refuting", "mixed", "unrelated"],
        metadatas=[
            {"doi": "support-1", "title": "Supporting Paper"},
            {"doi": "refute-1", "title": "Refuting Paper"},
            {"doi": "mixed-1", "title": "Mixed Paper"},
            {"doi": "unrelated-1", "title": "Unrelated Paper"},
        ],
    )

    responses = iter([
        "STANCE: supports\nRATIONALE: Provides evidence.",
        "STANCE: refutes\nRATIONALE: Contradicts claim.",
        "STANCE: mixed\nRATIONALE: Both sides.",
        "STANCE: unrelated\nRATIONALE: Different topic.",
    ])

    def fake_search(query_texts, n_results):
        return {
            "ids": [["support-1", "refute-1", "mixed-1", "unrelated-1"]],
            "documents": [["supporting", "refuting", "mixed", "unrelated"]],
        }

    def fake_chat(messages):
        return next(responses)

    result = build_competing_ideas_map(
        col, "test hypothesis", search_fn=fake_search, chat_fn=fake_chat
    )

    assert len(result["supports"]) == 1
    assert len(result["refutes"]) == 1
    assert len(result["mixed"]) == 1
    # unrelated should be excluded
    assert "unrelated" not in result


def test_build_competing_ideas_map_detects_disagreement():
    client = chromadb.PersistentClient(path=str(Path.home() / ".test_chroma_competing4"))
    col = client.get_or_create_collection("test-paper-knowledge4")
    col.add(
        ids=["support-1", "refute-1"],
        documents=["supporting doc", "refuting doc"],
        metadatas=[
            {"doi": "support-1", "title": "Support"},
            {"doi": "refute-1", "title": "Refute"},
        ],
    )

    responses = iter([
        "STANCE: supports\nRATIONALE: Evidence for.",
        "STANCE: refutes\nRATIONALE: Evidence against.",
    ])

    def fake_search(query_texts, n_results):
        return {
            "ids": [["support-1", "refute-1"]],
            "documents": [["supporting doc", "refuting doc"]],
        }

    def fake_chat(messages):
        return next(responses)

    result = build_competing_ideas_map(
        col, "hypothesis", search_fn=fake_search, chat_fn=fake_chat
    )

    # Both sides present = disagreement detected
    assert len(result["supports"]) > 0
    assert len(result["refutes"]) > 0


def test_build_competing_ideas_map_handles_consensus():
    client = chromadb.PersistentClient(path=str(Path.home() / ".test_chroma_competing5"))
    col = client.get_or_create_collection("test-paper-knowledge5")
    col.add(
        ids=["support-1", "support-2"],
        documents=["supporting", "also supporting"],
        metadatas=[
            {"doi": "support-1", "title": "Support 1"},
            {"doi": "support-2", "title": "Support 2"},
        ],
    )

    responses = iter([
        "STANCE: supports\nRATIONALE: Evidence.",
        "STANCE: supports\nRATIONALE: Also evidence.",
    ])

    def fake_search(query_texts, n_results):
        return {
            "ids": [["support-1", "support-2"]],
            "documents": [["supporting", "also supporting"]],
        }

    def fake_chat(messages):
        return next(responses)

    result = build_competing_ideas_map(
        col, "hypothesis", search_fn=fake_search, chat_fn=fake_chat
    )

    # All support, no refutes = consensus (empty-conflict is valid)
    assert len(result["supports"]) == 2
    assert not result["refutes"]


# ── render_markdown ────────────────────────────────────────────────────────

def test_render_markdown_includes_hypothesis():
    hypothesis = "Large models generalize better than small ones."
    map_result = {"supports": [], "refutes": [], "mixed": []}
    md = render_markdown(hypothesis, map_result)
    assert hypothesis in md


def test_render_markdown_shows_papers_by_stance():
    map_result = {
        "supports": [
            {
                "doi": "paper-1",
                "title": "Scaling Improves Generalization",
                "abstract": "Evidence that...",
                "rationale": "Provides scaling evidence.",
            }
        ],
        "refutes": [
            {
                "doi": "paper-2",
                "title": "Small Models Generalize Better",
                "abstract": "Shows that...",
                "rationale": "Contradicts with empirical data.",
            }
        ],
        "mixed": [],
    }
    md = render_markdown("hypothesis", map_result)

    assert "Supporting Evidence" in md
    assert "Scaling Improves Generalization" in md
    assert "Refuting Evidence" in md
    assert "Small Models Generalize Better" in md
    assert "⚠️" in md  # Refuting papers are flagged


def test_render_markdown_counts_papers_by_stance():
    map_result = {
        "supports": [
            {"doi": "p1", "title": "Paper 1", "abstract": "abstract", "rationale": "reason"},
            {"doi": "p2", "title": "Paper 2", "abstract": "abstract", "rationale": "reason"},
            {"doi": "p3", "title": "Paper 3", "abstract": "abstract", "rationale": "reason"},
        ],
        "refutes": [
            {"doi": "p4", "title": "Paper 4", "abstract": "abstract", "rationale": "reason"},
            {"doi": "p5", "title": "Paper 5", "abstract": "abstract", "rationale": "reason"},
        ],
        "mixed": [
            {"doi": "p6", "title": "Paper 6", "abstract": "abstract", "rationale": "reason"},
        ],
    }
    md = render_markdown("hypothesis", map_result)
    assert "(3 papers)" in md  # supports
    assert "(2 papers)" in md  # refutes
    assert "(1 paper)" in md   # mixed
