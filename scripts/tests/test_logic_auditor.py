"""Tests for logic_auditor.py. Run via:
    ~/.agents/venv/bin/python -m pytest scripts/tests/test_logic_auditor.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import pytest

from logic_auditor import (
    classify_paper_type,
    classify_all,
    PAPER_TYPES,
    _parse_fields,
    extract_structure,
    extract_all,
    EXTRACTION_FIELDS,
    _parse_findings,
    judge_extraction,
    judge_all,
    check_inference_validity,
    check_inference_validity_all,
    INFERENCE_FIELDS,
    RELIABILITY_LEVELS,
    compute_reliability_signal,
    compute_all_reliability_signals,
    needs_second_look,
    build_audit_report,
    render_markdown,
)


# ── classify_paper_type ──────────────────────────────────────────────────────

def test_classify_paper_type_recognizes_empirical():
    def fake_chat(messages):
        return "empirical"

    assert classify_paper_type("A Title", "An abstract", chat_fn=fake_chat) == "empirical"


def test_classify_paper_type_recognizes_survey():
    def fake_chat(messages):
        return "survey"

    assert classify_paper_type("A Title", "An abstract", chat_fn=fake_chat) == "survey"


def test_classify_paper_type_recognizes_benchmark():
    def fake_chat(messages):
        return "benchmark"

    assert classify_paper_type("A Title", "An abstract", chat_fn=fake_chat) == "benchmark"


def test_classify_paper_type_recognizes_conceptual():
    def fake_chat(messages):
        return "conceptual"

    assert classify_paper_type("A Title", "An abstract", chat_fn=fake_chat) == "conceptual"


def test_classify_paper_type_is_case_insensitive_and_strips_whitespace():
    def fake_chat(messages):
        return "  Empirical  \n"

    assert classify_paper_type("A Title", "An abstract", chat_fn=fake_chat) == "empirical"


def test_classify_paper_type_extracts_label_from_extra_prose():
    # models sometimes ignore "respond with only the label" and explain themselves anyway
    def fake_chat(messages):
        return "This paper is best classified as a survey, since it synthesizes existing work."

    assert classify_paper_type("A Title", "An abstract", chat_fn=fake_chat) == "survey"


def test_classify_paper_type_falls_back_to_unknown_on_unparseable_response():
    def fake_chat(messages):
        return "I'm not sure, this could be several things at once"

    assert classify_paper_type("A Title", "An abstract", chat_fn=fake_chat) == "unknown"


def test_classify_paper_type_includes_title_and_abstract_in_prompt():
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "empirical"

    classify_paper_type("My Title", "My abstract text", chat_fn=fake_chat)
    assert "My Title" in captured["prompt"]
    assert "My abstract text" in captured["prompt"]


def test_paper_types_is_the_four_documented_types():
    assert PAPER_TYPES == {"empirical", "survey", "benchmark", "conceptual"}


# ── classify_all ─────────────────────────────────────────────────────────────

def test_classify_all_classifies_every_paper_in_the_input_dict():
    responses = iter(["empirical", "survey", "conceptual"])

    def fake_chat(messages):
        return next(responses)

    papers = {
        "seed-a": ("Title A", "Abstract A"),
        "seed-b": ("Title B", "Abstract B"),
        "seed-c": ("Title C", "Abstract C"),
    }
    result = classify_all(papers, chat_fn=fake_chat)

    assert result == {"seed-a": "empirical", "seed-b": "survey", "seed-c": "conceptual"}


def test_classify_all_handles_empty_input():
    assert classify_all({}, chat_fn=lambda messages: "empirical") == {}


# ── _parse_fields (shared line-based parser for all extraction types) ──────

def test_parse_fields_extracts_single_line_values():
    response = "CLAIM: models degrade in the middle\nGROUNDS: two tasks measured\n"
    result = _parse_fields(response, ["CLAIM", "GROUNDS"])
    assert result == {"claim": "models degrade in the middle", "grounds": "two tasks measured"}


def test_parse_fields_accumulates_multiline_values_until_next_field():
    response = (
        "CLAIM: models degrade\nin the middle of long contexts\n"
        "GROUNDS: two tasks were measured\n"
    )
    result = _parse_fields(response, ["CLAIM", "GROUNDS"])
    assert result["claim"] == "models degrade\nin the middle of long contexts"
    assert result["grounds"] == "two tasks were measured"


def test_parse_fields_is_case_insensitive_on_labels():
    response = "claim: something\nGrounds: something else\n"
    result = _parse_fields(response, ["CLAIM", "GROUNDS"])
    assert result == {"claim": "something", "grounds": "something else"}


def test_parse_fields_missing_field_is_absent_not_empty_string():
    response = "CLAIM: something\n"
    result = _parse_fields(response, ["CLAIM", "GROUNDS"])
    assert result == {"claim": "something"}
    assert "grounds" not in result


# ── extract_structure (routes to type-appropriate extraction) ──────────────

def test_extract_structure_empirical_uses_toulmin_fields():
    def fake_chat(messages):
        return "CLAIM: c\nGROUNDS: g\nWARRANT: w\nQUALIFIER: q\n"

    result = extract_structure("Title", "Abstract", "empirical", chat_fn=fake_chat)
    assert result == {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"}


def test_extract_structure_survey_uses_taxonomy_fields():
    def fake_chat(messages):
        return "TAXONOMY: t\nCOVERAGE: c\nCLAIMED_GAPS: g\n"

    result = extract_structure("Title", "Abstract", "survey", chat_fn=fake_chat)
    assert result == {"taxonomy": "t", "coverage": "c", "claimed_gaps": "g"}


def test_extract_structure_benchmark_uses_construct_validity_fields():
    def fake_chat(messages):
        return "MEASURES: m\nCONSTRUCT_VALIDITY_EVIDENCE: e\nSCOPE: s\n"

    result = extract_structure("Title", "Abstract", "benchmark", chat_fn=fake_chat)
    assert result == {"measures": "m", "construct_validity_evidence": "e", "scope": "s"}


def test_extract_structure_conceptual_uses_structural_claim_fields():
    def fake_chat(messages):
        return "STRUCTURAL_CLAIMS: s\nKEY_DEFINITIONS: d\nINTERNAL_DEPENDENCIES: i\n"

    result = extract_structure("Title", "Abstract", "conceptual", chat_fn=fake_chat)
    assert result == {"structural_claims": "s", "key_definitions": "d", "internal_dependencies": "i"}


def test_extract_structure_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown"):
        extract_structure("Title", "Abstract", "unknown", chat_fn=lambda m: "")


def test_extract_structure_includes_title_and_abstract_in_prompt():
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "CLAIM: c\nGROUNDS: g\nWARRANT: w\nQUALIFIER: q\n"

    extract_structure("My Title", "My abstract", "empirical", chat_fn=fake_chat)
    assert "My Title" in captured["prompt"]
    assert "My abstract" in captured["prompt"]


def test_extraction_fields_covers_all_four_types():
    assert set(EXTRACTION_FIELDS) == PAPER_TYPES


# ── extract_all ──────────────────────────────────────────────────────────────

def test_extract_all_routes_each_paper_by_its_own_type():
    def fake_chat(messages):
        prompt = messages[0]["content"]
        if "GROUNDS" in prompt:
            return "CLAIM: c\nGROUNDS: g\nWARRANT: w\nQUALIFIER: q\n"
        return "TAXONOMY: t\nCOVERAGE: c\nCLAIMED_GAPS: g\n"

    papers = {
        "seed-a": ("Title A", "Abstract A", "empirical"),
        "seed-b": ("Title B", "Abstract B", "survey"),
    }
    result = extract_all(papers, chat_fn=fake_chat)

    assert result["seed-a"]["claim"] == "c"
    assert result["seed-b"]["taxonomy"] == "t"


def test_extract_all_handles_empty_input():
    assert extract_all({}, chat_fn=lambda m: "") == {}


# ── _parse_findings ──────────────────────────────────────────────────────────

def test_parse_findings_extracts_each_finding_line():
    response = "FINDING: the qualifier overclaims relative to a single-task result\nFINDING: possible cherry-picking in the examples cited\n"
    assert _parse_findings(response) == [
        "the qualifier overclaims relative to a single-task result",
        "possible cherry-picking in the examples cited",
    ]


def test_parse_findings_returns_empty_list_for_none_response():
    assert _parse_findings("NONE") == []
    assert _parse_findings("  none  \n") == []


def test_parse_findings_returns_empty_list_for_blank_response():
    assert _parse_findings("") == []
    assert _parse_findings("   \n  ") == []


def test_parse_findings_ignores_preamble_before_the_first_marker():
    response = "Here is my analysis:\nFINDING: a real issue\n"
    assert _parse_findings(response) == ["a real issue"]


def test_parse_findings_splits_multiple_findings_crammed_onto_one_line():
    # real qwen2.5:3b behavior, seen live 2026-07-13: doesn't always put one
    # FINDING per line -- sometimes runs several together separated only by
    # the next "FINDING:" marker, not a newline.
    response = "FINDING: the warrant has an unaddressed confound. FINDING: no hasty generalization. FINDING: no false dichotomy."
    assert _parse_findings(response) == [
        "the warrant has an unaddressed confound.",
        "no hasty generalization.",
        "no false dichotomy.",
    ]


def test_parse_findings_filters_out_stray_none_captured_as_a_finding():
    # real qwen2.5:3b behavior, seen live: occasionally emits "FINDING: NONE"
    # as an extra trailing item instead of using the bare NONE response.
    response = "FINDING: a real issue\nFINDING: NONE\n"
    assert _parse_findings(response) == ["a real issue"]


def test_parse_findings_strips_a_stray_trailing_none_glued_onto_a_real_finding():
    # real qwen2.5:14b behavior, seen live on sorry-bench 2026-07-13: appends
    # a trailing "NONE" line onto the end of a real finding with no second
    # "FINDING:" marker to separate them, so the whole thing gets captured
    # as one piece -- the WHOLE-piece "is it exactly NONE" check above
    # doesn't catch this since the piece isn't only "NONE".
    response = "FINDING: a real issue with the evidence.\n\nNONE"
    assert _parse_findings(response) == ["a real issue with the evidence."]


# ── judge_extraction (type-adaptive consistency judgment) ──────────────────

def test_judge_extraction_empirical_returns_findings_list():
    def fake_chat(messages):
        return "FINDING: qualifier overclaims relative to the grounds\n"

    extraction = {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"}
    result = judge_extraction(extraction, "empirical", chat_fn=fake_chat)
    assert result == ["qualifier overclaims relative to the grounds"]


def test_judge_extraction_survey_returns_findings_list():
    def fake_chat(messages):
        return "FINDING: claimed gaps aren't supported by the stated coverage\n"

    extraction = {"taxonomy": "t", "coverage": "c", "claimed_gaps": "g"}
    result = judge_extraction(extraction, "survey", chat_fn=fake_chat)
    assert result == ["claimed gaps aren't supported by the stated coverage"]


def test_judge_extraction_benchmark_returns_findings_list():
    def fake_chat(messages):
        return "FINDING: construct validity is assumed, not evidenced\n"

    extraction = {"measures": "m", "construct_validity_evidence": "e", "scope": "s"}
    result = judge_extraction(extraction, "benchmark", chat_fn=fake_chat)
    assert result == ["construct validity is assumed, not evidenced"]


def test_judge_extraction_benchmark_prompt_guards_against_dismissing_real_comparisons():
    # regression test for the strongreject bug found in task 19: the
    # judgment step called real comparison-based evidence ("agreement with
    # human judgments") "merely asserted", contradicting its own input. Two
    # rounds of prompt wording were tried before qwen2.5:7b reliably obeyed
    # it -- this checks the final, forceful wording is present.
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "NONE"

    extraction = {"measures": "m", "construct_validity_evidence": "e", "scope": "s"}
    judge_extraction(extraction, "benchmark", chat_fn=fake_chat)
    assert "DO NOT flag construct validity as a" in captured["prompt"]
    assert "human judgment" in captured["prompt"]


def test_judge_extraction_uses_14b_override_for_benchmark_when_no_chat_fn_given(monkeypatch):
    # 7b (the default JUDGMENT_MODEL) kept failing the strongreject case
    # even with forceful prompting -- 14b fixed it with zero regression on
    # the other 2 benchmarks. Only "benchmark" gets the override; the other
    # 3 types have no evidence of a problem at 7b.
    import logic_auditor as la

    captured = {}

    def fake_ollama_chat(model, messages, timeout=60):
        captured["model"] = model
        return "NONE"

    monkeypatch.setattr(la, "ollama_chat", fake_ollama_chat)

    extraction = {"measures": "m", "construct_validity_evidence": "e", "scope": "s"}
    judge_extraction(extraction, "benchmark")  # no chat_fn -- exercises the real default path
    assert captured["model"] == "qwen2.5:14b"


def test_judge_extraction_uses_default_7b_for_non_benchmark_types(monkeypatch):
    import logic_auditor as la

    captured = {}

    def fake_ollama_chat(model, messages, timeout=60):
        captured["model"] = model
        return "NONE"

    monkeypatch.setattr(la, "ollama_chat", fake_ollama_chat)

    extraction = {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"}
    judge_extraction(extraction, "empirical")
    assert captured["model"] == "qwen2.5:7b"


def test_judge_extraction_conceptual_returns_findings_list():
    def fake_chat(messages):
        return "FINDING: two structural claims contradict each other\n"

    extraction = {"structural_claims": "s", "key_definitions": "d", "internal_dependencies": "i"}
    result = judge_extraction(extraction, "conceptual", chat_fn=fake_chat)
    assert result == ["two structural claims contradict each other"]


def test_judge_extraction_returns_empty_list_when_no_issues_found():
    def fake_chat(messages):
        return "NONE"

    extraction = {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"}
    assert judge_extraction(extraction, "empirical", chat_fn=fake_chat) == []


def test_judge_extraction_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown"):
        judge_extraction({}, "unknown", chat_fn=lambda m: "")


def test_judge_extraction_handles_a_missing_field_without_crashing():
    # extract_structure can legitimately omit a field (see its docstring) --
    # judgment must not KeyError, and must still see the fields that ARE present.
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "NONE"

    extraction = {"claim": "a real claim", "grounds": "real grounds"}  # warrant, qualifier missing
    judge_extraction(extraction, "empirical", chat_fn=fake_chat)
    assert "a real claim" in captured["prompt"]
    assert "real grounds" in captured["prompt"]
    assert "(not extracted)" in captured["prompt"]


def test_judge_extraction_includes_extraction_fields_in_prompt_not_raw_abstract():
    # visible extraction principle: judgment reads the EXTRACTION, not the source text directly
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "NONE"

    extraction = {"claim": "a very specific claim text", "grounds": "specific grounds text", "warrant": "w", "qualifier": "q"}
    judge_extraction(extraction, "empirical", chat_fn=fake_chat)
    assert "a very specific claim text" in captured["prompt"]
    assert "specific grounds text" in captured["prompt"]


# ── judge_all ────────────────────────────────────────────────────────────────

def test_judge_all_judges_each_paper_by_its_own_type():
    def fake_chat(messages):
        prompt = messages[0]["content"]
        if "GROUNDS" in prompt or "grounds" in prompt.lower():
            return "FINDING: empirical issue\n"
        return "FINDING: survey issue\n"

    papers = {
        "seed-a": ({"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"}, "empirical"),
        "seed-b": ({"taxonomy": "t", "coverage": "c", "claimed_gaps": "g"}, "survey"),
    }
    result = judge_all(papers, chat_fn=fake_chat)

    assert result["seed-a"] == ["empirical issue"]
    assert result["seed-b"] == ["survey issue"]


def test_judge_all_handles_empty_input():
    assert judge_all({}, chat_fn=lambda m: "") == {}


# ── Layer 2: formal inference-validity check ────────────────────────────────

def test_check_inference_validity_parses_deductive_response():
    def fake_chat(messages):
        return (
            "P: models degrade when relevant info is in the middle\n"
            "Q: current models do not robustly use long-context information\n"
            "REASONING_TYPE: deductive\n"
            "ARGUMENT_FORM: P, if P then Q, therefore Q\n"
            "VALIDITY: valid (modus ponens)\n"
        )

    result = check_inference_validity("Title", "Abstract", chat_fn=fake_chat)
    assert result["reasoning_type"] == "deductive"
    assert result["validity"] == "valid (modus ponens)"
    assert result["p"] == "models degrade when relevant info is in the middle"
    assert result["q"] == "current models do not robustly use long-context information"


def test_check_inference_validity_parses_inductive_response():
    def fake_chat(messages):
        return (
            "P: 3,000 hours of red teaming found no universal jailbreak\n"
            "Q: Constitutional Classifiers are an effective defense\n"
            "REASONING_TYPE: inductive\n"
            "ARGUMENT_FORM: strong empirical evidence from extensive red-teaming\n"
            "VALIDITY: strong -- large, adversarial evaluation effort\n"
        )

    result = check_inference_validity("Title", "Abstract", chat_fn=fake_chat)
    assert result["reasoning_type"] == "inductive"
    assert result["validity"] == "strong -- large, adversarial evaluation effort"


def test_check_inference_validity_includes_title_and_abstract_in_prompt():
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "P: p\nQ: q\nREASONING_TYPE: deductive\nARGUMENT_FORM: f\nVALIDITY: valid\n"

    check_inference_validity("My Title", "My abstract text", chat_fn=fake_chat)
    assert "My Title" in captured["prompt"]
    assert "My abstract text" in captured["prompt"]


def test_check_inference_validity_asks_for_deductive_vs_inductive_classification():
    # the design doc's follow-up: classify reasoning type BEFORE the verdict,
    # rather than forcing everything through one valid/invalid binary
    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[0]["content"]
        return "P: p\nQ: q\nREASONING_TYPE: deductive\nARGUMENT_FORM: f\nVALIDITY: valid\n"

    check_inference_validity("Title", "Abstract", chat_fn=fake_chat)
    assert "deductive" in captured["prompt"].lower()
    assert "inductive" in captured["prompt"].lower()


def test_inference_fields_is_the_five_expected_fields():
    assert INFERENCE_FIELDS == ["P", "Q", "REASONING_TYPE", "ARGUMENT_FORM", "VALIDITY"]


def test_check_inference_validity_all_processes_each_paper():
    def fake_chat(messages):
        return "P: p\nQ: q\nREASONING_TYPE: deductive\nARGUMENT_FORM: f\nVALIDITY: valid\n"

    papers = {
        "seed-a": ("Title A", "Abstract A"),
        "seed-b": ("Title B", "Abstract B"),
    }
    result = check_inference_validity_all(papers, chat_fn=fake_chat)
    assert set(result) == {"seed-a", "seed-b"}
    assert result["seed-a"]["reasoning_type"] == "deductive"


def test_check_inference_validity_all_handles_empty_input():
    assert check_inference_validity_all({}, chat_fn=lambda m: "") == {}


def test_check_inference_validity_normalizes_reasoning_type_casing():
    # real qwen2.5:14b output varies casing: "Inductive", "INDUCTIVE", "inductive"
    def fake_chat(messages):
        return "P: p\nQ: q\nREASONING_TYPE: INDUCTIVE\nARGUMENT_FORM: f\nVALIDITY: strong\n"

    result = check_inference_validity("Title", "Abstract", chat_fn=fake_chat)
    assert result["reasoning_type"] == "inductive"


def test_check_inference_validity_reasoning_type_falls_back_to_unknown():
    def fake_chat(messages):
        return "P: p\nQ: q\nREASONING_TYPE: not sure honestly\nARGUMENT_FORM: f\nVALIDITY: strong\n"

    result = check_inference_validity("Title", "Abstract", chat_fn=fake_chat)
    assert result["reasoning_type"] == "unknown"


# ── compute_reliability_signal (rollup) ──────────────────────────────────────

_CLEAN_LAYER2 = {"reasoning_type": "inductive", "validity": "strong -- well supported"}


def test_reliability_levels_ordered_best_to_worst():
    assert RELIABILITY_LEVELS == ["high", "moderate", "low", "very-low"]


def test_zero_findings_and_clean_layer2_is_high():
    result = compute_reliability_signal([], _CLEAN_LAYER2)
    assert result["level"] == "high"
    assert result["findings"] == []


def test_one_finding_is_moderate():
    result = compute_reliability_signal(["a minor issue"], _CLEAN_LAYER2)
    assert result["level"] == "moderate"


def test_two_to_three_findings_is_low():
    result = compute_reliability_signal(["issue 1", "issue 2"], _CLEAN_LAYER2)
    assert result["level"] == "low"
    result = compute_reliability_signal(["issue 1", "issue 2", "issue 3"], _CLEAN_LAYER2)
    assert result["level"] == "low"


def test_four_or_more_findings_is_very_low():
    result = compute_reliability_signal(["1", "2", "3", "4"], _CLEAN_LAYER2)
    assert result["level"] == "very-low"


def test_deductive_invalid_floors_a_clean_paper_to_low_not_high():
    layer2 = {"reasoning_type": "deductive", "validity": "invalid (affirming the consequent)"}
    result = compute_reliability_signal([], layer2)
    assert result["level"] == "low"


def test_deductive_invalid_does_not_improve_an_already_worse_paper():
    layer2 = {"reasoning_type": "deductive", "validity": "invalid (denying the antecedent)"}
    result = compute_reliability_signal(["1", "2", "3", "4"], layer2)
    assert result["level"] == "very-low"  # floor is a minimum badness, not a cap upward


def test_deductive_invalid_adds_its_own_finding():
    layer2 = {"reasoning_type": "deductive", "validity": "invalid (affirming the consequent)"}
    result = compute_reliability_signal([], layer2)
    assert any("invalid" in f.lower() for f in result["findings"])


def test_deductive_valid_does_not_floor_or_add_a_finding():
    layer2 = {"reasoning_type": "deductive", "validity": "valid (modus ponens)"}
    result = compute_reliability_signal([], layer2)
    assert result["level"] == "high"
    assert result["findings"] == []


def test_inductive_weak_adds_a_finding_but_does_not_hard_floor():
    # weak inductive support is a finding like any other (contributes to the
    # count-based level), not the same automatic floor as a formal deductive
    # failure -- per the design doc, the floor rule is specific to FORMAL
    # inference-validity failures.
    layer2 = {"reasoning_type": "inductive", "validity": "weak -- overreaches the evidence"}
    result = compute_reliability_signal([], layer2)
    assert result["level"] == "moderate"  # 1 finding (the layer-2 one), not floored to low
    assert any("weak" in f.lower() for f in result["findings"])


def test_unknown_reasoning_type_does_not_crash_or_floor():
    layer2 = {"reasoning_type": "unknown", "validity": "unclear"}
    result = compute_reliability_signal([], layer2)
    assert result["level"] == "high"


def test_missing_layer2_fields_does_not_crash():
    result = compute_reliability_signal(["an issue"], {})
    assert result["level"] == "moderate"


# ── compute_all_reliability_signals ─────────────────────────────────────────

def test_compute_all_reliability_signals_combines_per_paper():
    layer1 = {"seed-a": [], "seed-b": ["issue"]}
    layer2 = {"seed-a": _CLEAN_LAYER2, "seed-b": _CLEAN_LAYER2}
    result = compute_all_reliability_signals(layer1, layer2)
    assert result["seed-a"]["level"] == "high"
    assert result["seed-b"]["level"] == "moderate"


def test_compute_all_reliability_signals_handles_empty_input():
    assert compute_all_reliability_signals({}, {}) == {}


# ── needs_second_look (task 17) ─────────────────────────────────────────────

def test_needs_second_look_true_for_low():
    assert needs_second_look({"level": "low", "findings": []}) is True


def test_needs_second_look_true_for_very_low():
    assert needs_second_look({"level": "very-low", "findings": []}) is True


def test_needs_second_look_false_for_high():
    assert needs_second_look({"level": "high", "findings": []}) is False


def test_needs_second_look_false_for_moderate():
    assert needs_second_look({"level": "moderate", "findings": []}) is False


# ── build_audit_report (task 18) ────────────────────────────────────────────

def _sample_report_inputs():
    titles = {"seed-a": "Seed A Title", "seed-b": "Seed B Title"}
    paper_types = {"seed-a": "empirical", "seed-b": "benchmark"}
    extractions = {
        "seed-a": {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"},
        "seed-b": {"measures": "m", "construct_validity_evidence": "e", "scope": "s"},
    }
    layer1_findings = {"seed-a": [], "seed-b": ["construct validity assumed, not evidenced"]}
    layer2_results = {
        "seed-a": {"p": "p", "q": "q", "reasoning_type": "inductive", "argument_form": "f", "validity": "strong"},
        "seed-b": {"p": "p2", "q": "q2", "reasoning_type": "inductive", "argument_form": "f2", "validity": "strong"},
    }
    return titles, paper_types, extractions, layer1_findings, layer2_results


def test_build_audit_report_includes_one_entry_per_slug_in_given_order():
    titles, paper_types, extractions, layer1_findings, layer2_results = _sample_report_inputs()
    report = build_audit_report(
        ["seed-b", "seed-a"], titles, paper_types, extractions, layer1_findings, layer2_results
    )
    assert [e["slug"] for e in report] == ["seed-b", "seed-a"]


def test_build_audit_report_entry_has_all_expected_fields():
    titles, paper_types, extractions, layer1_findings, layer2_results = _sample_report_inputs()
    report = build_audit_report(
        ["seed-a"], titles, paper_types, extractions, layer1_findings, layer2_results
    )
    entry = report[0]
    assert entry["title"] == "Seed A Title"
    assert entry["paper_type"] == "empirical"
    assert entry["layer1_extraction"] == {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"}
    assert entry["layer2"]["reasoning_type"] == "inductive"
    assert entry["reliability_level"] == "high"
    assert entry["reliability_findings"] == []
    assert entry["needs_second_look"] is False


def test_build_audit_report_flags_low_reliability_papers():
    titles, paper_types, extractions, layer1_findings, layer2_results = _sample_report_inputs()
    report = build_audit_report(
        ["seed-b"], titles, paper_types, extractions, layer1_findings, layer2_results
    )
    entry = report[0]
    assert entry["reliability_level"] == "moderate"  # 1 finding
    assert entry["needs_second_look"] is False  # moderate isn't flagged, only low/very-low


def test_build_audit_report_handles_empty_slug_list():
    titles, paper_types, extractions, layer1_findings, layer2_results = _sample_report_inputs()
    assert build_audit_report([], titles, paper_types, extractions, layer1_findings, layer2_results) == []


# ── render_markdown ──────────────────────────────────────────────────────────

def test_render_markdown_includes_title_type_and_reliability():
    report = [{
        "slug": "seed-a",
        "title": "Seed A Title",
        "paper_type": "empirical",
        "layer1_extraction": {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"},
        "layer1_findings": [],
        "layer2": {"p": "p", "q": "q", "reasoning_type": "inductive", "argument_form": "f", "validity": "strong"},
        "reliability_level": "high",
        "reliability_findings": [],
        "needs_second_look": False,
    }]
    md = render_markdown(report)
    assert "Seed A Title" in md
    assert "empirical" in md
    assert "high" in md


def test_render_markdown_flags_needs_second_look_papers_visibly():
    report = [{
        "slug": "seed-b",
        "title": "Seed B Title",
        "paper_type": "benchmark",
        "layer1_extraction": {"measures": "m", "construct_validity_evidence": "e", "scope": "s"},
        "layer1_findings": ["construct validity assumed"],
        "layer2": {"p": "p", "q": "q", "reasoning_type": "inductive", "argument_form": "f", "validity": "strong"},
        "reliability_level": "low",
        "reliability_findings": ["construct validity assumed"],
        "needs_second_look": True,
    }]
    md = render_markdown(report)
    assert "construct validity assumed" in md
    assert "⚠" in md  # visible flag for needs_second_look


def test_render_markdown_handles_no_findings_gracefully():
    report = [{
        "slug": "seed-a",
        "title": "Seed A Title",
        "paper_type": "empirical",
        "layer1_extraction": {"claim": "c", "grounds": "g", "warrant": "w", "qualifier": "q"},
        "layer1_findings": [],
        "layer2": {"p": "p", "q": "q", "reasoning_type": "inductive", "argument_form": "f", "validity": "strong"},
        "reliability_level": "high",
        "reliability_findings": [],
        "needs_second_look": False,
    }]
    md = render_markdown(report)
    assert "no findings" in md.lower() or "none" in md.lower()
