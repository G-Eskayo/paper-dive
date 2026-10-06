#!/usr/bin/env python3
"""
logic_auditor.py — judges whether a paper's own argument is internally
consistent (paper-dive synthesis tool, see marvin-roadmap.md §K's "long-term
vision" item, scoped down via a grill-with-docs session 2026-07-13). Full
design notes are kept in a separate research project.

Distinct from argument_mapper.py (structural core-claim + "builds on" graph
only, no judgment of argument quality). v1 scope: the 15 hand-picked seed
papers only, per-paper only (cross-paper comparison deferred to a future
competing-ideas-surface tool, see ADR 0001).

Model choice validated 2026-07-13 against all 15 real seed abstracts (see
the design doc's "Model" section): qwen2.5:14b for classification and layer
2 (formal inference-validity), qwen2.5:3b for layer-1 extraction, qwen2.5:7b
for layer-1 judgment -- model size was validated per task, never assumed:
7b was a real regression for classification but the fix for judgment (3b
produced literal single-word non-answers there), and 3b that was fine for
extraction produced the same kind of garbage on judgment. No single size
fits every stage.
"""
from __future__ import annotations
import json
import re
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
# Validated 2026-07-13 against all 15 real seed abstracts: 3b (even with a prompt refined to
# counter its "named attack/method -> conceptual" bias) still misclassified MINJA and
# many-shot-jailbreaking as conceptual despite both reporting concrete measured attack success
# rates. 7b was WORSE, not better -- overcorrected to "empirical" broadly and broke two
# previously-correct survey/benchmark classifications (sok-trust-authorization-mismatch,
# sorry-bench). 14b fixed both stubborn cases with zero regressions on the rest.
CLASSIFY_MODEL = "qwen2.5:14b"

PAPER_TYPES = {"empirical", "survey", "benchmark", "conceptual"}


def ollama_chat(model: str, messages: list[dict], timeout: int = 60) -> str:
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


CLASSIFY_PROMPT = """Below is an academic paper's title and abstract. Classify what KIND of argument it makes, choosing exactly one of these four types:

- empirical: presents new experiments or measurements to support a claim
- survey: synthesizes/categorizes existing work into a taxonomy (includes "SoK" / systematization-of-knowledge papers) -- the real argument is that the taxonomy correctly carves up the space, not new experimental evidence
- benchmark: introduces a measurement instrument or dataset -- the real argument is about construct validity (does it measure what it claims to)
- conceptual: proposes a model or framework via explicit structural claims ("if X, then Y"), without primarily relying on new experiments

IMPORTANT: naming a specific method, attack, or system (e.g. an attack called "MINJA" or an
algorithm like GCG) does NOT by itself make a paper conceptual. What matters is the EVIDENCE: if
the abstract reports specific measured results from running something -- percentages, success
rates, hours of testing, benchmark scores, statistical findings -- classify it empirical, even
though it also names and describes a method. Reserve "conceptual" for papers whose central
argument is a structural or theoretical claim without new experimental measurement backing it
(a taxonomy, a formal framework, an argument about how systems relate to each other).

Title: {title}

Abstract: {abstract}

Respond with ONLY one word: empirical, survey, benchmark, or conceptual."""


def classify_paper_type(title: str, abstract: str, chat_fn=None) -> str:
    """Returns one of PAPER_TYPES, or "unknown" if the response can't be
    parsed into exactly one of them -- deliberately not guessing/defaulting
    silently, since a wrong type routes the paper to the wrong evaluation
    method downstream."""
    chat_fn = chat_fn or (lambda messages: ollama_chat(CLASSIFY_MODEL, messages))
    prompt = CLASSIFY_PROMPT.format(title=title, abstract=abstract)
    response = chat_fn([{"role": "user", "content": prompt}]).strip().lower()

    found = [t for t in PAPER_TYPES if t in response]
    if len(found) == 1:
        return found[0]
    return "unknown"


def classify_all(papers: dict[str, tuple[str, str]], chat_fn=None) -> dict[str, str]:
    """papers: {slug: (title, abstract)}. Returns {slug: type}."""
    return {
        slug: classify_paper_type(title, abstract, chat_fn=chat_fn)
        for slug, (title, abstract) in papers.items()
    }


# ── Layer 1: type-adaptive visible extraction ───────────────────────────────
# Model: qwen2.5:3b, validated in the design's spot-check (task 11) -- clean,
# accurate Toulmin extraction on real abstracts, no hallucinated content.

EXTRACTION_MODEL = "qwen2.5:3b"

# Field names per paper type, per docs/logic-auditor-design.md's table.
# Order matters -- it's both the prompt's requested response order and
# _parse_fields' scan order.
EXTRACTION_FIELDS = {
    "empirical": ["CLAIM", "GROUNDS", "WARRANT", "QUALIFIER"],
    "survey": ["TAXONOMY", "COVERAGE", "CLAIMED_GAPS"],
    "benchmark": ["MEASURES", "CONSTRUCT_VALIDITY_EVIDENCE", "SCOPE"],
    "conceptual": ["STRUCTURAL_CLAIMS", "KEY_DEFINITIONS", "INTERNAL_DEPENDENCIES"],
}

_EXTRACTION_PROMPTS = {
    "empirical": """Below is an academic paper's title and abstract. Decompose its central argument into Toulmin's model. Be concise -- one or two sentences per field, quoting or closely paraphrasing the abstract's own language where possible, not inventing detail that isn't there.

Title: {title}
Abstract: {abstract}

Respond in exactly this format:
CLAIM: <the paper's core assertion>
GROUNDS: <the evidence/data presented for it>
WARRANT: <why the grounds are supposed to support the claim>
QUALIFIER: <how strong/hedged the claim is -- confidence level, stated limitations, scope>""",

    "survey": """Below is a survey/SoK (Systematization of Knowledge) paper's title and abstract. Its real argument is that its taxonomy correctly carves up the space it reviews, not new experimental evidence. Be concise, quoting or closely paraphrasing the abstract's own language, not inventing detail that isn't there.

Title: {title}
Abstract: {abstract}

Respond in exactly this format:
TAXONOMY: <the categorization/framework it proposes>
COVERAGE: <the scope of what it reviewed -- how many papers, what domain>
CLAIMED_GAPS: <what the survey says is missing or unaddressed in the field>""",

    "benchmark": """Below is a benchmark/dataset paper's title and abstract. Its real argument is about construct validity -- does it actually measure what it claims to measure. Be concise, quoting or closely paraphrasing the abstract's own language, not inventing detail that isn't there.

Title: {title}
Abstract: {abstract}

Respond in exactly this format:
MEASURES: <the real-world capability or property it claims to measure>
CONSTRUCT_VALIDITY_EVIDENCE: <evidence given that the measurement actually correlates with that capability>
SCOPE: <what the benchmark covers -- categories, size, source>""",

    "conceptual": """Below is a conceptual/framework paper's title and abstract. It proposes a model via explicit structural claims, without primarily relying on new experiments. Be concise, quoting or closely paraphrasing the abstract's own language, not inventing detail that isn't there.

Title: {title}
Abstract: {abstract}

Respond in exactly this format:
STRUCTURAL_CLAIMS: <the paper's "if X, then Y" style structural assertions>
KEY_DEFINITIONS: <the core concepts/terms the framework introduces>
INTERNAL_DEPENDENCIES: <how the framework's parts relate to or depend on each other>""",
}


def _parse_fields(response: str, field_names: list[str]) -> dict[str, str]:
    """Parses a "FIELD: value" formatted response into {lowercase_field: value},
    accumulating multi-line values until the next recognized field label.
    A field absent from the response is simply absent from the result, not
    an empty string -- callers can tell "not extracted" from "extracted as
    empty," which matters for downstream judgment."""
    upper_names = {name.upper(): name.lower() for name in field_names}
    result: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_key is not None:
            result[current_key] = "\n".join(current_lines).strip()

    for line in response.splitlines():
        matched = None
        for upper, lower in upper_names.items():
            prefix = f"{upper}:"
            if line.strip().upper().startswith(prefix):
                matched = (lower, line.strip()[len(prefix):].strip())
                break
        if matched:
            flush()
            current_key, first_line = matched
            current_lines = [first_line] if first_line else []
        elif current_key is not None:
            current_lines.append(line)
    flush()
    return result


def extract_structure(title: str, abstract: str, paper_type: str, chat_fn=None) -> dict[str, str]:
    """Extracts the type-appropriate structure as a reviewable intermediate
    artifact -- never a black-box verdict (see CONTEXT.md "visible
    extraction"). Judgment against this structure is a separate step (task
    14), not done here."""
    if paper_type not in _EXTRACTION_PROMPTS:
        raise ValueError(f"unknown paper_type {paper_type!r} -- expected one of {PAPER_TYPES}")

    chat_fn = chat_fn or (lambda messages: ollama_chat(EXTRACTION_MODEL, messages))
    prompt = _EXTRACTION_PROMPTS[paper_type].format(title=title, abstract=abstract)
    response = chat_fn([{"role": "user", "content": prompt}])
    return _parse_fields(response, EXTRACTION_FIELDS[paper_type])


def extract_all(papers: dict[str, tuple[str, str, str]], chat_fn=None) -> dict[str, dict]:
    """papers: {slug: (title, abstract, paper_type)}. Returns {slug: extraction_dict}."""
    return {
        slug: extract_structure(title, abstract, paper_type, chat_fn=chat_fn)
        for slug, (title, abstract, paper_type) in papers.items()
    }


# ── Layer 1, step 2: type-adaptive consistency judgment ─────────────────────
# Model: qwen2.5:7b, validated 2026-07-13 against all 15 real extractions.
# 3b produced genuine garbage on this harder reasoning task -- confirmed via
# raw response inspection, not parsing artifacts alone: e.g. the entire
# response for one paper was "FINDING: WARRANT", no content at
# all, plus repeated field-label bleed-through (echoing "QUALIFIER: ..."
# mid-finding) and frequent fake findings confirming things were fine
# instead of reporting real issues. 7b fixed all of it cleanly -- no
# truncation, no bleed-through, substantive well-reasoned findings, correct
# "(no findings)" on papers with genuinely strong support. Notably the
# mirror image of the classifier result (7b was a regression there) --
# reinforces that model size must be validated per-task, not assumed.
JUDGMENT_MODEL = "qwen2.5:7b"

# benchmark judgment specifically needs 14b, not the 7b default -- found
# 2026-07-13 reviewing flagged results (task 19): 7b called StrongREJECT's
# construct-validity evidence "merely asserted" even though its own
# extraction already states real comparison-based evidence ("achieves
# state-of-the-art agreement with human judgments"), and kept doing so
# even after two rounds of more forceful prompt wording -- the model was
# overriding explicit instructions with its own second-order skepticism
# about whether human-judgment agreement should count. 14b followed the
# instruction correctly on the first try, with zero regression on the
# other 2 benchmarks (jailbreakbench, sorry-bench correctly still flagged,
# since their own evidence genuinely has no comparison at all). The other
# 3 paper types stay on 7b -- no evidence of a problem there, no reason to
# pay for a bigger model where the cheaper one already works.
JUDGMENT_MODEL_OVERRIDES = {"benchmark": "qwen2.5:14b"}

_GENERAL_FALLACY_CHECKLIST = (
    "hasty generalization, circular reasoning, false dichotomy, unfalsifiable claims, "
    "survivorship bias, p-hacking/multiple-comparisons"
)

_JUDGMENT_PROMPTS = {
    "empirical": """Below is a Toulmin-model extraction of an empirical paper's argument (CLAIM/GROUNDS/WARRANT/QUALIFIER) -- judge the EXTRACTION below, not any paper you may know of with a similar topic. Check specifically:
- Does the QUALIFIER accurately reflect how strong the GROUNDS actually are (not overclaiming beyond what the grounds support, not underclaiming either)?
- Does the WARRANT commit a recognizable fallacy -- correlation treated as causation, an unaddressed confound, cherry-picked examples?
- General fallacy checklist: {fallacies}

CLAIM: {claim}
GROUNDS: {grounds}
WARRANT: {warrant}
QUALIFIER: {qualifier}

Only output a FINDING line for an actual problem -- never to confirm something checks out fine. If a check passes, simply say nothing about it; do not write a FINDING line like "the qualifier is accurate" or "no fallacy found." List each real issue you find, one per line, each starting with "FINDING:". If you find no issues at all, respond with exactly "NONE" and nothing else.""",

    "survey": """Below is an extraction of a survey/SoK paper's argument (TAXONOMY/COVERAGE/CLAIMED_GAPS) -- judge the EXTRACTION below, not any paper you may know of with a similar topic. Check specifically:
- Is the TAXONOMY internally consistent -- no categories presented as mutually exclusive that could actually overlap, no coverage claimed as exhaustive that isn't?
- Are the CLAIMED_GAPS actually supported by the stated COVERAGE, or just asserted without connection to what was reviewed?
- General fallacy checklist: {fallacies}

TAXONOMY: {taxonomy}
COVERAGE: {coverage}
CLAIMED_GAPS: {claimed_gaps}

Only output a FINDING line for an actual problem -- never to confirm something checks out fine. If a check passes, simply say nothing about it; do not write a FINDING line like "the qualifier is accurate" or "no fallacy found." List each real issue you find, one per line, each starting with "FINDING:". If you find no issues at all, respond with exactly "NONE" and nothing else.""",

    "benchmark": """Below is an extraction of a benchmark/dataset paper's argument (MEASURES/CONSTRUCT_VALIDITY_EVIDENCE/SCOPE) -- judge the EXTRACTION below, not any paper you may know of with a similar topic. Check specifically:
- Is the CONSTRUCT_VALIDITY_EVIDENCE actual evidence that the benchmark measures what it claims, or is construct validity merely assumed/asserted?

IMPORTANT: if CONSTRUCT_VALIDITY_EVIDENCE mentions ANY comparison to human judgment, ground
truth, or an existing benchmark/gold-standard measure, DO NOT flag construct validity as a
finding at all -- treat that comparison as sufficient evidence, full stop, even if you personally
think agreement-with-human-judgment has its own epistemic limits as a validation method. That
second-order concern is out of scope here; do not raise it. Only flag construct validity as
assumed/asserted when the evidence describes NO comparison of any kind -- e.g. reasoning from
design features alone (a taxonomy's granularity, a defined threat model) with nothing checked
against any external criterion.

- General fallacy checklist: {fallacies}

MEASURES: {measures}
CONSTRUCT_VALIDITY_EVIDENCE: {construct_validity_evidence}
SCOPE: {scope}

Only output a FINDING line for an actual problem -- never to confirm something checks out fine. If a check passes, simply say nothing about it; do not write a FINDING line like "the qualifier is accurate" or "no fallacy found." List each real issue you find, one per line, each starting with "FINDING:". If you find no issues at all, respond with exactly "NONE" and nothing else.""",

    "conceptual": """Below is an extraction of a conceptual/framework paper's argument (STRUCTURAL_CLAIMS/KEY_DEFINITIONS/INTERNAL_DEPENDENCIES) -- judge the EXTRACTION below, not any paper you may know of with a similar topic. Check specifically:
- Do the STRUCTURAL_CLAIMS and INTERNAL_DEPENDENCIES actually hold together, or does any part contradict another?
- General fallacy checklist: {fallacies}

STRUCTURAL_CLAIMS: {structural_claims}
KEY_DEFINITIONS: {key_definitions}
INTERNAL_DEPENDENCIES: {internal_dependencies}

Only output a FINDING line for an actual problem -- never to confirm something checks out fine. If a check passes, simply say nothing about it; do not write a FINDING line like "the qualifier is accurate" or "no fallacy found." List each real issue you find, one per line, each starting with "FINDING:". If you find no issues at all, respond with exactly "NONE" and nothing else.""",
}


_FINDING_SPLIT_RE = re.compile(r"FINDING:\s*", re.IGNORECASE)


def _parse_findings(response: str) -> list[str]:
    """qwen2.5:3b doesn't reliably put one FINDING per line -- seen live
    2026-07-13 running multiple together on one line, separated only by the
    next "FINDING:" marker rather than a newline. Splits on every occurrence
    of the marker anywhere in the text instead of assuming one per line.
    Also drops any resulting item that's just a stray "NONE" (also seen
    live: an extra trailing "FINDING: NONE" instead of the bare NONE
    response) -- a real finding is never literally the word "none". And
    strips a stray trailing "NONE" line glued onto the end of a real
    finding with no second "FINDING:" marker to separate them (seen live
    on qwen2.5:14b's sorry-bench output) -- the whole-piece check above
    doesn't catch this since the piece isn't JUST "NONE", it's a real
    finding with "NONE" appended after it."""
    response = response.strip()
    if not response or response.upper() == "NONE":
        return []
    pieces = _FINDING_SPLIT_RE.split(response)[1:]  # [0] is any preamble before the first marker
    cleaned = []
    for piece in pieces:
        piece = piece.strip()
        if not piece or piece.upper() == "NONE":
            continue
        lines = piece.splitlines()
        if lines and lines[-1].strip().upper() == "NONE":
            piece = "\n".join(lines[:-1]).strip()
        if piece:
            cleaned.append(piece)
    return cleaned


def judge_extraction(extraction: dict[str, str], paper_type: str, chat_fn=None) -> list[str]:
    """Judges a type-adaptive extraction (from extract_structure) against its
    type-specific rubric plus the general fallacy checklist. Reads the
    EXTRACTION, never the raw abstract directly -- per the visible-extraction
    principle, judgment happens against the reviewable intermediate artifact,
    not straight from source text. Returns a list of specific findings, never
    a bare score; an empty list means no issues found."""
    if paper_type not in _JUDGMENT_PROMPTS:
        raise ValueError(f"unknown paper_type {paper_type!r} -- expected one of {PAPER_TYPES}")

    if chat_fn is None:
        model = JUDGMENT_MODEL_OVERRIDES.get(paper_type, JUDGMENT_MODEL)
        chat_fn = lambda messages: ollama_chat(model, messages)
    # extract_structure can legitimately omit a field the model never returned
    # (see its docstring) -- fill any gap so .format() doesn't KeyError on a
    # missing extraction rather than surfacing it as a finding-worthy gap.
    # extraction's own keys are already lowercase (what _parse_fields produces).
    fields = {
        name.lower(): extraction.get(name.lower(), "(not extracted)")
        for name in EXTRACTION_FIELDS[paper_type]
    }
    prompt = _JUDGMENT_PROMPTS[paper_type].format(fallacies=_GENERAL_FALLACY_CHECKLIST, **fields)
    response = chat_fn([{"role": "user", "content": prompt}])
    return _parse_findings(response)


def judge_all(papers: dict[str, tuple[dict, str]], chat_fn=None) -> dict[str, list[str]]:
    """papers: {slug: (extraction_dict, paper_type)}. Returns {slug: findings_list}."""
    return {
        slug: judge_extraction(extraction, paper_type, chat_fn=chat_fn)
        for slug, (extraction, paper_type) in papers.items()
    }


# ── Layer 2: formal inference-validity check ────────────────────────────────
# Model: qwen2.5:14b, validated in the design's spot-check (task 11) --
# 3b/7b both got the inferential direction backwards on a real test case
# (treated evidence as a conclusion derived from the claim, rather than the
# evidence FOR the claim); 14b got it right on every test case and showed
# unprompted awareness of the deductive-vs-inductive distinction. Type-
# agnostic, unlike layer 1 -- operates directly on title+abstract, not
# routed through extract_structure's per-type extraction, since the
# inferential move being checked here is a different, universal question
# ("does the conclusion follow from the premises") independent of what kind
# of argument the paper makes.
INFERENCE_MODEL = "qwen2.5:14b"

INFERENCE_FIELDS = ["P", "Q", "REASONING_TYPE", "ARGUMENT_FORM", "VALIDITY"]

INFERENCE_PROMPT = """Below is an academic paper's title and abstract. Identify its single most important inferential move -- the step where it goes from evidence/premises to its conclusion. Symbolize it into propositions (P, Q, etc.), using the paper's own actual premises and conclusion -- do not invent an inference that isn't there.

First classify whether this move is DEDUCTIVE (the conclusion is claimed to follow necessarily and with certainty from the premises, as a matter of pure logical form) or INDUCTIVE (the conclusion is supported by evidence but doesn't follow with logical necessity -- a matter of degree, common in empirical science). Most empirical papers are inductive; treat DEDUCTIVE as the exception, not the default.

Then:
- If DEDUCTIVE: state the argument form explicitly (e.g. "P, if P then Q, therefore Q") and assess VALID (a recognized valid form like modus ponens) or INVALID (a recognized fallacy like affirming the consequent or denying the antecedent).
- If INDUCTIVE: state what evidence supports the conclusion, and assess STRONG (the evidence genuinely supports the conclusion, appropriately hedged) or WEAK (the conclusion overreaches what the evidence can support).

Judge only whether the conclusion follows from the stated premises -- not whether the premises themselves are true.

Title: {title}
Abstract: {abstract}

Respond in exactly this format:
P: <what you're calling P, in the paper's own terms>
Q: <what you're calling Q, in the paper's own terms>
REASONING_TYPE: <deductive or inductive>
ARGUMENT_FORM: <the explicit form for deductive, or a brief description of the evidential support for inductive>
VALIDITY: <for deductive: valid/invalid + named form or fallacy. for inductive: strong/weak + brief reason>"""


_REASONING_TYPES = {"deductive", "inductive"}


def check_inference_validity(title: str, abstract: str, chat_fn=None) -> dict[str, str]:
    """Symbolizes the paper's key inferential move and checks the argument's
    FORM -- does the conclusion follow from the stated premises -- never
    whether the premises are true (that's layer 1's job). Visible
    extraction: the symbolization (P, Q, reasoning type) is always part of
    the returned result, not hidden behind a bare verdict.

    reasoning_type is normalized the same way classify_paper_type normalizes
    paper type -- real qwen2.5:14b output varies casing ("Inductive",
    "INDUCTIVE", "inductive"), and task 16's rollup needs to branch on this
    value reliably, not match free text."""
    chat_fn = chat_fn or (lambda messages: ollama_chat(INFERENCE_MODEL, messages))
    prompt = INFERENCE_PROMPT.format(title=title, abstract=abstract)
    response = chat_fn([{"role": "user", "content": prompt}])
    result = _parse_fields(response, INFERENCE_FIELDS)

    raw_type = result.get("reasoning_type", "").strip().lower()
    found = [t for t in _REASONING_TYPES if t in raw_type]
    result["reasoning_type"] = found[0] if len(found) == 1 else "unknown"
    return result


def check_inference_validity_all(papers: dict[str, tuple[str, str]], chat_fn=None) -> dict[str, dict]:
    """papers: {slug: (title, abstract)}. Returns {slug: inference_check_dict}."""
    return {
        slug: check_inference_validity(title, abstract, chat_fn=chat_fn)
        for slug, (title, abstract) in papers.items()
    }


# ── Reliability signal rollup ────────────────────────────────────────────────
# Combines layer 1 (type-adaptive judgment) findings and layer 2 (formal
# inference-validity) into the one common comparable output across all
# paper types, per docs/logic-auditor-design.md's "Reliability signal"
# section. A categorical rating, not a blended numeric score -- a numeric
# average would imply layer 1 and layer 2 findings are commensurable and
# averageable, which they aren't (a formal-validity failure and a weak-
# warrant finding are different KINDS of problems). No calibration data
# exists yet for the count thresholds below; this is a deliberately simple,
# transparent v1 -- a living design already flagged as expected to evolve
# once real use surfaces better information (see the grill session).

RELIABILITY_LEVELS = ["high", "moderate", "low", "very-low"]  # best to worst


def _level_from_finding_count(n: int) -> str:
    if n == 0:
        return "high"
    if n == 1:
        return "moderate"
    if n <= 3:
        return "low"
    return "very-low"


def _worse_of(a: str, b: str) -> str:
    return a if RELIABILITY_LEVELS.index(a) >= RELIABILITY_LEVELS.index(b) else b


def compute_reliability_signal(layer1_findings: list[str], layer2_result: dict) -> dict:
    """layer1_findings: from judge_extraction. layer2_result: from
    check_inference_validity. Returns {"level": one of RELIABILITY_LEVELS,
    "findings": full list from both layers}.

    Floor rule: a FORMAL inference-validity failure (deductive reasoning
    assessed invalid) floors the level at "low" -- a minimum badness, not a
    cap; a paper that's already worse than "low" from its own layer-1
    findings stays at that worse level. Weak INDUCTIVE support is not the
    same thing -- per the design doc, the floor is specific to formal
    validity failures, since inductive evidential strength is a matter of
    degree, not a strict logic bug. A weak-inductive result is folded in as
    an ordinary finding instead, contributing to the count-based level like
    any other."""
    findings = list(layer1_findings)

    reasoning_type = layer2_result.get("reasoning_type", "unknown")
    validity = layer2_result.get("validity", "")
    validity_lower = validity.lower()

    is_formal_failure = reasoning_type == "deductive" and "invalid" in validity_lower
    is_weak_inductive = reasoning_type == "inductive" and "weak" in validity_lower

    if is_formal_failure:
        findings.append(f"Layer 2: deductive argument assessed invalid -- {validity}")
    elif is_weak_inductive:
        findings.append(f"Layer 2: inductive support assessed weak -- {validity}")

    level = _level_from_finding_count(len(findings))
    if is_formal_failure:
        level = _worse_of(level, "low")

    return {"level": level, "findings": findings}


def compute_all_reliability_signals(
    layer1_findings: dict[str, list[str]], layer2_results: dict[str, dict]
) -> dict[str, dict]:
    """layer1_findings: {slug: findings_list} from judge_all.
    layer2_results: {slug: inference_check_dict} from check_inference_validity_all.
    Returns {slug: {"level": ..., "findings": [...]}}."""
    slugs = set(layer1_findings) | set(layer2_results)
    return {
        slug: compute_reliability_signal(layer1_findings.get(slug, []), layer2_results.get(slug, {}))
        for slug in slugs
    }


# ── needs-a-second-look flagging ─────────────────────────────────────────────
# Lighter human-review mechanism than safety-monitor's formal quarantine gate
# -- no approve/modify/deny workflow. This tool's output is closer to a
# research assistant's annotated notes for Gil to read while drafting, not
# an autonomous loop shipping unattended (what quarantine was built for).
# Flags low/very-low results for a deliberate second read, rather than
# trusting the categorical call outright. Kept as its own small function
# (composable on top of compute_reliability_signal) rather than folded into
# the rollup itself, per the same build-small-primitives principle used
# throughout this codebase.

SECOND_LOOK_LEVELS = {"low", "very-low"}


def needs_second_look(reliability_signal: dict) -> bool:
    return reliability_signal.get("level") in SECOND_LOOK_LEVELS


# ── Output artifacts (JSON + markdown) ───────────────────────────────────────
# Mirrors argument_mapper.py's dual-output pattern: structured JSON for
# future tools to consume (e.g. a future competing-ideas-surface tool), a
# human-readable markdown report for reading directly while drafting.

def build_audit_report(
    slugs: list[str],
    titles: dict[str, str],
    paper_types: dict[str, str],
    extractions: dict[str, dict],
    layer1_findings: dict[str, list[str]],
    layer2_results: dict[str, dict],
) -> list[dict]:
    """Assembles the full per-paper audit -- type, both layers' visible
    extraction, findings, and the rolled-up reliability signal -- in the
    given slug order. One entry per slug; missing upstream data for a slug
    just produces empty/default fields rather than erroring, since a
    partial pipeline run (e.g. layer 2 not yet computed for a new paper)
    shouldn't block building a report for what IS available."""
    report = []
    for slug in slugs:
        l1_findings = layer1_findings.get(slug, [])
        l2_result = layer2_results.get(slug, {})
        signal = compute_reliability_signal(l1_findings, l2_result)
        report.append({
            "slug": slug,
            "title": titles.get(slug, slug),
            "paper_type": paper_types.get(slug, "unknown"),
            "layer1_extraction": extractions.get(slug, {}),
            "layer1_findings": l1_findings,
            "layer2": l2_result,
            "reliability_level": signal["level"],
            "reliability_findings": signal["findings"],
            "needs_second_look": needs_second_look(signal),
        })
    return report


def render_markdown(report: list[dict]) -> str:
    lines = [
        "# Logic Audit Report",
        "",
        "Generated by `logic_auditor.py`. Judges whether each paper's own argument is "
        "internally consistent -- distinct from `argument_mapper.py`'s core-claim extraction, "
        "which makes no judgment of argument quality. Papers flagged ⚠️ scored low/very-low "
        "reliability and are worth a deliberate second read, not an automated final word.",
        "",
    ]
    for entry in report:
        flag = " ⚠️" if entry["needs_second_look"] else ""
        lines.append(f"## {entry['title']}{flag}")
        lines.append("")
        lines.append(f"**Type**: {entry['paper_type']}  ")
        lines.append(f"**Reliability**: {entry['reliability_level']}")
        lines.append("")
        lines.append("### Layer 1 — Extraction")
        for field, value in entry["layer1_extraction"].items():
            lines.append(f"- **{field.replace('_', ' ').title()}**: {value}")
        lines.append("")
        lines.append("### Layer 2 — Formal Inference Check")
        for field, value in entry["layer2"].items():
            lines.append(f"- **{field.replace('_', ' ').title()}**: {value}")
        lines.append("")
        lines.append("### Findings")
        if entry["reliability_findings"]:
            for finding in entry["reliability_findings"]:
                lines.append(f"- {finding}")
        else:
            lines.append("No findings.")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classify paper-knowledge seed papers by argument type.")
    ap.add_argument("--seeds-json", required=True, help="Path to a JSON file: {slug: title, ...}")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from pathlib import Path
    import chromadb

    CHROMA_PATH = Path.home() / ".claude" / "chroma"
    seed_titles = json.loads(Path(args.seeds_json).read_text())

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection("paper-knowledge")
    data = collection.get(ids=list(seed_titles), include=["documents"])
    abstracts = dict(zip(data["ids"], data["documents"]))

    papers = {slug: (title, abstracts[slug]) for slug, title in seed_titles.items() if slug in abstracts}
    result = classify_all(papers)

    Path(args.out_json).write_text(json.dumps(result, indent=2))
    for slug, ptype in result.items():
        print(f"{slug}: {ptype}")
