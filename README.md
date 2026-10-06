# paper-dive

A Socratic reading partner for scientific papers, for [Claude Code](https://claude.ai/code) — and a recursive citation-graph knowledge-base builder for a topic, not just a single paper.

## What it does

**Reading mode**: walks you from plain-English understanding up to expert-level synthesis through conversation, not a summary dump. Adapts explanations to analogy domains you already understand. Runs challenge and compare modes to stress-test a paper's claims.

**Citation-graph mode** (`/paper-graph`): given a seed paper — published (has a DOI) or not (a draft, a preprint, anything you're reading/writing) — recursively traverses its citation graph (both what it cites and what cites it), building a locally-stored, deduplicated knowledge base on the topic. Ordered by relevance to the seed via a blended embedding score, not plain breadth/depth-first order, with a guardrail against silently dropping rebuttals and a checkpoint instead of a silent cutoff when the traversal's budget runs out. Full design rationale in [`docs/adr/`](docs/adr/): seven ADRs covering the traversal algorithm, the relevance scoring, a real Semantic Scholar rate-limit finding, and when to fetch full text. They start at 0007 because they were written inside MARVIN and keep its numbering.

**Synthesis tools** (v1, all run on local models through Ollama, no API cost). They work on a knowledge base the citation-graph mode built:

- `argument_mapper.py` pulls each seed paper's core claim and maps which papers it builds on.
- `logic_auditor.py` checks whether a paper's own argument holds together. Model sizes were picked per task and checked against 15 real seed abstracts.
- `competing_ideas.py` finds papers that support or refute a hypothesis. The useful result is when both lists are non-empty: that's a live disagreement in the field.
- `continuity_checker.py` checks whether later papers really build on the earlier work they cite, or contradict it without saying so.

`competing_ideas` and `continuity_checker` have not been validated on real abstracts yet. Treat their output as a lead to check, not a finding.

## Setup

Reading mode has no special dependencies beyond Claude Code itself.

Citation-graph mode needs:
```bash
pip install chromadb sentence-transformers adapters transformers torch requests
```
Plus [Ollama](https://ollama.ai) running locally with `nomic-embed-text` pulled (`ollama pull nomic-embed-text`) — used for one half of the relevance-scoring blend (the other half is [SPECTER2](https://github.com/allenai/SPECTER2), a citation-graph-aware embedding model, loaded automatically on first use).

None of this requires any other project to be installed — it's a self-contained local setup, not a dependency on anything else.

The synthesis tools also need `qwen2.5:3b`, `qwen2.5:7b` and `qwen2.5:14b` pulled in Ollama.

## Status

In use. The citation-graph mode built the bibliography for a research paper in July 2026; it also flagged two suspect citations that turned out to be embedding false positives, which is why SKILL.md now says to check anything from a visibly different field than the seed. 175 tests. The synthesis tools are v1.

## Usage

Drop a PDF path or paper URL, or say "walk me through this paper." See [`SKILL.md`](SKILL.md) for the full command list (`/paper-dive`, `/climb`, `/challenge`, `/compare`, `/paper-graph`, etc.).

## Origin

Built as part of [MARVIN](https://github.com/G-Eskayo/marvin), a memory/skills layer for Claude Code — extracted here because it's a complete, independently useful capability on its own, not something that needs MARVIN's other infrastructure to work. If you already run MARVIN's ChromaDB instance, pointing this at the same path gets you cross-pollination with MARVIN's other memory (an optional bonus, not a requirement) — see MARVIN's README for that integration note.

## License

MIT
