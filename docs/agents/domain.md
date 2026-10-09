# Domain Docs

**Layout**: Single-context

## Single-context (most repos)

- `CONTEXT.md` at the repo root — the domain glossary. Devoid of implementation details; terms
  and their meanings only.
- `docs/adr/` at the repo root — one file per architectural decision, numbered sequentially.

## Consumer rules

- `improve-codebase-architecture`, `diagnose`, and `tdd` read `CONTEXT.md` for domain vocabulary
  before proposing names for new modules/concepts — a suggestion using a term not in `CONTEXT.md`
  should either match existing language or trigger adding the new term, not invent parallel
  vocabulary.
- These skills read `docs/adr/` for past decisions in the area being touched, and should not
  re-litigate a settled decision without flagging that they're doing so and why.
- If `CONTEXT.md`/`docs/adr/` don't exist yet, that's fine — `grill-with-docs` creates them
  lazily, only when the first term/decision actually resolves. Don't scaffold empty files
  preemptively.
