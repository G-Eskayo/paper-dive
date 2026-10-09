# Issue Tracker

**Type**: GitHub Issues
**Repo**: G-Eskayo/paper-dive (from `git remote -v`)
**CLI**: `gh` (must be authenticated — `gh auth status` to check)

## Conventions

- Create an issue: `gh issue create --title "..." --body "..." [--label ...]`
- Read an issue: `gh issue view <number>`
- Comment on an issue: `gh issue comment <number> --body "..."`
- List open issues: `gh issue list`
- Apply/remove labels: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`

## Consumer rules

- `to-issues` creates new issues via `gh issue create`; it does not assume any project-board
  automation exists — if this repo uses one, note it here so the skill knows to also add the
  issue to the right project/column.
- `triage` reads open issues via `gh issue list` and applies labels from
  `docs/agents/triage-labels.md` — it does not invent label names not defined there.
- Never force-push, close, or delete issues without explicit user confirmation — these skills
  only create/comment/label by default.
