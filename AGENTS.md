# GLM-5.3-Flash drafter

Target exactly `zai-org/GLM-5.3-Flash` (`glm5_next`). Source revisions and
reading paths live in `UPSTREAMS.toml`.

## Required terminology and naming

1. You can tell what a function does from its name at the call site.
2. One concept has one name everywhere.
3. Name helps me understand what else it relates to in the system and uses the
   same terms used elsewhere.

Apply these rules across code, documentation, and every session.
Use the canonical names and meanings in `CONTEXT.md`.
If an important, recurring concept is missing, add its name and a concise
definition to `CONTEXT.md` before using it.

## Working rules

Before planning work, read `README.md` for the project goal and current milestone.

- Coding: read `docs/agents/coding.md` before planning, changing, reviewing, or
  explaining project-owned code.
- Before changing target features, prompt formatting, logits, or cache behavior,
  read `docs/target-contract.md` and the referenced implementation paths.
- For DSpark changes, start with the task's paths under
  `upstream/deepspec/deepspec/modeling/dspark/`, then follow trainer or verifier
  callers. Keep the complete upstream checkout for execution.
- Default searches cover project-owned files. Inspect upstream references with
  an explicit path, for example `rg --no-ignore PATTERN reference/transformers`.
- Preserve pinned reference snapshots byte-for-byte. Intentional updates change
  `UPSTREAMS.toml`, `reference/SHA256SUMS`, and the target contract together.
- Run `python -m glm_flash_drafter` and `./run-gates.sh` from the repo root.
  The bootstrap lockfile contains development tools; the GPU environment remains
  a separate, unvalidated integration decision.
- Work in a Git worktree. If delegating, use `openai/gpt-6.1-sol`.
- Lessons: read `docs/learning/AUTHORING.md` before creating, revising, or
  reviewing lesson prose, numerical examples, or animations.

## Agent skills

### Issue tracker

Use GitHub Issues in `PhillipChaffee/glm-5.3-flash-drafter`.
Read `docs/agents/issue-tracker.md` before tracker operations.

### Triage labels

Use the standard triage roles in `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context glossary and decision records described in
`docs/agents/domain.md`.

## Evidence

Support claims about code behavior with inspected sources or executed checks.
Distinguish source inspection from execution in the result.

## Comments

- Describe the present state.
- Explain why; let code show what.
- Attach issue numbers only to live constraints.
- Display only measured or source-backed numbers.

## Human-facing writing

Run `/avoid-ai-writing` and `/simple-english` in Plain mode on human-facing
documentation or presentation deliverables so they are human consumable.
Preserve canonical terms, factual meaning, evidence limits, numbers, code,
commands, paths, links, and quoted text.
