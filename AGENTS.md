# GLM-5.3-Flash drafter

Target exactly `zai-org/GLM-5.3-Flash` (`glm5_next`). Source revisions and
reading paths live in `UPSTREAMS.toml`.

Before planning work, read `README.md` for the project goal and current milestone.

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

## Agent skills

### Issue tracker

Use GitHub Issues in `PhillipChaffee/glm-5.3-flash-drafter`.
Read `docs/agents/issue-tracker.md` before tracker operations.

### Triage labels

Use the standard triage roles in `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context glossary and decision records described in
`docs/agents/domain.md`.

## Comments

- Describe the present state.
- Explain why; let code show what.
- Attach issue numbers only to live constraints.
- Display only measured or source-backed numbers.
