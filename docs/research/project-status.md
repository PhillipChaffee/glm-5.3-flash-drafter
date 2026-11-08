# GLM-5.3-Flash drafter project status

- **Date:** `2026-11-07` (`2026-11-08 UTC`)
- **Tier:** `3` (4 collectors, 4 researchers, 1 synthesizer)
- **Scope:** Local repository code and tests, git commit history and stashes, all 5 git worktrees, GitHub issues `#2` through `#9` and pull requests `#1` through `#15` in `PhillipChaffee/glm-5.3-flash-drafter`, and `memex` session transcripts across `glm-5.3-flash-drafter`, `GLM-5`, and Cursor.
- **Orchestrator:** Cursor session `ab1f9724-1f21-4f17-919a-1799ef2b2457`
- **Collectors:** C1 `e36dbae7-83b9-4ad3-9200-1c95240cfa0c` (code and docs), C2 `c9e7a560-0908-40f5-aee2-f4c5a3d3332a` (git and worktrees), C3 `a8dce713-5529-42ec-810c-10a50e21d7ba` (GitHub tracker), C4 `d3889f35-1c2d-4aee-aad4-f96004c8dec9` (`memex` and transcripts)
- **Researchers:** R1 `66c411a6-2de5-47f4-b33a-e7af9c499f8b` (code, target contract, baseline, and salvage), R2 `448fb433-4bbe-4e8c-bc80-373a0ac0bc57` (git history, branches, stashes, and worktrees), R3 `5e2856b2-2283-452e-833e-877eb3b91076` (GitHub issues and PRs), R4 `98873cf7-658d-4c7e-a6e9-bc84ddee5cd0` (`memex` session archive and decision history)
- **Synthesizer:** S1 `6bc96126-8695-43d1-aa5f-8ddf973cfeef`

## Summary

`glm-5.3-flash-drafter` is in its pre-implementation contract and architecture-specification phase (`README.md:1-15`). A drafter is a trained model that proposes candidate tokens in parallel so the target model (`zai-org/GLM-5.3-Flash`) can verify them in one forward pass (`CONTEXT.md:56-62`). On `main`, the repository contains only a 4-file Python bootstrap package and its unit tests (`glm_flash_drafter/references.py:1-32`, `tests/test_references.py:1-77`, `tests/test_mhc_lesson.py:1-135`). No drafter model, data collection script, feature extractor, or training loop has been ported into `glm_flash_drafter/` yet (`docs/research/glm5-salvage-assessment.md:546-553`).

Active development paused on `2026-10-27T07:29:41Z` in OpenCode session `ses_e8d5f2ee7ffeeOZ9AdI07Wyt63` (`turn_id=553`) after the user approved Controls `27` and `28` for [Define the first dataset and per-turn training contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4). Only two read-only Cursor status runs (`3025c002-4c27-4501-a596-93fe2267e74f` on `2026-11-05` and `ab1f9724-1f21-4f17-919a-1799ef2b2457` on `2026-11-07`) have occurred in the 11 days since that session. Two operational items require action before implementation starts:

1. **Unbacked-up worktree changes:** Worktree `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract` holds 10 modified tracked files (`+742/-51` lines) and 25 untracked files (`6,655` lines by `wc -l`, totaling `7,397` added lines). Comparison against `stash@{0}` (`418a3cc`) and `stash@{1}` (`edbbcdd`) shows that 27 of those 35 dirty files and `3,677` net lines (`3,756` insertions) exist only in the live worktree and are not backed up in any git commit, stash, or pull request (`docs/handoffs/dataset-training-contract.md:533-542`).
2. **Active planning frontier:** [Choose the first drafter recipe and feature contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3) is closed with an approved 5-layer dense RoPE block-8 drafter baseline (`docs/research/first-drafter-recommendation.md:1-8`). [Define the first dataset and per-turn training contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4) has 28 approved controls and 6 remaining open items (`docs/handoffs/dataset-training-contract.md:384-411`). [Choose the training stack and compute setup](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5) is unassigned and is the sole unblocked ticket on the [Plan our first custom GLM-5.3-Flash drafter experiment](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/2) map.

## Code

### Executed verification

- **Executed:** Running `.venv/bin/python -m glm_flash_drafter` on `Python 3.12.3` exits `0` and prints `Pinned reference checksums verified.` (`glm_flash_drafter/__main__.py:1-10`).
- **Executed:** Running `.venv/bin/pytest` (`pytest-9.1.1`) on `Python 3.12.3` at commit `2727e2b` passes all 17 unit tests (`9` in `tests/test_references.py` and `8` in `tests/test_mhc_lesson.py`) in `0.13 s` with `100%` line and branch coverage (`25` statements, `0` missed, `10` branches, `0` partial) across `glm_flash_drafter/`.
- **Executed:** Running `./run-gates.sh` passes all 21 gate commands (`advisories`, `cognitive`, `dead-code`, `duplication`, `file-length`, `format`, `import-layers`, `license-check`, `link-check`, `lint`, `lockfile`, `markdown-lint`, `secret-scan`, `shell-format`, `shell-lint`, `spell-check`, `tests+coverage`, `types`, `unused-deps`, `workflow-lint`, `workflow-yaml-lint`).

### Inspected package and upstream references

- **Project-owned Python:**
  - `glm_flash_drafter/__init__.py:1` (`1` line): package docstring.
  - `glm_flash_drafter/__main__.py:1-10` (`10` lines): CLI entry point calling `verify_references(Path.cwd())`.
  - `glm_flash_drafter/references.py:1-32` (`32` lines): reads `reference/SHA256SUMS` and checks each pinned reference snapshot byte-for-byte.
  - `tests/test_references.py:1-77` (`77` lines): 9 tests covering intact checksums, modified files, missing files, empty and malformed checksum lists, CLI execution, and target contract metadata.
  - `tests/test_mhc_lesson.py:1-135` (`135` lines): 8 tests checking the worked numerical trace and animation chapter slots in `docs/learning/lessons/0001-follow-one-token-through-mhc.html`.
- **Dependencies:** `pyproject.toml:11-14` declares no runtime dependencies. `requirements.in:1-10` pins 10 development tools (`ruff`, `mypy`, `complexipy`, `pytest`, `pytest-cov`, `coverage[toml]`, `deptry`, `vulture`, `import-linter`, `mutmut`). The GPU training environment remains an open decision under [Choose the training stack and compute setup](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5) (`README.md:54-58`).
- **Pinned upstreams (`UPSTREAMS.toml:1-46`, `reference/SHA256SUMS:1-18`):**
  - `zai-org/GLM-5.3-Flash` at `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` (`reference/glm-5.3-flash/config.json`, `chat_template.jinja`, `generation_config.json`, `tokenizer_config.json`).
  - `huggingface/transformers` at `50e35cc0ca18ec19b80653690de553e6decf74f5` (`reference/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py`, `cache_utils.py`).
  - `deepseek-ai/DeepSpec` submodule at `005e03b81cec38b7da6399833d609ee89a2587f2` (`upstream/deepspec`, clean checkout). Its uninstalled upstream `requirements.txt` carries 105 OSV vulnerabilities recorded in `docs/bootstrap-verification.md:46-57`.
  - `PhillipChaffee/67-sus-95-clean` strict Python baseline at `4f982ea91585c6607bbba3a6a95da2555b88ded6`.

### Target contract (`docs/target-contract.md`)

- **Inspected architecture (`docs/target-contract.md:18-44`, `reference/glm-5.3-flash/config.json:1-288`):** `GLM-5.3-Flash` (`model_type = "glm5_next"`) is a 45-layer hybrid Mixture-of-Experts model with hidden width `4,096`, vocabulary size `154,880`, 288 routed experts (`top-8` active plus 1 shared expert after 3 dense layers), and 4-stream Manifold-Constrained Hyper-Connections (`hc_mult = 4`, widened stream state `[B, T, 4, 4096]`). Thirty-four layers are Kimi Delta Attention (`linear_attention`) layers and 11 layers (`3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43`, 0-indexed) are DeepSeek Sparse Attention layers.
- **Inspected inference-state rollback constraint (`docs/target-contract.md:45-57`, `reference/transformers/src/transformers/cache_utils.py:1107-1152`):** `LinearAttentionCacheLayerMixin.is_croppable` is `False` once `is_recurrent_states_initialized` is true. Calling `.crop()` only slices `conv_states` and cannot roll back KDA recurrent states, so rejection rollback during verification requires state checkpointing or backend-supported accepted-prefix recovery.
- **Inspected upstream configuration-key mismatch (`docs/target-contract.md:60-68`):** `reference/glm-5.3-flash/config.json:83` emits `"deepseek_sparse_attention"` in `layer_types`, while `modeling_glm5_next.py:1490` keys `causal_mask_mapping` by `"indexed_attention"` and `"linear_attention"`.

### Approved first-run drafter baseline and head-to-head comparison

[Choose the first drafter recipe and feature contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3) closed on `2026-10-24T10:56:42Z` after approving the baseline recipe in `docs/research/first-drafter-recommendation.md:1-8,150-222`:

- **Approved baseline architecture:** 5 dense layers, hidden width `4,096`, SiLU MLP intermediate size `12,288`, single-stream drafter residual streams (`hc_mult = 1`), 64 query heads and 64 key-value heads (head dimension `64`), full-head RoPE (`theta = 10,000`), bidirectional 8-row draft block (`1` anchor token + `7` copies of internal draft mask token `154856`) proposing `8` tokens per block, and a `2,048`-position target context window (`a - 2048 <= t < a`).
- **Approved feature set and heads:** 6 unweighted 4-stream mean target context features `[M20, M28, M32, M36, M40, M44]` concatenated and projected (`24,576 -> 4,096` + `RMSNorm`), separate raw boundary-45 mean `M45` normalized once by frozen target `model.norm` to form the teacher state and projected once by frozen `lm_head`, rank-256 vanilla Markov head, and a confidence head (`4,352 -> 1`).
- **Approved training loss:** `0.1 * CE(teacher_argmax) + 0.9 * TV(p_target, p_draft) + 1.0 * BCE(confidence, detach(1 - TV))` with exponential position decay `w_j = exp(-j / 4)`.

| Metric | Approved baseline (this repo) | Red Hat public DSpark (`dspark-preview`) | Old custom GLM-5 port (`5e87e16d`) | Public DFlash2 (`incoai`) |
| --- | --- | --- | --- | --- |
| Proposed draft tokens per block | `8` (`1` anchor + `7` masks) | `8` (`1` anchor + `7` masks) | `7` (`1` anchor + `6` masks) | `15` (`16`-row block) |
| Drafter layers and position encoding | `5` dense, full-head RoPE | `5` dense, full-head RoPE | `5` dense, NoPE | `5` dense, DFlash2 block diffusion |
| Context features in feature set | `6` (`M20,28,32,36,40,44`) | `6` (`20,28,32,36,40,44`) | `7` (`1,12,23,34,42,43,44`) | Not published in repo docs |
| Training examples completed | `0` (not yet trained) | `1,723,733` (`3` passes) | `686` (`20`-step smoke run) | Not published in repo docs |
| Greedy verification progress (`T=0` 4-task mean) | Not measured | `3.77` tokens/step (`34.67%` accept) | Not measured (`1.05` on 12 prompts) | Not measured |
| Greedy verification progress on HumanEval (`T=0`) | Not measured | `4.74` tokens/step (`4.46` at pass 1) | Not measured | Not measured |
| Sampled verification progress (`T=1.0`, 6-task `c1`) | Not measured | Not measured | Not measured (`1.01` on 12 prompts) | `4.4044` tokens/step (`5.599` HumanEval) |
| Serving speedup at concurrency `1` (`TP4` 4x B300) | Not measured | Not measured | Not measured | `1.573x` vs plain (`1.194x` vs MTP) |

Sources: `docs/research/first-drafter-recommendation.md:150-177`, `docs/research/first-drafter-design-comparison.md:54-215`, `docs/research/glm5-salvage-assessment.md:218-266`, `docs/research/drafter-data-scaling.md:146-162`, and `docs/research/configuration-pilot-plan.md:303`.

### Predecessor `GLM-5` salvage and Modal volumes

- **Inspected salvage inventory (`docs/research/glm5-salvage-assessment.md:45-143,546-553`, `docs/research/artifact-inventory.md:31-163`):** The predecessor repository (`/Users/phillipchaffee/git/GLM-5` at `5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18`) contains four reusable code groups:
  1. The `GLM-5.3-Flash` DSpark model registration and 4-stream mean hook adapter (`deepspec/modeling/dspark/glm5/`, requiring adaptation to RoPE, 6 context features, separate `M45`, and 8-row block width).
  2. `token_budget_batcher.py`, tested across `140,616` CPU cases, plus feature-cache preparation and slice merge scripts.
  3. The SGLang `GLM-5.3-Flash` verification patch (`glm5_dspark.py` and `dspark_bringup.py`), tested on `12` live greedy and sampled prompts (`1,084` verification rounds).
  4. Corrected `ELSI` and `ELSI-C` feature-selection scripts (`elsi_analyze.py`, `elsi_capture_glm.py`).
- **Inspected Modal storage (`docs/research/artifact-inventory.md:31-163`):** Three Modal volumes hold `1.261 TB` (`1,260,975,637,568` bytes): `glm-drafter-run` (`817.56 GB`, including `698` regenerated conversations, a `686`-sample feature cache of `34.73 GB`, and 6 step-20 training checkpoint bundles of `66.90 GB` each), `glm-serve-weights` (`397.65 GB`, holding target-model weights, step-20 serving weights, and DFlash2 weights), and `dspark-proxy-run` (`45.76 GB`, holding Qwen3-4B proxy runs). None of these files or volumes have been migrated or approved for reuse under [Select prior code and artifacts for the first experiment](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/6) yet.

## Branches and worktrees

### Local branches and remote divergence

- **Inspected commit graph:** All 5 local branches on the workstation (`main`, `4-dataset-training-contract`, `mhc-numerical-walkthrough`, `repository-setup-playbook`, and `software-factory-prompts`) plus 2 git stashes contain 23 commit objects in total (19 non-stash commits).
- **Inspected remote divergence:** Workstation `main` and `origin/main` point to `0120409b5003a187ac760391345a630fcf1fdc5a` (`2026-10-24 18:58:39 -0700`, merge of [Add coding guidelines from software factory notes](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/14)). Worktree `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/mhc-numerical-walkthrough` is clean at commit `7e2fdd5fc54015d9bb3f469ad07e0586cf1af915` (`2026-10-24 19:56:08 -0700`), which merged into GitHub `main` via [Explain mHC with a numerical walkthrough and video](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/15) as commit `2727e2b9b323b8cbd336d658a04384bcd2b23ba9` on `2026-10-25T03:10:03Z`. Because `git fetch` did not run on the workstation after `2026-10-24 19:33:07 -0700` (`.git/FETCH_HEAD`), workstation `main` is 1 merge behind remote GitHub `main`. Zero committed local branches contain unmerged work.

### Worktree inventory and stash divergence

| Worktree path | Branch and HEAD | Working tree state | Backup status |
| --- | --- | --- | --- |
| `/Users/phillipchaffee/git/glm-5.3-flash-drafter` | `main` (`0120409`) | `0` modified tracked files, `3` untracked files (`docs/research/project-status.md`, `opencode.json` 11 lines, `uv.lock` 8 lines) | Tracked tree matches `0120409` |
| `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/mhc-numerical-walkthrough` | `mhc-numerical-walkthrough` (`7e2fdd5`) | Clean (`0` modified, `0` untracked) | Merged into remote `main` via [PR #15](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/15) (`2727e2b`) |
| `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/software-factory-prompts` | `software-factory-prompts` (`5905904`) | Clean (`0` modified, `0` untracked) | Merged into `main` via [PR #14](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/14) (`0120409`) |
| `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract` | `4-dataset-training-contract` (`0120409`) | `10` modified tracked files (`+742/-51` lines) and `25` untracked files (`6,655` lines by `wc -l`, `7,397` total added lines) | **At risk:** `27` of `35` files and `3,677` net lines are absent from both git stashes |
| `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/repository-setup-playbook` | `repository-setup-playbook` (`0120409`) | `0` modified tracked files, `5` untracked files (`681` lines across `skills/setup-repository/` and `docs/research/`) | Uncommitted locally (`PhillipChaffee/.agents` PR `#24` opened in `ses_e8f0919f6ffedBK9l9t8PV1zlA`) |

- **Inspected `4-dataset-training-contract` contents:**
  - **10 modified tracked files (`+742/-51` lines):** `CONTEXT.md` (`+419/-2`, expanding the glossary from 238 to 655 lines), `docs/learning/NOTES.md` (`+125/-0`), `docs/learning/guides/drafter-config.html` (`+98/-37`), `docs/learning/RESOURCES.md` (`+62/-2`), `.gitleaks.toml` (`+11/-3`), `docs/target-contract.md` (`+9/-0`), `.typos.toml` (`+8/-0`), `docs/research/first-drafter-design-comparison.md` (`+6/-4`), `README.md` (`+3/-0`), and `docs/research/first-drafter-recommendation.md` (`+2/-2`).
  - **25 untracked files (`6,655` lines by `wc -l`):**
    - `docs/handoffs/dataset-training-contract.md` (`646` lines): master handoff mapping Controls `1` through `28`, remaining open questions, and stash warnings.
    - `docs/research/token-record-and-recovery-contract.md` (`567` lines) and `docs/research/token-record-and-recovery/check-contract.sh` (`386` lines): executable contract checker for Controls `27` and `28`.
    - 13 additional research reports under `docs/research/`: `open-dataset-selection.md` (`458` lines), `dataset-generation-and-mixture-controls.md` (`455` lines), `thinking-mode-drafter-acceptance.md` (`428` lines), `configuration-pilot-plan.md` (`415` lines), `openperfectblend-conversation-policy.md` (`392` lines), `drafter-data-scaling.md` (`362` lines), `elsi-evaluation-review.md` (`341` lines), `training-data-temperature-and-size.md` (`319` lines), `opb-components-and-nemotron-license.md` (`315` lines), `hosted-provider-qualification.md` (`312` lines), `dspark-data-and-reasoning.md` (`260` lines), `hosted-generation-and-feature-capture.md` (`214` lines), and `glm-training-and-workload-focus.md` (`175` lines).
    - 2 HTML learning pages and 6 Python example scripts (`999` lines) plus `opencode.json` (`11` lines): `docs/learning/guides/pilot-controls-and-metrics.html` (`469` lines), `docs/learning/lessons/0002-score-a-saved-response.html` (`156` lines), `pilot_examples.py` (`83` lines), `collection_examples.py` (`71` lines), `configuration_examples.py` (`70` lines), `optimizer_example.py` (`63` lines), `feature_projection_example.py` (`44` lines), and `replay_example.py` (`43` lines).
- **Inspected stash comparison (`stash@{0}` `418a3cc` and `stash@{1}` `edbbcdd` from `2026-10-24`):** Eleven of the 35 dirty files (`2` tracked and `9` untracked totaling `2,635` untracked lines, or `2,624` lines excluding `opencode.json`) do not exist in either stash. Sixteen additional files present in `stash@{0}` were modified between `2026-10-24` and `2026-10-27` by `+1,101/-79` lines (`+1,022` net lines). Dropping or resetting the `4-dataset-training-contract` worktree would destroy `3,677` net lines of unbacked-up contract and research work (`docs/handoffs/dataset-training-contract.md:533-542`).

## GitHub issues

### Issue tracker overview (`PhillipChaffee/glm-5.3-flash-drafter`)

The tracker has 7 issues (`1 CLOSED`, `6 OPEN`, issue numbers `#2` through `#7` and `#9`) with 54 total issue comments, and 8 merged pull requests (`#1, #8, #10, #11, #12, #13, #14, #15`). All 8 merged pull requests added documentation, governance rules, or Lesson 01 (`mHC`). Zero runtime code PRs have been opened since PR `#1`. The last GitHub tracker activity occurred on `2026-10-27T07:29:00Z`.

| Issue | Title | State | Assignee | Comments | Blockers and status |
| --- | --- | --- | --- | --- | --- |
| [#2](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/2) | Wayfinder map: plan our first custom GLM-5.3-Flash drafter experiment | `OPEN` | Unassigned | `4` | Tracks `#3, #4, #5, #6, #9, #7` (`1` of `6` closed) |
| [#3](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3) | Choose the first drafter recipe and feature contract | `CLOSED` | `PhillipChaffee` | `8` | Closed `2026-10-24T10:56:42Z` with approved 5-layer RoPE block-8 6-feature baseline |
| [#4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4) | Define the first dataset and per-turn training contract | `OPEN` | `PhillipChaffee` | `29` | Controls `1` to `28` approved (`2026-10-27T07:29:00Z`), `6` open items remain |
| [#5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5) | Choose the training stack and compute setup | `OPEN` | Unassigned | `5` | **Sole unblocked frontier ticket.** Must satisfy Controls `27` and `28` |
| [#6](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/6) | Select prior code and artifacts for the first experiment | `OPEN` | Unassigned | `3` | Blocked by `#4`. Audits `GLM-5` code and `1.261 TB` Modal volumes |
| [#9](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/9) | Define the broad-capture and ELSI comparison protocol | `OPEN` | Unassigned | `2` | Blocked by `#4` and `#5`. Specifies `45`-tap (`M1..M45`) capture pool and `6-tap` vs `S6` vs `S7` |
| [#7](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/7) | Define the first-run protocol and evidence | `OPEN` | Unassigned | `3` | Blocked by `#5`, `#6`, and `#9`. Requires live pre-collection qualification |

### Issue #4 approved controls (`1` through `28`) and remaining blockers

Across 29 comments on [Define the first dataset and per-turn training contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4), the user approved 28 binding controls in 7 decision rounds (`docs/handoffs/dataset-training-contract.md:56-383`):

- **Controls 1 to 4 ([comment 6396369039](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6396369039), `2026-10-25`):** Use `mlabonne/open-perfectblend` (`af60f3c18201652a83a93f46fcfee1b646ba3df7`, `1,420,909` rows), English-first scope, target mixture `40% code / 30% maths / 20% other STEM / 10% chat/instructions`, and native multi-turn plus single-turn prompts with sequential target-regenerated assistant replies and current-turn-only loss supervision.
- **Controls 5 to 8 ([comment 6397222203](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6397222203), `2026-10-25`):** Generate replies at `temperature = 1.0`, `top_p = 0.95`, `reasoning_effort = max`, and `clear_thinking = false`, with 1 regenerated version per conversation and response-example counting for the pilot mixture.
- **Controls 9 to 12 ([comment 6397857074](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6397857074), `2026-10-25`):** Classify category from the user request, require both an English prompt and an intended English reply, require self-contained text context, and use a combined manual pilot annotation and follow-up review.
- **Controls 13 to 16 ([comment 6397965477](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6397965477), `2026-10-25`):** Apply exact whole-conversation deduplication, prompt-family grouping so related prompts never cross splits, `80 / 10 / 10` train/validation/test family splits, and a deterministic `seed = 42` split manifest.
- **Controls 17 to 20 ([comment 6398063289](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6398063289), `2026-10-25`):** Impose no fixed single-turn vs multi-turn quota and no turn-count cap, keep only complete responses in the initial training control, and cap transient error retries at `<= 2`.
- **Controls 21 to 24 ([comment 6404497538](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6404497538), `2026-10-25`):** Enforce generation-pilot limits of `65,536` input tokens, `max_tokens = 32,768` output per attempt, `50` attempts, `500,000` accounted output tokens, and a `2-hour` wall-clock deadline with pre-launch `32,768`-token budget reservations.
- **Controls 25 and 26 ([comment 6434100386](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6434100386), `2026-10-27`):** Self-host pinned `zai-org/GLM-5.3-Flash` (`46a050b`) on Modal for target-model generation, retiring third-party hosted endpoints (`Munito`, `OrcaRouter`, and `OpenCode Go`).
- **Controls 27 and 28 ([comment 6436010406](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4#issuecomment-6436010406), `2026-10-27`):** Require exact unpadded engine prompt and completion token IDs, positional handling of stop and padding tokens (`154820` `<|endoftext|>`, `154827` `<|user|>`, `154829` `<|observation|>`), explicit cumulative-vs-delta stream assembly (forbidding retokenized text fallback), and a single-owner persistent `Collection journal` with pre-decode attempt reservation and per-turn commit durability.
- **Recorded contract verification (`check-contract.sh`):** In session `ses_e8d5f2ee7ffeeOZ9AdI07Wyt63`, `docs/research/token-record-and-recovery/check-contract.sh` passed on `Python 3.12.13` (`torch 2.9.0`, `tokenizers 0.22.1`, `Jinja2 3.1.6`) and proved that the predecessor `GLM-5` script `generate_train_data_glm.py` (`1356aee`) still fails Controls `27` and `28` on 4 defects: error rows counted as completed, malformed interior rows skipped, unchecked source-content identity, and second-turn failures overwriting the first turn's reply while deleting `_usage` from `errors.jsonl` (`docs/research/token-record-and-recovery-contract.md:256-312,441-534`).
- **Six remaining open items blocking Issue #4 closure (`docs/handoffs/dataset-training-contract.md:384-411`):**
  1. Training-pilot and full training-data volume ceilings and training sequence-length cutoffs.
  2. Feature-cache storage limits (`57,344` bytes per token position in `bf16` for `6` context features plus `M45`).
  3. Full-corpus category weighting and automated bulk semantic filtering beyond the `50`-attempt manual generation pilot.
  4. Disposition of the historical `GLM-5` Modal dataset (`698` conversations and `686` cached feature samples).
  5. Artifact access, retention, and publication policy.
  6. Holistic user sign-off on the complete assembled contract.

### Issue #5 decision scope (sole unblocked frontier)

[Choose the training stack and compute setup](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5) ([comment 6436052389](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5#issuecomment-6436052389), `2026-10-27T07:16:19Z`) establishes that Modal is approved for target-model response generation only. The ticket must resolve:

1. **Serving and extraction runtime:** Choose among `vLLM`, `SGLang`, and offline `Transformers` on Modal. Any candidate runtime that cannot satisfy Controls `27` and `28` (exact unpadded token IDs and durable per-turn journal recovery) is rejected before speed or cost comparisons.
2. **Durable storage backend:** `Modal Volume v1` is the provisional baseline (`500,000`-inode hard limit and no distributed lock, requiring a single-writer journal owner), whereas `Modal Volume v2` warns of potential data loss.
3. **Training compute and hyperparameters:** Select GPU type, tensor/data parallelism (`TP`/`DP`), precision (`bf16` vs `FP8`), optimizer, learning-rate schedule, token-budget batch size, and hard training spend caps.

## Prior sessions (Memex)

### Session inventory

`memex` (`0.33.0`, lexical FTS index active, semantic vector index not built) records:

- **`glm-5.3-flash-drafter` OpenCode sessions (`137` total: `11` main + `126` fork subagents):**
  - `125` sessions (`9` main + `116` fork) under `/Users/phillipchaffee/git/glm-5.3-flash-drafter`.
  - `12` sessions (`2` main, including `ses_e8d5f2ee7ffeeOZ9AdI07Wyt63`, + `10` fork) under `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract`.
- **Predecessor `GLM-5` OpenCode sessions (`188` total: `46` main + `142` fork subagents)** under `/Users/phillipchaffee/git/GLM-5` from `2026-09-28` to `2026-10-24`.
- **Cursor transcripts (`2` main sessions):** `3025c002-4c27-4501-a596-93fe2267e74f` (`2026-11-06 UTC` status audit) and `ab1f9724-1f21-4f17-919a-1799ef2b2457` (`2026-11-08 UTC` current Tier 3 synthesis).

### Key OpenCode main sessions in `glm-5.3-flash-drafter`

| Session ID | Date (UTC) | Messages | Primary outcome |
| --- | --- | --- | --- |
| `ses_e941eff34ffeW6LdneqtAhmwNY` | `2026-10-24` | `406` | Bootstrapped repository (`2544afc`), audited `GLM-5` salvage and Modal volumes, merged `PR #1` and `PR #8`, and charted Wayfinder map `#2` |
| `ses_e92f3ca9cffeC3bTVOsjoN3zxf` | `2026-10-24` | `353` | Compared Red Hat `dspark-preview`, DeepSeek V4 DSpark, and old `GLM-5` port, closed [Issue #3](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3), created [Issue #9](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/9), and merged `PR #10` |
| `ses_e923c39e7ffefZWbNBwem5GBTX` | `2026-10-24` to `2026-10-25` | `753` | Claimed [Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4) in worktree `4-dataset-training-contract`, approved Controls `1` to `4`, and handed off at Controls `5` to `8` |
| `ses_e904aac24ffeK8rpoOuxLlzFcK` & `ses_e8f3e716dffecwHMlPI3Mr5zte` | `2026-10-24` to `2026-10-25` | `558` / `292` | Authored Lesson 01 (`mHC` numerical walkthrough and video) and merged `PR #11`, `PR #12`, `PR #13`, and `PR #15` (`2727e2b`) |
| `ses_e8f4b6e31ffeE53hLfyg3ufXq1` | `2026-10-25` | `98` | Added `docs/agents/coding.md` and `AGENTS.md` software-factory rules, merged `PR #14` (`0120409`) |
| `ses_e8f0919f6ffedBK9l9t8PV1zlA` | `2026-10-25` | `186` | Built `setup-repository` skill in `repository-setup-playbook` worktree (`5/5` vs `0/5` subagent validation) and opened `PhillipChaffee/.agents` PR `#24` |
| `ses_e8d5f2ee7ffeeOZ9AdI07Wyt63` | `2026-10-25` to `2026-10-27` | `554` | Conducted 6 decision rounds on [Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4) (Controls `5` to `28`), built `check-contract.sh`, and updated `docs/handoffs/dataset-training-contract.md` |

### Superseded decisions that must not be revived

Cross-session review in `ses_e92f3ca9cffeC3bTVOsjoN3zxf`, `ses_e923c39e7ffefZWbNBwem5GBTX`, and `ses_e8d5f2ee7ffeeOZ9AdI07Wyt63` explicitly retired the following earlier proposals (`docs/handoffs/dataset-training-contract.md:88-107,312-383`):

1. **Phone-economics and production adoption gate:** Replaced by a learning-first research goal (`README.md:1-15`).
2. **Nemotron Post-Training Dataset v2 as primary corpus:** Replaced by `mlabonne/open-perfectblend` (`af60f3c`, Control `1`).
3. **Earlier `40% code / 30% chat / 20% maths / 10% STEM` mix:** Replaced by `40% code / 30% maths / 20% other STEM / 10% chat/instructions` (Control `3`).
4. **Single-turn-only supervision and `4,096 / 8,192 / 250,000` pilot caps:** Replaced by native multi-turn + single-turn prompts with sequential regenerated replies (Control `4`) and `65,536` input / `32,768` output / `50`-attempt / `500,000`-token pilot caps (Controls `21` to `24`).
5. **Third-party hosted endpoints (`Munito`, `OrcaRouter`, `OpenCode Go`):** Replaced by Modal self-hosting of pinned `zai-org/GLM-5.3-Flash` (`46a050b`, Controls `25` and `26`).
6. **Greedy (`temperature = 0`) or `temperature = 0.7` generation, `reasoning_effort = low`, and `clear_thinking = true`:** Replaced by `temperature = 1.0, top_p = 0.95, reasoning_effort = max, clear_thinking = false` (Controls `5` and `6`).
7. **Retokenized-text fallback, value-based `154820` stop/padding stripping, and line-count resume:** Replaced by exact engine token IDs, positional stop/padding slicing, and the single-owner `Collection journal` (Controls `27` and `28`).
8. **Predecessor `GLM-5` NoPE, 7-row block, 7-feature, full-L1 drafter recipe:** Replaced by the [Issue #3](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3) 5-layer RoPE, 8-row block, 6-feature (`M20..M44` + separate `M45`), `0.1 CE + 0.9 TV + 1.0 BCE` recipe.

## Timeline

- **`2026-09-28` to `2026-10-24` (Memex `GLM-5`):** Predecessor `GLM-5` repository ran 188 OpenCode sessions, measuring `DFlash2` (`4.4044` mean tokens/step, `1.573x` vs plain at `c1`), collecting `698` conversations (`686` cached feature samples) on Modal, and running a 20-step smoke test (`1.05` greedy tokens/step on 12 SGLang prompts) (`docs/research/glm5-salvage-assessment.md:17-66,218-266`).
- **`2026-10-23` (Git `2544afc`):** Bootstrapped `glm-5.3-flash-drafter`, pinned `UPSTREAMS.toml` and `reference/SHA256SUMS`, documented `docs/target-contract.md`, and checked all 21 gate commands on `Python 3.12.13` (`docs/bootstrap-verification.md:14-16`).
- **`2026-10-24` to `2026-10-25` (Git `7dcc555` to `2727e2b`, PRs `#1, #8, #10–#15`, Issue `#3`):** Completed salvage and baseline research, closed [Issue #3](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3) with the approved 5-layer RoPE block-8 baseline, approved Controls `1` through `24` on [Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4), and merged Lesson 01 (`mHC`) in [PR #15](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/15) (`2727e2b`).
- **`2026-10-27` (Memex `ses_e8d5f2ee7ffeeOZ9AdI07Wyt63`, Issue `#4` and `#5`):** Approved Controls `25` through `28` on [Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4), built and ran `check-contract.sh`, updated [Issue #5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5) with Controls `27` and `28` runtime gates, and updated `docs/handoffs/dataset-training-contract.md`. Active development stopped at `2026-10-27T07:29:41Z`.
- **`2026-11-05` and `2026-11-07` (Cursor `3025c002` and `ab1f9724`):** Read-only Tier 3 status audits checked repository state, worktree diffs, GitHub issues, and `memex` history.

## Risks

- **Data-loss exposure in `4-dataset-training-contract`:** Twenty-seven of the 35 dirty files and `3,677` net lines (`3,756` insertions) in `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract` exist nowhere in git commits, stashes, or GitHub PRs (`docs/handoffs/dataset-training-contract.md:533-542`).
- **Stale workstation `main` branch:** Workstation `main` (`0120409`) lacks merged [PR #15](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/15) (`2727e2b`) because `git fetch` has not run on the workstation since `2026-10-24`. Branching from `0120409` without fetching omits the `mHC` numerical walkthrough and causes avoidable merge conflicts.
- **Unqualified historical data generator:** `check-contract.sh` proved that the predecessor `GLM-5` generator (`generate_train_data_glm.py` at `1356aee`) violates Controls `27` and `28` on 4 token-fidelity and recovery bugs (`docs/research/token-record-and-recovery-contract.md:256-312,441-534`). Reusing that script or its `686` cached samples without remediation and [Issue #6](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/6) qualification corrupts training supervision.
- **Modal storage and feature-cache scaling:** Storing 6 context features plus `M45` in `bf16` consumes `57,344` bytes per token position (`~57.34 GB` per million tokens), while `Modal Volume v1` enforces a `500,000`-inode limit and lacks distributed locking ([Issue #5 comment 6435782911](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5#issuecomment-6435782911)).

## Open questions

1. **Serving and extraction runtime ([Issue #5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5)):** Which Modal execution path (`vLLM`, `SGLang`, or offline `Transformers`) satisfies Controls `27` and `28` while supporting hidden-state extraction for `[M20, M28, M32, M36, M40, M44]` and `M45`?
2. **Training-pilot and feature-cache ceilings ([Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4), [Issue #5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5)):** What token-volume ceiling, sequence-length cap, and shard format keep the `57,344`-byte/position feature cache within Modal storage and inode limits?
3. **Historical Modal data disposition ([Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4), [Issue #6](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/6)):** Should the `698` historical `GLM-5` conversations (`686` cached samples in `glm-drafter-run`) be restricted to plumbing smoke checks because they were generated under superseded `reasoning_effort=low` and `clear_thinking=true` rules?

## Next steps

1. **Commit and push the `4-dataset-training-contract` worktree:** In `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract`, stage the 10 modified tracked files and 24 project untracked files (excluding `opencode.json`), commit them on branch `4-dataset-training-contract`, and open a pull request referencing [Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4).
2. **Fetch and fast-forward workstation `main`:** In `/Users/phillipchaffee/git/glm-5.3-flash-drafter`, run `git fetch origin` and fast-forward `main` from `0120409` to `2727e2b` ([PR #15](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/15)), then clean up the merged `mhc-numerical-walkthrough` and `software-factory-prompts` worktrees.
3. **Claim and resolve [Issue #5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5):** Assign [Issue #5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5), audit `vLLM`, `SGLang`, and `Transformers` on Modal against Controls `27` and `28`, select the durable storage layout (`Modal Volume v1` single-writer shard layout), and lock GPU, precision, and spend caps.
4. **Complete the last 6 items on [Issue #4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4):** Using the [Issue #5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5) storage numbers, finalize training-pilot volume caps, feature-cache limits, bulk filtering rules, historical artifact disposition, and holistic user approval to close `#4` and unblock `#6` and `#9`.

## Sources

1. **Executed and Inspected (`66c411a6-2de5-47f4-b33a-e7af9c499f8b`, `e36dbae7-83b9-4ad3-9200-1c95240cfa0c`):** `README.md:1-92`, `CONTEXT.md:1-238`, `UPSTREAMS.toml:1-46`, `pyproject.toml:1-189`, `requirements.in:1-10`, `requirements-lock.txt:1-6`, `run-gates.sh:1-66`, `glm_flash_drafter/__init__.py:1`, `glm_flash_drafter/__main__.py:1-10`, `glm_flash_drafter/references.py:1-32`, `tests/test_references.py:1-77`, `tests/test_mhc_lesson.py:1-135`, `docs/bootstrap-verification.md:1-67`, `docs/target-contract.md:1-87`, `docs/research/first-drafter-recommendation.md:1-297`, `docs/research/first-drafter-design-comparison.md:1-278`, `docs/research/glm5-salvage-assessment.md:1-553`, `docs/research/artifact-inventory.md:1-273`, `docs/research/drafter-data-scaling.md:1-362`, and `docs/research/configuration-pilot-plan.md:1-415`.
2. **Inspected Git and Worktrees (`448fb433-4bbe-4e8c-bc80-373a0ac0bc57`, `c9e7a560-0908-40f5-aee2-f4c5a3d3332a`):** Local git refs (`0120409b5003a187ac760391345a630fcf1fdc5a`, `7e2fdd5fc54015d9bb3f469ad07e0586cf1af915`), stashes (`stash@{0}` `418a3cc`, `stash@{1}` `edbbcdd`), `.git/FETCH_HEAD`, worktree `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract` (`docs/handoffs/dataset-training-contract.md:1-646`, `docs/research/token-record-and-recovery-contract.md:1-567`, `docs/research/token-record-and-recovery/check-contract.sh:1-386`), and worktree `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/repository-setup-playbook` (`skills/setup-repository/SKILL.md:1-74`, `docs/research/setup-repository-validation.md:1-194`).
3. **Inspected GitHub Tracker (`5e2856b2-2283-452e-833e-877eb3b91076`, `a8dce713-5529-42ec-810c-10a50e21d7ba`):** GitHub Issues [#2](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/2), [#3](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3), [#4](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4), [#5](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5), [#6](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/6), [#7](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/7), [#9](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/9), and merged Pull Requests [#1](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/1), [#8](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/8), [#10](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/10), [#11](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/11), [#12](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/12), [#13](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/13), [#14](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/14), and [#15](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/pull/15) (`2727e2b9b323b8cbd336d658a04384bcd2b23ba9`).
4. **Memex and Transcript Archive (`98873cf7-658d-4c7e-a6e9-bc84ddee5cd0`, `d3889f35-1c2d-4aee-aad4-f96004c8dec9`, `6bc96126-8695-43d1-aa5f-8ddf973cfeef`):** OpenCode main sessions `ses_e941eff34ffeW6LdneqtAhmwNY`, `ses_e92f3ca9cffeC3bTVOsjoN3zxf`, `ses_e923c39e7ffefZWbNBwem5GBTX`, `ses_e904aac24ffeK8rpoOuxLlzFcK`, `ses_e8f3e716dffecwHMlPI3Mr5zte`, `ses_e8f4b6e31ffeE53hLfyg3ufXq1`, `ses_e8f0919f6ffedBK9l9t8PV1zlA`, and `ses_e8d5f2ee7ffeeOZ9AdI07Wyt63` (`turn_id=553`, closeout subagents `ses_e83c19465ffeaaFBiXgIW6idUR` and `ses_e83c15561ffeP2X4K5vAft1Wi9`), plus Cursor transcripts `3025c002-4c27-4501-a596-93fe2267e74f` and `ab1f9724-1f21-4f17-919a-1799ef2b2457`.
