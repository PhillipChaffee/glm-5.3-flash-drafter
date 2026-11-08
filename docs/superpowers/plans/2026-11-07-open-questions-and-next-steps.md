# Open Questions and Next Steps Walkthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect uncommitted workstation worktrees, resolve the open decision questions on GitHub Issues `#5` and `#4` four decisions at a time, and close the remaining Wayfinder planning tickets (`#6`, `#9`, `#7`) so implementation of the first `zai-org/GLM-5.3-Flash` drafter experiment can begin.

**Architecture:** We first eliminate the data-loss risk by syncing workstation `main` to `2727e2b` and committing the 34 project-owned dirty files in the `4-dataset-training-contract` worktree. Next, we resolve the sole unblocked Wayfinder frontier ticket ([Choose the training stack and compute setup](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5)) and the 6 remaining open items on [Define the first dataset and per-turn training contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4) using the project's four-decisions-per-exchange review protocol. Finally, we resolve the downstream tickets (`#6`, `#9`, `#7`) and transition into `/to-spec`, `/to-tickets`, and `/implement`.

**Tech Stack:** Git worktrees, GitHub CLI (`gh`), Python `3.12+`, `./run-gates.sh` (21 static, test, and hygiene gates), `docs/research/token-record-and-recovery/check-contract.sh` (`torch==2.9.0`, `tokenizers==0.22.1`, `Jinja2==3.1.6`), and Modal (`modal==1.6.2`).

## Global Constraints

- Target model: `zai-org/GLM-5.3-Flash` (`model_type = "glm5_next"`) pinned at `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` in `UPSTREAMS.toml`.
- Pinned references: `deepseek-ai/DeepSpec` at `005e03b81cec38b7da6399833d609ee89a2587f2` (`upstream/deepspec`), `huggingface/transformers` at `50e35cc0ca18ec19b80653690de553e6decf74f5` (`reference/transformers`), and `PhillipChaffee/67-sus-95-clean` at `4f982ea91585c6607bbba3a6a95da2555b88ded6`. Preserve `reference/SHA256SUMS` byte-for-byte.
- Approved drafter baseline ([Choose the first drafter recipe and feature contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3)): 5 dense layers, hidden width `4,096`, gated SiLU MLP width `12,288`, single-stream residual streams, 64 query heads, 64 key-value heads, head dimension `64`, full-head RoPE (`theta = 10,000`), bidirectional 8-row block (`1` anchor + `7` copies of internal mask token `154856`) proposing `8` tokens, `2,048`-position target context window (`a - 2048 <= t < a`), 6 unweighted 4-stream mean context features `[M20, M28, M32, M36, M40, M44]` (`24,576 -> 4,096` + `RMSNorm`), separate raw boundary-45 mean `M45` normalized once by frozen target `RMSNorm` to form the teacher state before frozen `lm_head`, rank-256 vanilla Markov head, confidence head (`4,352 -> 1`), and loss `0.1 * CE(teacher_argmax, q) + 0.9 * TV(p, q) + 1.0 * BCE(confidence, detach(1 - TV))` with position weights `exp(-j / 4)`.
- Capability-first generation rule ([Define the first dataset and per-turn training contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4) Controls `27` and `28`): any target-model generation method that cannot supply exact unpadded engine prompt and completion token IDs, positional stop/padding handling (`154820`, `154827`, `154829`), retained reasoning (`clear_thinking = false`, `reasoning_effort = max`), and a single-owner persistent collection journal with pre-decode `32,768`-token attempt reservations is rejected before cost or speed comparison. Retokenized parsed text is forbidden as a fallback.
- Naming and terminology (`AGENTS.md`, `CONTEXT.md`): one concept has one name everywhere; you can tell what a function does from its name at the call site; add any new recurring domain concept to `CONTEXT.md` before using it.
- Human-facing writing (`AGENTS.md`): apply `/avoid-ai-writing` and `/simple-english` in Plain mode (active voice, simple tenses, no contractions, no semicolons, no em-dashes).
- Quality gates (`run-gates.sh`, `pyproject.toml`): Python `>=3.12`, branch-inclusive test coverage `>=95%`, max cognitive complexity `15`, max effective lines `1,000`, and all 21 gate commands passing.

---

## Scope Check and File Structure

This walkthrough plan covers two sequential phases before code implementation:

1. **Worktree preservation and branch hygiene (Tasks 1 and 2):** Fast-forward workstation `main` to `2727e2b`, rebase and commit the 34 project-owned files in `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract`, and back up the 5 untracked skill files in `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/repository-setup-playbook`.
2. **Wayfinder decision walkthrough (Tasks 3 through 6):** Walk through the open questions on [Choose the training stack and compute setup](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5) (8 controls across 2 four-decision rounds) and [Define the first dataset and per-turn training contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4) (6 remaining controls across 2 rounds), then resolve downstream tickets `#6`, `#9`, and `#7`.

Once Tasks 1 through 6 close the Wayfinder map ([Plan our first custom GLM-5.3-Flash drafter experiment](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/2)), separate implementation plans will be generated via `/to-spec` and `/to-tickets` for each code subsystem (collection journal and Modal generator, target feature-cache extractor, 5-layer RoPE drafter and trainer, and KDA state-recovery verifier).

### Files Created or Modified Across This Plan

- Create: `docs/research/project-status.md` (Tier 3 status report documenting code, worktrees, issues `#2` through `#9`, and Memex sessions)
- Create: `docs/superpowers/plans/2026-11-07-open-questions-and-next-steps.md` (this walkthrough plan)
- Modify (in worktree `4-dataset-training-contract`): `.gitleaks.toml`, `.typos.toml`, `CONTEXT.md`, `README.md`, `docs/learning/NOTES.md`, `docs/learning/RESOURCES.md`, `docs/learning/guides/drafter-config.html`, `docs/research/first-drafter-design-comparison.md`, `docs/research/first-drafter-recommendation.md`, `docs/target-contract.md`, plus 24 untracked research/learning/handoff files
- Create (during Task 3 for Issue `#5`): `docs/research/training-stack-and-compute-contract.md`
- Create (during Task 4 for Issue `#4`): `docs/research/first-dataset-contract.md`
- Create (during Task 5 for Issue `#6`): `docs/research/prior-code-and-artifact-selection.md`
- Create (during Task 6 for Issues `#9` and `#7`): `docs/research/broad-capture-and-elsi-protocol.md` and `docs/research/first-run-protocol-and-evidence.md`

---

### Task 1: Sync Workstation `main` and Commit the `4-dataset-training-contract` Worktree

**Files:**

- Modify (in `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract`):
  - `.gitleaks.toml:1-13`
  - `.typos.toml:12-19`
  - `CONTEXT.md:55-655`
  - `README.md:25-27`
  - `docs/learning/NOTES.md:27-151`
  - `docs/learning/RESOURCES.md:23-89`
  - `docs/learning/guides/drafter-config.html:11-177`
  - `docs/research/first-drafter-design-comparison.md:114-170`
  - `docs/research/first-drafter-recommendation.md:166-227`
  - `docs/target-contract.md:45-53`
- Create (stage existing untracked files in `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract`):
  - `docs/handoffs/dataset-training-contract.md`
  - `docs/learning/guides/drafter-config/configuration_examples.py`
  - `docs/learning/guides/pilot-controls-and-metrics.html`
  - `docs/learning/guides/pilot-controls-and-metrics/collection_examples.py`
  - `docs/learning/guides/pilot-controls-and-metrics/feature_projection_example.py`
  - `docs/learning/guides/pilot-controls-and-metrics/optimizer_example.py`
  - `docs/learning/guides/pilot-controls-and-metrics/pilot_examples.py`
  - `docs/learning/lessons/0002-score-a-saved-response.html`
  - `docs/learning/lessons/0002-score-a-saved-response/replay_example.py`
  - `docs/research/configuration-pilot-plan.md`
  - `docs/research/dataset-generation-and-mixture-controls.md`
  - `docs/research/drafter-data-scaling.md`
  - `docs/research/dspark-data-and-reasoning.md`
  - `docs/research/elsi-evaluation-review.md`
  - `docs/research/glm-training-and-workload-focus.md`
  - `docs/research/hosted-generation-and-feature-capture.md`
  - `docs/research/hosted-provider-qualification.md`
  - `docs/research/opb-components-and-nemotron-license.md`
  - `docs/research/open-dataset-selection.md`
  - `docs/research/openperfectblend-conversation-policy.md`
  - `docs/research/thinking-mode-drafter-acceptance.md`
  - `docs/research/token-record-and-recovery-contract.md`
  - `docs/research/token-record-and-recovery/check-contract.sh`
  - `docs/research/training-data-temperature-and-size.md`
- Test: `tests/test_references.py`, `tests/test_mhc_lesson.py`, `docs/research/token-record-and-recovery/check-contract.sh`

**Interfaces:**

- Consumes: `reference/SHA256SUMS`, `UPSTREAMS.toml`, and remote `origin/main` at `2727e2b9b323b8cbd336d658a04384bcd2b23ba9`.
- Produces: Pushed branch `origin/4-dataset-training-contract` containing all 34 project-owned dataset-contract files rebased onto `2727e2b9b323b8cbd336d658a04384bcd2b23ba9`.

- [ ] **Step 1: Verify workstation `main` divergence and worktree dirty file counts**

Run:

```bash
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter rev-parse HEAD origin/main
git -C /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract status --short | wc -l
```

Expected: `0120409b5003a187ac760391345a630fcf1fdc5a` for both refs before fetch, and `35` dirty entries (`10` modified tracked + `25` untracked) in `4-dataset-training-contract`.

- [ ] **Step 2: Fetch `origin` and fast-forward workstation `main` to `2727e2b`**

Run:

```bash
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter fetch origin main
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter merge --ff-only origin/main
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter rev-parse HEAD
```

Expected: Fast-forward from `0120409` to `2727e2b9b323b8cbd336d658a04384bcd2b23ba9`.

- [ ] **Step 3: Stage the 34 project-owned files in `4-dataset-training-contract` (excluding `opencode.json`), commit, and rebase onto `origin/main`**

```bash
cd /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract
git add \
  .gitleaks.toml \
  .typos.toml \
  CONTEXT.md \
  README.md \
  docs/handoffs/dataset-training-contract.md \
  docs/learning/NOTES.md \
  docs/learning/RESOURCES.md \
  docs/learning/guides/drafter-config.html \
  docs/learning/guides/drafter-config/configuration_examples.py \
  docs/learning/guides/pilot-controls-and-metrics.html \
  docs/learning/guides/pilot-controls-and-metrics/collection_examples.py \
  docs/learning/guides/pilot-controls-and-metrics/feature_projection_example.py \
  docs/learning/guides/pilot-controls-and-metrics/optimizer_example.py \
  docs/learning/guides/pilot-controls-and-metrics/pilot_examples.py \
  docs/learning/lessons/0002-score-a-saved-response.html \
  docs/learning/lessons/0002-score-a-saved-response/replay_example.py \
  docs/research/configuration-pilot-plan.md \
  docs/research/dataset-generation-and-mixture-controls.md \
  docs/research/drafter-data-scaling.md \
  docs/research/dspark-data-and-reasoning.md \
  docs/research/elsi-evaluation-review.md \
  docs/research/first-drafter-design-comparison.md \
  docs/research/first-drafter-recommendation.md \
  docs/research/glm-training-and-workload-focus.md \
  docs/research/hosted-generation-and-feature-capture.md \
  docs/research/hosted-provider-qualification.md \
  docs/research/opb-components-and-nemotron-license.md \
  docs/research/open-dataset-selection.md \
  docs/research/openperfectblend-conversation-policy.md \
  docs/research/thinking-mode-drafter-acceptance.md \
  docs/research/token-record-and-recovery-contract.md \
  docs/research/token-record-and-recovery/check-contract.sh \
  docs/research/training-data-temperature-and-size.md \
  docs/target-contract.md
git commit -m "Document dataset controls 1-28 research and token recovery contract"
git rebase origin/main
```

- [ ] **Step 4: Run the repository gates and contract checker in `4-dataset-training-contract`**

Run:

```bash
cd /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract
git submodule update --init upstream/deepspec
python -m glm_flash_drafter
pytest
./run-gates.sh
```

Expected: `Pinned reference checksums verified.`, `17 passed`, and `all gates pass`.

- [ ] **Step 5: Push `4-dataset-training-contract` to `origin`**

```bash
cd /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract
git push -u origin 4-dataset-training-contract
```

---

### Task 2: Preserve the `repository-setup-playbook` Worktree and Clean Up Merged Worktrees

**Files:**

- Create (stage existing untracked files in `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/repository-setup-playbook`):
  - `docs/research/reusable-repository-setup.md`
  - `docs/research/setup-repository-validation.md`
  - `skills/setup-repository/SKILL.md`
  - `skills/setup-repository/references/research.md`
  - `skills/setup-repository/references/working-rules.md`
- Test: `tests/test_references.py`, `tests/test_mhc_lesson.py`

**Interfaces:**

- Consumes: Fast-forwarded `origin/main` at `2727e2b9b323b8cbd336d658a04384bcd2b23ba9`.
- Produces: Pushed backup branch `origin/repository-setup-playbook` and removal of the already-merged `mhc-numerical-walkthrough` (`7e2fdd5`) and `software-factory-prompts` (`5905904`) worktrees.

- [ ] **Step 1: Verify that `mhc-numerical-walkthrough` and `software-factory-prompts` are clean and merged into `origin/main`**

Run:

```bash
git -C /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/mhc-numerical-walkthrough status --short
git -C /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/software-factory-prompts status --short
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter branch --merged origin/main
```

Expected: Empty status output for both worktrees, and both `mhc-numerical-walkthrough` and `software-factory-prompts` listed as merged into `origin/main`.

- [ ] **Step 2: Commit and push the 5 untracked files in `repository-setup-playbook`**

```bash
cd /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/repository-setup-playbook
git rebase origin/main
git add \
  docs/research/reusable-repository-setup.md \
  docs/research/setup-repository-validation.md \
  skills/setup-repository/SKILL.md \
  skills/setup-repository/references/research.md \
  skills/setup-repository/references/working-rules.md
git commit -m "Add setup-repository skill and behavioral validation record"
git push -u origin repository-setup-playbook
```

- [ ] **Step 3: Remove the two clean, already-merged worktrees**

Run:

```bash
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter worktree remove /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/mhc-numerical-walkthrough
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter worktree remove /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/software-factory-prompts
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter worktree list
```

Expected: Only `/Users/phillipchaffee/git/glm-5.3-flash-drafter`, `4-dataset-training-contract`, and `repository-setup-playbook` remain, and all three have remote commits backing up their work.

---

### Task 3: Walk Through Open Question 1 — Resolve Issue `#5` (Training Stack and Compute Setup)

**Files:**

- Create: `docs/research/training-stack-and-compute-contract.md`
- Modify: `docs/target-contract.md:60-75`
- Test: `tests/test_references.py`

**Interfaces:**

- Consumes: Approved baseline in `docs/research/first-drafter-recommendation.md` and Controls `25` through `28` in `docs/research/token-record-and-recovery-contract.md`.
- Produces: Approved stack and compute contract (`docs/research/training-stack-and-compute-contract.md`) and closure of [Choose the training stack and compute setup](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5).

- [ ] **Step 1: Claim Issue `#5` and create a dedicated worktree**

Run:

```bash
gh issue edit 5 --repo PhillipChaffee/glm-5.3-flash-drafter --add-assignee PhillipChaffee
git -C /Users/phillipchaffee/git/glm-5.3-flash-drafter worktree add -b 5-training-stack-compute /Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/5-training-stack-compute origin/main
```

Expected: Issue `#5` is assigned to `PhillipChaffee` and worktree `5-training-stack-compute` is checked out at `2727e2b`.

- [ ] **Step 2: Present Round 1 of Issue `#5` (Stack Controls `S1` to `S4`: Generation Engine, Recorder API, Durable Storage, and Feature Capture Runtime)**

Walk through these four decisions with the user (referencing what published drafters use and the capability-first rule from Controls `27` and `28`):

| Control | Question | Recommended Option | Rationale and Published Practice |
| --- | --- | --- | --- |
| **S1: Target generation engine on Modal** | Which engine serves pinned `zai-org/GLM-5.3-Flash` (`46a050b`) for response generation? | **vLLM** (`vllm` pinned revision supporting `glm5_next` with native `AsyncLLM`) | At inspected SGLang `961bcf481`, OpenAI Chat rejects `return_token_ids` when streaming and `schedule_batch.py` can substitute a stop token ID after an invalid-ID error. vLLM supports native cumulative token output and KDA state recovery. |
| **S2: Recorder capture interface** | How does the recorder collect prompt token IDs, completion token IDs, reasoning, and stop evidence? | **Native vLLM engine wrapper (`RequestOutputKind.CUMULATIVE`, `n=1`)** with positional unpadding and explicit `<think>` (`154841`) / `</think>` (`154842`) splitting | Avoids OpenAI Chat delta-mode chunk loss and reasoning-parser suppression; copies each cumulative snapshot, checks prefix monotonicity, and verifies terminal stop IDs (`154820`, `154827`, `154829`). |
| **S3: Durable collection journal and storage** | Where do the collection journal and attempt artifacts live during Modal generation? | **Modal Volume v1 (`glm-drafter-v2`) with a single-owner coordinator process** and shard-packed attempt archives | Modal Volume v1 provides explicit `commit()`/`reload()` durability barriers; single-owner coordination avoids missing distributed file locks, and shard packing stays well under the `500,000`-inode hard limit (target `<5,000` inodes). Modal Volume v2 is rejected due to documented data-loss warnings. |
| **S4: Target feature-capture and verification path** | How are `[M20, M28, M32, M36, M40, M44]` and `M45` captured and how is KDA state recovered during verification? | **Offline fixed-token teacher replay with hook capture** (checking `"deepseek_sparse_attention"` -> `"indexed_attention"` mapping) for training caches, and **vLLM accepted-prefix KDA recovery** for speculative verification | Separates generation throughput from `57,344`-byte/position feature storage; matches DeepSpec's two-stage cache builder while using vLLM's accepted-length recurrent/conv1d state recovery instead of unsupported `.crop()`. |

- [ ] **Step 3: Present Round 2 of Issue `#5` (Stack Controls `S5` to `S8`: GPU Hardware, Precision, Optimizer and Batch Recipe, and Spend Caps)**

Walk through the second set of four decisions with the user:

| Control | Question | Recommended Option | Rationale and Published Practice |
| --- | --- | --- | --- |
| **S5: GPU hardware and parallelism** | Which Modal GPU configuration runs target generation, feature capture, and drafter training? | **`4x B200` (`TP=4`) for target generation and feature replay; `4x B200` (`DP=4`) for 5-layer drafter training** | `zai-org/GLM-5.3-Flash` weights (`~328 GB` in `bf16`) fit cleanly across `4x B200` (`720 GB` aggregate HBM) with headroom for long-context KV/KDA caches and 7-tap activations. |
| **S6: Numerical precision and compilation** | What precision and compiler settings govern target replay and drafter training? | **`bfloat16` for target replay, cached features, and drafter weights (`float32` optimizer state and loss reduction); `eager` mode initially** (`torch.compile` tested only as a gated parity check) | Matches Red Hat and DeepSpec reference precision; avoids historical `torch.compile` graph-break and recompilation stalls until eager learning is proved. |
| **S7: Optimizer, learning rate, and effective batch size** | What optimization recipe runs the initial drafter training control? | **AdamW (`lr = 3e-4`, cosine decay, `5%` linear warmup, `weight_decay = 0.01`, `grad_clip = 1.0`) with token-budget batching (`effective_batch = 64` sequences / `~262k` tokens)**, plus a bounded LR pilot (`1e-4`, `3e-4`, `6e-4`) | Follows `#5 comment 2` rule that a bounded LR check takes precedence over adding feature candidates if the initial control is unstable. |
| **S8: Stage-gated GPU-hour and spend ceilings** | What hard compute ceilings bound the initial pilots? | **Per-stage hard caps:** Stage 0 qualification `<= 2` GPU-hours; Generation pilot `<= 8` GPU-hours (`2h` wall clock on `4x B200`); Feature capture + Training pilot `<= 16` GPU-hours | Enforces explicit stop gates before scaling beyond the pilot corpus. |

- [ ] **Step 4: Write `docs/research/training-stack-and-compute-contract.md` and run `./run-gates.sh`**

Create `docs/research/training-stack-and-compute-contract.md` recording the user-approved choices for `S1` through `S8` and the audited GPU dependency pins, then run:

```bash
PATH="/workspace/.venv/bin:/home/ubuntu/.local/bin:/usr/local/bin:/home/ubuntu/.nvm/versions/node/v22.23.4/bin:$PATH" ./run-gates.sh
```

Expected: `all gates pass`.

- [ ] **Step 5: Post resolution comment on Issue `#5`, close `#5`, and update Wayfinder map `#2`**

```bash
gh issue comment 5 --repo PhillipChaffee/glm-5.3-flash-drafter --body-file docs/research/training-stack-and-compute-contract.md
gh issue close 5 --repo PhillipChaffee/glm-5.3-flash-drafter --reason completed
```

---

### Task 4: Walk Through Open Question 2 & 3 — Resolve Remaining Controls `29` to `34` on Issue `#4`

**Files:**

- Create: `docs/research/first-dataset-contract.md`
- Modify: `docs/handoffs/dataset-training-contract.md:384-411`
- Test: `tests/test_references.py`

**Interfaces:**

- Consumes: Approved Controls `1` through `28` in `docs/handoffs/dataset-training-contract.md` and approved storage/compute limits from Task 3 (`Issue #5`).
- Produces: Complete assembled dataset contract (`docs/research/first-dataset-contract.md`) and closure of [Define the first dataset and per-turn training contract](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4), unblocking Issues `#6` and `#9`.

- [ ] **Step 1: Present Round 8 of Issue `#4` (Controls `29` to `32`: Training Volume Ceilings, Sequence Length & Feature-Cache Storage Limits, Full-Corpus Filtering, and Historical Data Treatment)**

Walk through these four decisions with the user in `/Users/phillipchaffee/worktrees/glm-5.3-flash-drafter/4-dataset-training-contract`:

| Control | Question | Recommended Option | Rationale and Published Practice |
| --- | --- | --- | --- |
| **Control 29: Training-pilot and first-experiment volume ceilings** | How many retained assistant-turn examples and tokens feed the training pilot vs. the first full training run? | **Training pilot:** `1,000` retained complete assistant-turn examples (`~2M–4M` supervised tokens, `800` train / `100` val / `100` test prompt families). **First full run:** `10,000` retained examples (`~20M–40M` tokens), expandable only after pilot learning curves pass | Bridges the `50`-attempt generation pilot (`Controls 21–24`) and full-scale training (published EAGLE-3 data-scaling curves show clear learning progression at `8.5k–17k` dialogues before committing to `1.72M`-example scale). |
| **Control 30: Training sequence-length cap and feature-cache storage ceiling** | What sequence length is cached for training replay, and how is the `57,344`-byte/position feature cache bounded? | **Replay/training sequence cap = `8,192` tokens** (`4,096` in initial pilot); **on-disk feature-cache cap = `250 GB`** using rolling tar/binary shards (`<= 1,000` files total) and streaming replay for multi-epoch training | At `57,344` bytes/position (`6` taps + `M45` in `bf16`), `1M` positions = `57.34 GB` and `4M` positions = `229.38 GB`. Matches Red Hat's `8,192` training cap while keeping Modal Volume v1 under `250 GB` and `<1,000` inodes. |
| **Control 31: Scaling classification and eligibility beyond the manual pilot** | How are `40/30/20/10` category assignment, English eligibility, self-contained context, and follow-up fit applied beyond the `50`-attempt manual pool? | **Two-stage deterministic + pinned-LLM filter audited against the manual pilot gold set** (`>= 95%` agreement on the manual pilot annotations before bulk labeling; deterministic rules first, pinned local/Modal classifier second) | Preserves Control `9–12` semantics without manual review bottlenecks at `1,000–10,000` examples. |
| **Control 32: Treatment of historical `GLM-5` Modal data (`698` conversations, `686` cached samples)** | Are `glm-drafter-run:data/regen.jsonl` and `cache/shard-00000.bin` used in training, or excluded? | **Strictly excluded from training and model evaluation; retained read-only as an engineering smoke fixture only** | Historical rows used `reasoning_effort=low`, `clear_thinking=true`, `T=0.7`, had a prompt-vs-replay conditioning mismatch, and came from the buggy pre-Control-27/28 generator. |

- [ ] **Step 2: Present Round 9 of Issue `#4` (Controls `33` and `34`: Artifact Access/Publication Policy and Holistic Contract Approval)**

Walk through the final two dataset decisions with the user:

| Control | Question | Recommended Option | Rationale and Published Practice |
| --- | --- | --- | --- |
| **Control 33: Artifact access, retention, and publication policy** | Where are generated datasets, manifests, feature caches, and checkpoints stored, and are they private or public? | **Private Modal Volume (`glm-drafter-v2`) + private Hugging Face dataset/model backups (`PhillipChaffee/glm-5.3-flash-drafter-data`) initially**, with a mandatory per-record provenance and upstream license ledger (`OpenPerfectBlend` source row + `GLM-5.3-Flash` MIT license) before any public release | Keeps large artifacts outside Git (`README.md:57-58`) while preserving reproducibility and clean provenance. |
| **Control 34: Complete assembled contract sign-off** | Does the complete 34-control contract satisfy all requirements of Issue `#4`? | **Approve complete assembled contract in `docs/research/first-dataset-contract.md`** | Consolidates Controls `1` through `34` into one normative document and closes Issue `#4`. |

- [ ] **Step 3: Write `docs/research/first-dataset-contract.md` and run `./run-gates.sh`**

Assemble Controls `1` through `34` into `docs/research/first-dataset-contract.md`, verify links and terminology against `CONTEXT.md`, and run:

```bash
PATH="/workspace/.venv/bin:/home/ubuntu/.local/bin:/usr/local/bin:/home/ubuntu/.nvm/versions/node/v22.23.4/bin:$PATH" ./run-gates.sh
```

Expected: `all gates pass`.

- [ ] **Step 4: Post resolution comment on Issue `#4`, close `#4`, and update Wayfinder map `#2`**

```bash
gh issue comment 4 --repo PhillipChaffee/glm-5.3-flash-drafter --body-file docs/research/first-dataset-contract.md
gh issue close 4 --repo PhillipChaffee/glm-5.3-flash-drafter --reason completed
```

---

### Task 5: Resolve Unblocked Issue `#6` (Select Prior Code and Artifacts for the First Experiment)

**Files:**

- Create: `docs/research/prior-code-and-artifact-selection.md`
- Test: `tests/test_references.py`

**Interfaces:**

- Consumes: Closed Issues `#3`, `#4`, and `#5`, plus `docs/research/glm5-salvage-assessment.md` and `docs/research/artifact-inventory.md`.
- Produces: Approved salvage and repair checklist (`docs/research/prior-code-and-artifact-selection.md`) and closure of [Select prior code and artifacts for the first experiment](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/6).

- [ ] **Step 1: Claim Issue `#6` and walk through the 4 salvage and artifact disposition decisions**

Present the 4 concrete decisions for `#6`:

1. **GLM DSpark modeling & trainer port (`deepspec/modeling/dspark/glm5/`):** Port `modeling.py`, `config.py`, `hidden_states.py`, and `glm5_trainer.py` into project-owned code with 5 mandatory repairs: (a) pin `46a050b`, (b) enable full-head RoPE (`theta=10,000`), (c) bidirectional block size `8` with mask ID `154856` and `2,048` context window, (d) ordered 6-tap unweighted stream means `[M20, M28, M32, M36, M40, M44]` + separate `M45` normalized once by frozen target `RMSNorm`, and (e) `0.1*CE(teacher_argmax) + 0.9*TV + 1.0*BCE` loss.
2. **Batching & cache tools (`token_budget_batcher.py`, `prepare_target_cache_glm.py`, `merge_cache_slices.py`):** Port `token_budget_batcher.py` after adding negative-length and negative-`max_row_tokens` input validation; rebuild the generation recorder from scratch to satisfy Controls `27` and `28` rather than patching buggy `generate_train_data_glm.py`.
3. **ELSI analysis tools (`elsi_capture_glm.py`, `elsi_analyze.py`):** Port with explicit 1-based boundary labels (`M1..M45` raw stream means vs. `R45 = target_norm(M45)`) and strict set-union conditional CKA (`ELSI-C`).
4. **Modal volume disposition (`1.261 TB` across 3 old volumes):** Keep `glm-serve-weights/weights/target/` (`328.37 GB`, after verifying SHA-256 against `46a050b`) and `weights/drafter/` (`2.34 GB` DFlash2 baseline); archive or delete the `401.42 GB` of superseded step-20 `GLM-5` checkpoints once smoke checks finish.

- [ ] **Step 2: Write `docs/research/prior-code-and-artifact-selection.md`, run `./run-gates.sh`, and close Issue `#6`**

```bash
PATH="/workspace/.venv/bin:/home/ubuntu/.local/bin:/usr/local/bin:/home/ubuntu/.nvm/versions/node/v22.23.4/bin:$PATH" ./run-gates.sh
gh issue comment 6 --repo PhillipChaffee/glm-5.3-flash-drafter --body-file docs/research/prior-code-and-artifact-selection.md
gh issue close 6 --repo PhillipChaffee/glm-5.3-flash-drafter --reason completed
```

---

### Task 6: Resolve Unblocked Issues `#9` (ELSI Protocol) and `#7` (First-Run Protocol and Evidence)

**Files:**

- Create: `docs/research/broad-capture-and-elsi-protocol.md`
- Create: `docs/research/first-run-protocol-and-evidence.md`
- Modify: `README.md:82-92`
- Test: `tests/test_references.py`

**Interfaces:**

- Consumes: Closed Issues `#3`, `#4`, `#5`, and `#6`.
- Produces: Closed Issues [Define the broad-capture and ELSI comparison protocol](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/9) and [Define the first-run protocol and evidence](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/7), completing all 6 sub-issues on [Plan our first custom GLM-5.3-Flash drafter experiment](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/2).

- [ ] **Step 1: Walk through and close Issue `#9` (Broad-Capture and ELSI Comparison Protocol)**

Confirm the 4 protocol parameters from `#9 comments 1–2` and `docs/research/elsi-evaluation-review.md`:

1. **Calibration subset:** `250` training-split conversations (`~30,000` retained assistant positions after top-`30%` predictive-entropy filtering and `32`-token pooling).
2. **Capture pool:** All `45` raw completed-block 4-stream means (`M1..M45`) plus post-norm teacher state `R45 = target_norm(M45)` (normalizing each token before pooling).
3. **Three-arm feature comparison:** Arm 1 = approved 6-tap control `[M20, M28, M32, M36, M40, M44]` (`24,576 -> 4,096`); Arm 2 = top distinct 6-tap candidate `S6` (`24,576 -> 4,096`); Arm 3 = top 7-tap candidate `S7` (`28,672 -> 4,096`).
4. **RNG-matched initialization:** Explicitly copy shared initial module tensors across 6-tap and 7-tap arms so the extra `4,096 x 4,096` (`16,777,216`-parameter) projection block in `S7` does not perturb backbone weight initialization.

Write `docs/research/broad-capture-and-elsi-protocol.md` and close `#9`:

```bash
gh issue comment 9 --repo PhillipChaffee/glm-5.3-flash-drafter --body-file docs/research/broad-capture-and-elsi-protocol.md
gh issue close 9 --repo PhillipChaffee/glm-5.3-flash-drafter --reason completed
```

- [ ] **Step 2: Walk through and close Issue `#7` (First-Run Protocol and Evidence)**

Confirm the 5-stage execution and evidence gates from `#7 comments 1–3` and `docs/target-contract.md:75-86`:

1. **Gate 1 (CPU & live pre-collection qualification):** Pass `check-contract.sh` and live Modal qualification for Controls `27` and `28` (exact token IDs, positional `154820/154827/154829` stop handling, `<think>` retention, crash/restart journal recovery, and `2h` deadline cancellation).
2. **Gate 2 (Target feature & logit parity):** Prove on GPU that offline hook capture of `[M20, M28, M32, M36, M40, M44]` matches online serving capture and that `lm_head(target_norm(M45))` matches native target logits within `bf16` tolerance.
3. **Gate 3 (Tiny cache, train, save, reload, and KDA recovery):** Train the 5-layer baseline on a tiny cache, verify loss decrease, save and reload the checkpoint with exact state continuity, and verify that forced draft rejection, partial acceptance, and full acceptance with recovered KDA state match fresh greedy target-only decoding.
4. **Gate 4 (50-attempt generation pilot & 3-arm feature pilot):** Run the `50`-attempt generation pilot, then compare the 6-tap control against `S6` and `S7` under matched exposures.
5. **Gate 5 (Evaluation counters & latency):** Report submitted draft tokens (`8`), accepted draft tokens, draft-token acceptance fraction, verification progress (`accepted + 1`), and wall-clock latency/throughput against target-only decoding and public `RedHatAI/GLM-5.3-Flash-speculator.dspark` / `DFlash2` on identical hardware and concurrency.

Write `docs/research/first-run-protocol-and-evidence.md` and close `#7` and map `#2`:

```bash
gh issue comment 7 --repo PhillipChaffee/glm-5.3-flash-drafter --body-file docs/research/first-run-protocol-and-evidence.md
gh issue close 7 --repo PhillipChaffee/glm-5.3-flash-drafter --reason completed
```

---

### Task 7: Verify Repository Gates on This Plan and Commit

**Files:**

- Create: `docs/research/project-status.md`
- Create: `docs/superpowers/plans/2026-11-07-open-questions-and-next-steps.md`
- Test: `tests/test_references.py`, `tests/test_mhc_lesson.py`

**Interfaces:**

- Consumes: `run-gates.sh`, `.markdownlint-cli2.jsonc`, `lychee.toml`, `.typos.toml`.
- Produces: Clean passing run of all 21 repository gates on branch `phillip/open-questions-and-next-steps-plan-459b`.

- [ ] **Step 1: Run `python -m glm_flash_drafter` and `pytest`**

Run:

```bash
/workspace/.venv/bin/python -m glm_flash_drafter
/workspace/.venv/bin/pytest -v
```

Expected: `Pinned reference checksums verified.` and `17 passed` with `100.00%` coverage.

- [ ] **Step 2: Run all 21 repository gates via `./run-gates.sh`**

Run:

```bash
PATH="/workspace/.venv/bin:/home/ubuntu/.local/bin:/usr/local/bin:/home/ubuntu/.nvm/versions/node/v22.23.4/bin:$PATH" ./run-gates.sh
```

Expected: All 21 gates output `PASS` and final line prints `all gates pass`.

- [ ] **Step 3: Commit the status report and walkthrough plan**

```bash
git add docs/research/project-status.md docs/superpowers/plans/2026-11-07-open-questions-and-next-steps.md
git commit -m "Add project status report and open-questions walkthrough plan"
```
