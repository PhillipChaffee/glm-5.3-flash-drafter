# Surviving artifacts and prompt-conditioning audit

Checked 2026-10-24 UTC with Modal 1.6.2 and huggingface_hub 2.3.0.
This follows the [repository salvage assessment](glm5-salvage-assessment.md).

For reuse decisions, read the subsequent
[first-principles feasibility review](dspark-flash-feasibility.md). The prompt
mismatch does not itself prove corrupt tensors: internally consistent
off-policy supervision may remain useful. Rebuild the states whose token or
feature contract changes, rather than assuming every cache must be discarded.

## Result

The GLM smoke corpus, canonical cache, six step-20 training checkpoints, serving
copy, compiler-audit logs, and corrected layer-analysis reports still exist on
Modal. The authenticated Hugging Face account exposes the Qwen proxy backup;
the expected GLM backup repository was not found.

Remote operations were read-only listings and bounded downloads of metadata,
logs, and the small GLM smoke corpus. No model weights, optimizer tensors, or
hidden-state shards were downloaded or loaded. Names/sizes establish presence,
not binary integrity or resumability.

## 1. Modal inventory

The accessible environment list contained `main`. These three experiment
volumes were recursively inventoried; unrelated volumes were not inspected.
Sizes below sum regular-file lengths, excluding symlinks. They are decimal GB,
not billed storage measurements or an atomic snapshot.

| Volume | Listed regular-file bytes | Approximate size | Main contents |
| --- | ---: | ---: | --- |
| `glm-drafter-run` | 817,563,394,261 | 817.56 GB | Training checkpoints, target-download cache, corpus, feature cache, ELSI, probes. |
| `glm-serve-weights` | 397,651,361,419 | 397.65 GB | Target weights, DFlash baseline, GLM serving copy, serving results/logs. |
| `dspark-proxy-run` | 45,760,881,888 | 45.76 GB | Qwen vanilla/RNN checkpoints, corpus, benchmark results. |

Combined listed size: 1,260,975,637,568 bytes, approximately 1.261 TB.

### GLM corpus and feature cache

All paths in this subsection are within `glm-drafter-run`.

| Path | Bytes | Inspection result |
| --- | ---: | --- |
| `data/perfectblend_train.jsonl` | 43,212,013 | Input corpus present by listing; not downloaded. |
| `data/regen.jsonl` | 2,525,861 | Downloaded and parsed: 698 conversations. |
| `data/regen_error.jsonl` | 21,457 | Downloaded and parsed: 2 error rows. |
| `cache/manifest.json` | 651 | Downloaded: version 2, 686 samples, one shard, max length 4096, hidden size 4096. |
| `cache/samples.idx` | 38,416 | Listed size equals 686 × 56-byte records; contents not downloaded. |
| `cache/shard-00000.bin` | 34,730,640,258 | Present by listing; tensor contents not verified. |

The manifest records layer IDs `[1,12,23,34,42,43,44]`, bf16 features, and
`zai-org/GLM-5.3-Flash`, but no resolved target revision or reasoning/history
template options. The 686 cached samples are not automatically identical to
all 698 corpus rows; filtering/truncation needs reconstruction to identify the
exact affected cache subset.

Corpus statistics, without reproducing conversation content:

- 599 single-turn conversations; 99 multi-turn conversations.
- 501 conversations contain nonempty reasoning.
- 69 multi-turn conversations contain earlier assistant reasoning that changes
  the context under the clear-thinking policy.
- Zero success or error rows carry `input_index`. These artifacts predate the
  identity-based resume fix; recover stable identities before using them as
  resume state. Counting completed lines is not a safe substitute.

### GLM checkpoints

Six experiment directories exist under `home/checkpoints/deepspec/`:

```text
glm_drafter_block7_glm53flash/step_20/
glm_drafter_swp_m4_c1/step_20/
audit_cmp/step_20/
audit_cmp_20261014-060246/step_20/
audit_eager1_20261014-060246/step_20/
audit_eager2_20261014-060246/step_20/
```

Each lists `config.json`, `train_config.py`, a 5,112,185,298-byte
`model.safetensors`, and all four `training_state.rank{0,1,2,3}.pt` files, each
15,447,835,535 bytes. Each bundle is about 66.90 GB; all six total about
401.42 GB including configs. File presence does not establish complete saves,
optimizer/RNG contents, equality of weights, or successful restoration.

The original config identifies `Glm5DsparkModel`, five layers, hidden size 4096,
block 7, vanilla Markov rank 256, seven target features, and Transformers
5.19.0. It retains the historical inherited fields, including `head_dim=0`.
The serving copy has the cleaned configuration instead.

No later-than-step-20 GLM checkpoint appeared in the listed experiment paths.
The recovered compiler-audit logs contain completed compiled/eager training
runs, although the old tracker decision remains open. This is more evidence
than the issue-only review established; a numerical audit verdict still
requires reviewing those runs and any comparison outputs.

### Serving copy and analysis artifacts

In `glm-serve-weights`:

- `weights/drafter-glm-step20/`: original step-20-sized weights plus four
  rank-state files and configs, about 66.90 GB. The inspected serving config
  sets `head_dim=64` and `target_final_hidden_normed=true` and uses a dense
  configuration whitelist. It differs intentionally from the training config.
- `weights/target/`: about 328.37 GB of target assets.
- `weights/drafter/`: about 2.34 GB of baseline drafter assets.
- `results/`: 18 files, including `dspark_bringup.json`, `dspark_prep.json`,
  `bringup_boot.log`, `dflash2_nightly_boot.log`, and baseline/probe results.

In `glm-drafter-run`:

- `elsi/select.json`, `select_postfix.json`, `select_allraw.json`, and
  `stage1.json` were downloaded and hashed. These are the previously missing
  raw report files, not proof that their numerical conclusions were reproduced.
- `elsi/capture/col_000.pt` through `col_045.pt` are all listed, each
  13,751,269 bytes; `entropies.pt` and `capture_meta.json` are present too.
- The downloaded capture metadata records 250 requested rows, 46 columns,
  89,432 tokens seen, and 26,834 retained positions. The 242 retained-count
  entries should be distinguished from the requested row count.
- `elsi/stage1_sums.pt` is listed at 3,466,682,821 bytes. The whole ELSI tree
  is about 4.10 GB, including dry-run and later scratch-analysis artifacts.
- `audit-torch-compile/` contains five logs; `home/tensorboard/` has 21 files.
- `cache-probe/` contains about 47.27 GB of derived probe caches.

Keep real capture data distinct from `elsi/capture-dry/`.

### Qwen proxy artifacts on Modal

`dspark-proxy-run` contains:

- `checkpoints/dspark_block7_qwen3_4b_full_step_380/`: vanilla export,
  model/config/train config; model file 2,786,273,970 bytes.
- `home/checkpoints/deepspec/dspark_block7_qwen3_4b_rnn/step_380/`:
  RNN model 2,790,994,306 bytes, configs, and all four rank-state files.
- An RNN step-5 smoke checkpoint with only rank0 state listed.
- Three corpus files and 11 result files. The large proxy feature cache is
  absent from the Modal root and present on Hugging Face.

## 2. Authenticated Hugging Face inventory

`HfApi.whoami()` confirmed account `phillipchaffee`. Account model/dataset
listings exposed zero model repositories and one dataset repository; separate
Space and bucket listings were empty. Coverage is limited to resources visible
to this login/token, not other accounts or organizations.

The dataset [phillipchaffee/dspark-proxy-run][hf-proxy] exists at revision
`84a947bcfc6c1438df23e30e92e829be0a2a1cc1`:

- 19 files totaling 614,026,768,978 bytes, about 614.03 GB.
- One `main` branch; no tags returned.
- Nine feature-cache shards, `samples.idx`, and a manifest declaring 19,817
  Qwen3-4B samples.
- Three corpus files and the vanilla step-380 model/config/train config.
- No RNN checkpoint or per-rank optimizer-state backup listed in this repo.
- Hub LFS SHA-256 values were recorded for binary files, but not independently
  verified by downloading those files.

An authenticated lookup of dataset `phillipchaffee/glm-drafter-run` returned
404. It also did not appear in the account listing. Thus there is no confirmed
GLM Hub backup available to this session; absence of access cannot prove that
one never existed or is not held under another account/name.

## 3. Prompt-conditioning mismatch: confirmed behavior and impact

The [generation code][generation] supplies `reasoning_effort=low` and
`clear_thinking=true`. The [cache parser][parser] supplies neither option.
These options change the actual text passed to the tokenizer; they are not
merely logging preferences.

A synthetic Jinja rendering check confirmed:

| Setting | Generation | Old cache rendering |
| --- | --- | --- |
| Reasoning header | `Reasoning Effort: Low` | `Reasoning Effort: Max` |
| Earlier assistant reasoning | Cleared before a later user turn | Retained |
| Current assistant reasoning | Generated and retained for its own turn | Retained |

The target-download cache lists historical Hub revisions
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` and
`91b69215804f2d2036a8867f6ce44848e278dcda`. Modal's file API could not read
their symlinks directly, so their templates were fetched from the immutable
Hub revisions instead. Those two historical templates are byte-identical to
each other and exhibit the same defaults/history behavior in the rendering
test. They are not byte-identical to the new repository's template pin.

The source-level mismatch is therefore corroborated against the historical
templates, not inferred solely from a newer template. Cache binary token
prefixes and old server request transcripts were not read; exact per-run
impact still requires that provenance or a controlled rerun.

### What this would do

The cache pass replays the stored text through the frozen target under its
rendered prompt. Its hidden features and teacher probabilities can still be
internally consistent with that replay. However, the text was sampled under
another condition: low effort and, for relevant later turns, a different
history. It is not on-policy data for the cache prompt.

Consequences are a training/deployment distribution mismatch, potentially lower
acceptance and poorer confidence calibration, and weaker relevance of the
layer-selection measurements. A falling training loss does not settle those
questions. The effect cannot be quantified from template differences alone,
and it does not identify the cause of the step-20 drafter's low acceptance.

The generated text is still recoverable material. Infrastructure findings
remain useful, although throughput/memory should be remeasured after prompt
lengths change. The Qwen head comparison and separate pretrained DFlash2
baseline were not trained with this GLM cache parser. Correct speculative
verification is a separate requirement; poor draft matching alone normally
hurts acceptance rather than changing the target distribution.

### Repair the data contract before a quality run

1. Choose explicit deployment conditions. For a max-effort coding drafter,
   collect representative max-effort completions. Historical low-effort text
   can remain useful for smoke tests or explicitly off-policy experiments.
2. Use one assistant-completion example per turn. Render the history as it was
   visible at that turn, retain that turn's own reasoning, and supervise only
   its completion. Simply setting clear-thinking=true on a whole multi-turn
   transcript deletes earlier reasoning targets.
3. Assert that each training example's token prefix matches the generation
   request prefix, including reasoning header, retained history, and assistant
   prefix. Test single-turn and multi-turn examples.
4. Recompute affected hidden-state caches, masks, and teacher probabilities.
   A metadata rename cannot repair tensors computed under another context.
5. Repeat the small training/acceptance comparison before using the old
   checkpoint or layer ranking as quality evidence.

## 4. Recovery priorities

Preserve the small corpus, configs, result JSON, and logs first. Then preserve
the original training checkpoint and its distinct serving configuration,
including all rank states if exact resume is required. A snapshot of public
target weights is not a substitute for those experiment artifacts.

The later audit runs and ELSI artifacts also deserve preservation before
deduplication. Same filenames/sizes do not establish identical binary contents.
Derived feature/probe caches can be ranked for retention after the corrected
prompt policy is chosen. No remote data was changed or deleted in this audit.

Selected downloaded-file SHA-256 values, for future provenance checks:

```text
glm-drafter-run:data/regen.jsonl
a005a2284a0efaa903ec14d780f5af1f3549bfca4d0be9da4a31609e718cbc92

glm-drafter-run:cache/manifest.json
7a57cfb77e9a6e8b5e2f40d9ed77d42ac6c3caaf800d8a59dd8fc4fd3913140d

glm-drafter-run:elsi/select_postfix.json
f92a9d2b9f65aff6952b45d7d83f02ef7675bc6128685b66b52211dac1139e4f

glm-drafter-run:elsi/select_allraw.json
072519dbcb366d6c346401cd0d88110d4f9e7400bad9487b7756b4c09e2c8ada
```

## Evidence and limitations

Direct evidence came from Modal `Volume.iterdir`/`read_file`, authenticated
Hugging Face account/repository/tree/ref APIs, downloaded JSON/configs/logs,
and assert-based synthetic rendering checks. The local scratch collection is
under the session's `glm-5.3-flash-drafter-salvage-76237/artifact-inventory/`
directory; this document retains the important locations, sizes, hashes, and
coverage boundaries without copying corpus content into Git.

Listings and bounded metadata reads do not prove full checkpoint integrity,
optimizer-state restoration, target-feature parity, or drafter quality. No GPU
compute was launched and no tensor archive was deserialized.

[hf-proxy]: https://huggingface.co/datasets/phillipchaffee/dspark-proxy-run/tree/84a947bcfc6c1438df23e30e92e829be0a2a1cc1
[generation]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/generate_train_data_glm.py#L80-L176
[parser]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/data/parser.py#L113-L118
