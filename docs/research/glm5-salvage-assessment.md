# GLM-5 salvage assessment

## Summary

**Yes—salvage the focused Flash implementation, regression tests, corrected
experiment records, and operational failure lessons.** The initial bootstrap
framing needs correction: upstream DeepSpec lacks this integration, but the old
fork contains a real custom GLM-5.3-Flash DSpark port, including training, feature
capture, canonical-cache preparation, and SGLang serving adaptations. These are
useful implementation candidates, not a completed high-quality drafter.

The demonstrated frontier is narrower: committed port code; committed Qwen proxy
and exact-target DFlash2 baseline results; and historical reports of a 20-step GLM
training smoke, serving liveness, cache measurements, and interruption drills.
Useful trained-GLM acceptance, current-pin numerical parity, rejection recovery,
losslessness, calibrated scheduling, and matched speedup remain unestablished.

Review basis: old main [`5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18`][old-main],
reviewed October 23–24, 2026. The collector verified 48 issues, 126 ordinary
comments, 35 merged-PR metadata/file lists, and 17 branches. Five evidence reviews
and the synthesis used `openai/gpt-6.1-sol`. All issue bodies/comments were
covered; this does not imply every branch or PR received a full code review.

Evidence labels: *inspected* means committed source or test bodies were read,
not executed; *committed results* means actual checked-in experiment records;
*reported* means historical issue/document claims whose cloud artifacts were
not recovered.

Follow-up: the [authenticated artifact inventory](artifact-inventory.md)
subsequently located the Modal checkpoints, corpus, raw ELSI reports, and later
compiler-audit logs. Use that dated inventory for current artifact availability
and the more detailed prompt-conditioning analysis.

The subsequent [first-principles feasibility review](dspark-flash-feasibility.md)
qualifies the cache-rebuild recommendation, identifies public mHC capture
precedent, and recommends an existing-checkpoint pilot before selecting a
training recipe.

## 1. Prioritized selective salvage

Paths below are from the basis SHA. Test basenames in this section are relative
to `experiments/glm-drafter/tests/`.

### Priority 1 — GLM training adapter and feature contract

Retain selectively:

- `experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/{modeling.py,config.py,hidden_states.py}`.
- `experiments/glm-drafter/deepspec_glm/deepspec/trainer/glm5_trainer.py`.
- The GLM entry in `experiments/glm-drafter/deepspec_glm/deepspec/data/parser.py`.
- `test_glm5_parser_roundtrip.py`, `test_glm5_drafter_forward.py`, and
  `test_glm5_hook_equivalence.py`.

The implementation supplies dense NoPE attention, context-to-K/V projection,
upstream DSpark training integration, target embedding/head loading, and explicit
mHC feature handling. The committed recipe is five dense layers, block 7, mask
ID 154821, rank-256 vanilla Markov head, and confidence prediction. Preserve it
as a dated starting candidate, not an optimal configuration.
[Model][model] · [configuration][config] · [trainer][trainer] · [recipe][recipe] ·
[tests][forward-tests]

Boundary: project-owned GLM adapters over pinned DeepSpec common machinery.
Avoid copying all six mirrored upstream files: the parser mirror also changes
unrelated Qwen boundaries. Resolve pinned target files explicitly; the old
unrestricted, unrevisioned `snapshot_download` can fetch the complete checkpoint
despite avoiding full model construction.
[Parser changes][parser] · [trainer loading][trainer]

### Priority 1 — Data, cache, batching, and recovery as a coherent unit

Preserve the relevant parts of:

- `experiments/glm-drafter/scripts/data/{generate_train_data_glm.py,prepare_target_cache_glm.py,token_budget_batcher.py,merge_cache_slices.py}`.
- `experiments/glm-drafter/glm_drafter.py:493–646`.
- `experiments/lib/modal_pipeline.py:363–546`.
- `test_cache_resume.py`, `test_target_cache_batcher.py`, and root
  `tests/test_modal_pipeline.py`.

High-value work includes input-index-based regeneration resume, partial-line
handling, one-process sharded-target capture, whole-conversation packing
constrained by padded peak, resumable slices, canonical-format merge, and
checkpoint healing/commit integration. Keep their failure-reproduction tests
with them. [Regeneration][regen] · [capture][capture] · [packer][packer] ·
[merge][merge] · [stage integration][stages] · [recovery][recovery]

Repair before reuse: checkpoint validation can accept missing required
files/ranks; existing bad symlinks are not healed; slice identity does not
fingerprint the input; merge does not establish matching semantics across
slices. The committed `budget_probe.py` also lost its Modal decorator while its
entrypoint still calls `.remote()`. These are inspected static limitations,
not newly reproduced failures.
[Recovery implementation][recovery] · [probe defect][probe] · [entrypoint][probe-entry]

### Priority 2 — Bounded SGLang integration

Retain together:

- `experiments/glm-drafter/sglang-patch/glm5_dspark.py`.
- Relevant conversion and target-capture patches in
  `experiments/glm-drafter/dspark_bringup.py`.
- The final compatibility addendum in
  `experiments/glm-drafter/notes/dspark-compat-report.md`.

The adapter handles NoPE, q/k normalization, contiguous V storage, and stacked
context K/V. Serving also required a clean dense configuration whitelist, not
merely changing `model_type`, and capture of the final norm's return value.
[Adapter][serve-adapter] · [conversion/patches][serve-conversion] ·
[corrected compatibility findings][compat]

Boundary: this specific dense, bidirectional GLM-port architecture. Separate
engine adaptation from Modal deployment and hard-coded step-20 paths.
Exact-string patch checks fail closed on changed snippets but do not
authenticate the installed engine source.

### Priority 2 — Experiment evidence and evaluation specification

Preserve or immutably index:

- `experiments/proxy-run/results/*.json` and its corrected README interpretation.
- `experiments/dflash2-smoke/data/*.jsonl` and `results/*.json`.
- The final evaluation decision in [#42][i42].

These preserve actual inputs, aggregates, measurement pitfalls, and comparator
settings. Keep the original workload/provenance attached; benchmark data
permissions need separate review. [Proxy results][proxy-vanilla] ·
[exact-target baseline][baseline-suite] · [measurement controls][baseline-analysis]

### Priority 3 — Corrected layer-analysis tools, only if needed

Optionally retain:

- `experiments/glm-drafter/scripts/data/{elsi_analyze.py,elsi_capture_glm.py}`.
- `test_glm5_elsi_math.py` and `test_glm5_elsi_pipeline.py`.

Useful assets include bounded-memory statistics, shared pooling grids, padding
fixes, independent mathematical references, and corrected conditional-CKA
search. They are research tools, not a layer-selection oracle or prerequisite
for first training. [Corrected search][elsi-search] · [math tests][elsi-tests] ·
[capture][elsi-capture]

The PDF extractor is optional: carry `scripts/extract_paper.py`,
`scripts/extract-paper`, and `tests/test_extract_paper.py` together only if local
paper sidecars are wanted. [Implementation/tests][extractor]

## 2. Reconciled contracts: training, serving, and caches

### Feature semantics agree in intent; metadata wiring is incomplete

The old training contract uses zero-based completed decoder outputs:

- IDs `0..43`: raw decoder-block output, unweighted mean over four mHC streams.
- ID `44`: post-final-RMSNorm state, not raw block 44.
- Features concatenate in requested order; seven features have width 28,672.
- The separately retained normalized final state goes directly through the
  copied `lm_head`; apply no additional target norm.

This is implemented historically and corroborated by the destination's pinned
Transformers mean-fold/final-norm path. It is not yet verified across the new
runtime. [Feature implementation][features] · [teacher logits][teacher-logits] ·
[pinned Transformers][tf-final]

ELSI uses another namespace: columns `0..44` are raw folded block outputs;
column `45` is `postN`. Training ID 44 maps to analysis column 45. A winner
containing analysis `raw44` cannot be silently relabeled as training ID 44.
[Column contract][elsi-columns]

The SGLang patch correctly targets the final norm return rather than a
nonexistent layer 45 or a pre-norm capture seam. However, the promised
`target_final_hidden_normed` contract is unfinished: the training builder omits
it, serving conversion injects `True`, and capture behavior is inferred from
the final ID. Serialize and validate feature IDs, ordering, folding, and
normalization explicitly; reject unknown conventions.
[Builder][config] · [serving conversion][serve-conversion]

Derive dimensions from the resolved checkpoint: historical training needed a
`head_dim=64` fallback, and the inherited MLP width 12,288 differed from the
custom class default 10,240. Serving must remove irrelevant inherited hybrid
fields. Matching shapes alone does not establish feature equivalence.

### Canonical caches include a separate teacher stream

The writer stores selected context features plus a separate final hidden state,
even when the selected features already include the final state. Seven features
therefore require:

`2 × 4096 × (7 + 1) + 6 = 65,542 bytes/token`

before sample-index/filesystem overhead. Earlier seven-stream-total storage
estimates undercounted. Preserve DeepSpec's canonical version-2 format.
[Writer call][capture] · [upstream format/writer][cache-format]

Five-feature training from a seven-feature cache is not implemented by changing
config alone: the inherited reader requires exact layer-list equality. A
validated column-selection view or derived subset cache is necessary; another
teacher pass is not inherently required. [Cache validation][cache-validation]

### Prompt conditioning needs correction before reuse

Historical regeneration explicitly selected low effort, `clear_thinking=true`,
temperature .7/top-p .8/top-k 20. Cache parser rendering supplies neither GLM
template kwarg, while the new pinned template defaults to max effort and
clear-thinking=false. Existing round-trip tests use those defaults too, so they
do not expose this mismatch. Pin explicit conditioning and test exact
tokens/masks. Tool-call and multimodal training remain unproven; decide whether
the next-user boundary belongs in the loss mask.
[Request handling][regen-requests] · [parser rendering][parser-render] ·
[pinned template][target-template]

Also correct old teaching material: DSpark uses γ input rows—anchor plus γ−1
masks—and bidirectional attention within a draft block, not γ+1 rows and a
causal draft mask. [Pinned input/mask construction][dspark-common]

## 3. Completion frontier and results worth preserving

| Evidence | Defensible conclusion | Not established |
| --- | --- | --- |
| Inspected GLM source and tests | A substantive implementation exists; the old and new DeepSpec pin is `005e03b…`. | Current-pin execution or numerical correctness. |
| Reported GLM training smoke | 20 optimizer steps on four B200s; reported loss 5.95→2.94. Correct launch arithmetic: micro-4 × 4 ranks × accumulation-32 = global 512. | A fully trained or useful drafter. With 686 cached samples, the whole-batch schedule consumed 512 per effective epoch. |
| Reported SGLang bring-up | Step-20 checkpoint completed 12/12 prompts and 1,084 verification rounds; τ≈1.05 greedy/1.01 sampled. | Quality, speedup, feature parity, losslessness, or STS calibration. |
| Committed proxy aggregates | Vanilla beat RNN by mean τ≈0.0224 in the matched-recipe Qwen experiment. | GLM superiority or statistical equivalence across seeds/scales. |
| Committed exact-target baseline | Historical DFlash2/MTP/plain measurements and reproducible prompt inputs exist. | Current target-revision compatibility or the new drafter's performance. |

Sources: [training report][training-report], [batch correction][i35],
[bring-up resolution][i47], [vanilla][proxy-vanilla], [RNN][proxy-rnn],
[baseline data][baseline-suite].

Preserve these decisions and qualifications:

- Objective: [#9][i9] retired the business adopt/abandon gate; [#22][i22] retired
  the phone-agent premise in favor of learning and daily max-effort coding.
  Low-effort training was a conscious historical mismatch, not evidence of
  max-effort performance.
- Vanilla first: the Qwen A/B supports avoiding RNN complexity initially. Both
  runs retrained the backbone; this was not a frozen-backbone head swap. Old
  timing schemas differ, so there is no defensible throughput head verdict.
- Recipe priors: block 7, five layers, rank 256, confidence prediction, 4096 cap,
  ten-epoch schedule, and compiled micro-4 are dated candidates. Only
  micro-4/compile-on completed the reported sweep; this does not settle compiler
  correctness, which remains open in [#73][i73].
- Operational measurements: [#57][i57] reports 895 real/939 padded tokens/s on
  249 valid rows and 180,314 real tokens, on B200:4/seven features. The synthetic
  8192-padded-token arm passed and 16,384 OOMed. These are workload/stack-specific
  observations, not a universal memory ceiling. [#39][i39] likewise preserves
  a measured regeneration ramp, not current capacity or prices.
- Layer analysis: original ELSI-C used a surrogate; [#81][i81] corrected
  strict-union conditional CKA and other numerical defects. Five-spread led
  certain predefined families, not every subset. Post-fix results remain
  issue-reported; the claimed post-fix findings tables are absent from the
  committed document. No set has won an online τ comparison; [#52][i52] is open.

### Baseline corrections must travel with the results

The committed DFlash2 six-task τ values are 5.035 / 4.155 / 5.599 / 5.150 /
3.142 / 3.346, in GSM8K/MATH500/HumanEval/MBPP/MT-Bench/Alpaca order:
mean 4.4044. Setup: SGLang v0.5.21, TP4/four B300s, c1, temperature 1.
[Raw results][baseline-suite]

Committed tables imply DFlash2/plain aggregate speedups of 1.573× at c1 and
1.037× at c16. The README's 1.9× c1 claim was against MTP; the raw aggregate
DFlash2/MTP ratio is approximately 1.194×. Higher-concurrency plain/MTP
controls are missing. Historical “losslessness PASS” also needs qualification:
initial exact equality was 16/24, and a subsequent divergence chose the
fourth-ranked token, 1.25 log-probability units below top-1—not merely a near
tie. This does not identify a verifier bug; it does rule out treating
“near-ties only” as proven.
[Controls/arithmetic][baseline-analysis] · [probe][baseline-probe]

[#42][i42] remains useful preregistration: four arms on one engine image,
matched workloads, advisory epoch probes, schedule-end judgment. Before reuse,
resolve 420 listed examples versus “400,” >4.40 versus ≥4.40, and the rounded
versus exact mean. Keep its low-effort protocol distinct from the new project's
max-effort relevance question.

## 4. First preservation actions: external artifacts

Inventory artifacts before deleting, rerunning, or relying on old backup
scripts. Preserve hashes, sizes, resolved revisions, launch configuration, and
lineage. This review did not retrieve cloud data.

| Priority | Historical coordinates | Preservation target |
| --- | --- | --- |
| First | Modal `glm-drafter-run:data/{perfectblend_train.jsonl,regen.jsonl,regen_error.jsonl}` | Ordered inputs, generated reasoning/text, errors, row identities. Bind resume state to the exact input snapshot. |
| First | `glm-drafter-run:home/checkpoints/deepspec/glm_drafter_block7_glm53flash/step_20/` | Model/config/train config and every rank's training state, if present. Model-only export is not exact-resume backup. |
| Next | `glm-serve-weights:results/{dspark_bringup.json,bringup_boot.log,dflash2_nightly_boot.log}`; `glm-drafter-run:results/budget-probe.json`, `data/ramp/`, `home/tensorboard/`, `elsi/` | Raw serving, probe, ramp, training, and corrected-analysis evidence. |
| Derived | `glm-drafter-run:cache/`, slice plans/manifests/indexes; `elsi/capture/` | Inventory metadata first; retain large derived tensors only when their reproducibility value justifies storage. |

These coordinates come from [recovery reports][recovery-drills],
[bring-up][i47], and [analysis corrections][i81]. Availability is unknown.
The local ELSI `col_044.pt` fragment was explicitly rejected as a dry-run
fragment, not recovered full evidence.

Reviewer metadata checks could not establish a GLM Hub backup:
`phillipchaffee/glm-drafter-run` returned HTTP 401, which does not distinguish
private, inaccessible, or nonexistent. Public Qwen backup metadata was
accessible at revision `84a947bcfc6c1438df23e30e92e829be0a2a1cc1`; that establishes
neither GLM backup nor binary integrity. Recheck authenticated artifact access
when preserving data.

The current backup/restore path needs repair: checkpoint selection is
lexicographic; exports omit optimizer/RNG/rank state; restore omits `data/`
and does not reconstruct training hierarchy. Prove restore separately from
restart recovery. [Backup][backup] · [restore][restore]

## 5. Leave behind; resolve current constraints

Leave behind retired phone economics/SLAs, generic GLM context dumps, old cost
promises, automatic workflow policy, machine-specific handoffs, obsolete
scratch defaults, full upstream mirrors, and wholesale paper copies. Preserve
a small bibliography and corrected decision register instead.

The old offline evaluator is unsuitable as the acceptance gate: cache `.crop()`
does not restore recurrent state. The SGLang pivot already acknowledged this
blocker. [Historical pivot][eval-pivot] · [pinned cache implementation][tf-cache]

Keep the new strict baseline. Old default pytest/coverage measured root
`tests/` and `scripts/`, not the experiment port. Mock feature tests, dependency
skips, and a CUDA-gated test whose tensors remain on CPU do not certify GPU
execution. [Gate scope][old-gates] · [forward tests][forward-tests]

Unresolved pins/security: destination pins are DeepSpec `005e03b…`, Transformers
`50e35cc…`, and target Hub `46a050be…`. Historical serving used image
`lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48`, source
`961bcf481d184be65c2a2879e62faa3e9d2fdc65`; no immutable image digest was verified.
Old regeneration installed unpinned `sglang[all]`; the training dependency/image
graph is not the bootstrap lock. Audit imports, advisories, deserialization,
compilation engagement, graph breaks, and compiled/reference outputs, losses,
and gradients before GPU spending.
[Historical image recipe][old-image] · [open compiler work][i73]

Alternative checkpoints: retain [#56][i56] as an open evaluation, not an assumed
warm-start decision. The public RedHat preview is MIT-declared but uses
different architecture/taps, block 8, mask 154856, another target snapshot,
and a vLLM/Speculators deployment stack. The DFlash2 card declares Creative
Commons Attribution-NonCommercial-NoDerivatives 4.0 and commercial terms;
it is not a permissive derivative base. Neither
matching model names nor public metadata establishes interchangeability.
[Pinned RedHat card][redhat-card] · [RedHat config][redhat-config] ·
[DFlash2 terms][dflash-card]

Licenses: old root Apache-2.0 does not replace DeepSpec's MIT/NOTICE lineage,
Transformers/SGLang Apache terms, target-model MIT terms, benchmark/data rights,
or paper permissions. Record original paths, SHA, attribution, modifications,
and applicable notices for each extracted unit.
[Old license][old-license] · [DeepSpec NOTICE][ds-notice] ·
[target license][target-license]

## 6. Small ordered migration plan

1. Recover and verify irreplaceable corpus/checkpoint/log artifacts; document
   missing items. Inventory large derived caches before deciding retention.
2. Freeze the runtime and implement explicit checkpoint feature/tokenizer
   metadata. Extract minimal GLM code; keep DeepSpec common code upstream-owned
   and runner plumbing separate.
3. Wire retained tests into the new measured test surface, without silent asset
   skips. Test parser conditioning, canonical-cache interoperability/subsets,
   padded batching, input fingerprints, interrupted merge/save, complete
   checkpoints, and actual restore.
4. Validate configuration conversion and patches against authenticated engine
   source; compare native/reconstructed logits and HF/cache/SGLang features.
   Force rejection, partial acceptance, and full acceptance; compare subsequent
   target-only continuation and sampled behavior.
5. Run a tiny optimizer/save/reload check, then matched candidate/comparator
   experiments. Clarify the protocol, preserve per-request records and
   uncertainty, and re-cost measured throughput/storage/retention before a
   full campaign.

### Recommended target-contract updates

These are recommendations for the subsequent port, not adopted runtime claims:

- Acknowledge the historical custom port and SGLang backend choice as candidates
  requiring revalidation.
- Specify mean-folded intermediate features, exceptional post-norm final ID,
  separate teacher-logit state, and distinct analysis numbering.
- Require producer/consumer metadata agreement; reject inferred
  foreign-checkpoint semantics.
- Record exact tokenizer/template/sampling settings and the accepted historical
  train-low/serve-max mismatch.
- Add canonical-cache identity, separate-final-stream accounting, and
  subset-selection requirements.
- Keep feature/logit parity and forced-state-recovery gates; training smoke,
  serving liveness, and spectral rankings do not satisfy them.

## 7. Exhaustive issue ledger

Exactly 48 issues, once each below: 13 open, 34 closed/completed, and one
closed/not planned. “Completed” is tracker state, not certification of the
broader experiment.

| Issue | State | Disposition | Latest evidence and chronology |
| --- | --- | --- | --- |
| [#1][i1] | Open | Extract destination/preferences; replace sprawling map. | Latest map includes daily max-effort coding and October 14 updates; phone title/early costing are stale. |
| [#2][i2] | Closed/completed | Preserve port-surface lineage. | Final audit resolution predates the implemented GLM overlay; “no port” and early costs are superseded. |
| [#3][i3] | Closed/completed | Keep dated target/engine research only. | Final survey's gated-DFlash2/no-public-DSpark framing is superseded by later baseline and checkpoint evidence. |
| [#4][i4] | Closed/completed | Archive estimation method, not budget. | Early business gate and cost envelope superseded by #9/#22; later erratum corrects SLA-summary assumptions. |
| [#5][i5] | Closed/completed | Retain sourced literature links. | Inferred phone τ bands/low-effort advice are not current performance evidence; #12 later supplies measured baselines. |
| [#6][i6] | Closed/completed | Preserve committed Qwen proxy results. | PR #15 supplies raw aggregates; #14 adds head comparison. Neither establishes GLM scaling. |
| [#7][i7] | Closed/completed | Keep acceptance/load/integration evidence bar. | Offline-first sequence superseded by #23's SGLang pivot and #42 protocol. |
| [#8][i8] | Closed/completed | Retain architecture starting rationale. | Dense five-layer/block-7/confidence decision survives; five-spread capture was superseded by provisional seven-union and measurement work. |
| [#9][i9] | Closed/completed | Preserve scope correction. | Final decision retires cost-per-conversation adopt/abandon gate; no business threshold was achieved. |
| [#10][i10] | Open | Retain unfinished close-out fields. | Body requests final τ/cost/lessons; no result comments or completed trained-GLM close-out. |
| [#11][i11] | Closed/completed | Keep cache-format/economic rationale. | PR #30 reasoning later refined by #44 resumability, measured #39/#57 throughput, and #22 retention policy. |
| [#12][i12] | Closed/completed | Preserve raw exact-target baseline; correct narrative. | PR #27 results survive; speedup arithmetic, absent high-load controls, and losslessness interpretation need qualifications above. |
| [#13][i13] | Closed/completed | Keep concise Markov/RNN explanation. | Survey found an existing RNN; its fallback priority was subsequently reduced by #14. |
| [#14][i14] | Closed/completed | Preserve negative proxy A/B result. | Latest closure confirms vanilla default; mean Δτ≈−0.0224 is not GLM or statistical-equivalence proof. |
| [#16][i16] | Closed/completed | Keep enforcement lessons, not old configuration. | Last comment reports fixes/full compliance; actual gate scope excludes the experiment port. |
| [#22][i22] | Closed/completed | Preserve actual objective and corpus/mode provenance. | Final decision retires phone premise; later costing/features supersede old envelope and fixed capture assumptions. |
| [#23][i23] | Open | Salvage port and staged evidence; training/quality unfinished. | Latest update reports recovery drills; earlier SGLang liveness and corrected launch/import evidence do not close full training. |
| [#25][i25] | Open | Narrow to remaining evaluation work. | Latest comment records baseline-rig lessons; #47 later implements serving. Scheduler/load/quality campaign remains unfinished. |
| [#28][i28] | Open | Optional corrected learning pointer. | Body only; K/V explanation unfinished. Correct input-row convention before reuse. |
| [#29][i29] | Open | Optional corrected learning pointer. | Body only; γ+1/causal-mask description conflicts with pinned DSpark. No completed exercise. |
| [#33][i33] | Closed/completed | Preserve feature contract and provisional arms. | Final decision keeps seven provisionally; #81 permits revisiting capture. Metadata wiring is incomplete; no winner. |
| [#34][i34] | Closed/completed | Keep independent cache/train knobs and batch-2 fallback. | PR #54 landed; optimized admission evolved through #57 to PR #84 padded-peak packing. |
| [#35][i35] | Closed/completed | Preserve corrected launch arithmetic. | Saved-artifact review establishes micro-4/ranks-4/accumulation-32; micro-2-derived arithmetic rejected. |
| [#36][i36] | Closed/completed | Keep compiled micro-4 candidate and failed arms. | Sweep completed one arm; PR #62 restored export. Later #23 correction supersedes “stranded changes” claim. |
| [#37][i37] | Closed/completed | Preserve schedule-coupling rationale, re-cost run. | Final decision retains ten epochs/fallback; later checkpoint cadence 100 supersedes body's save wording. |
| [#38][i38] | Closed/completed | Keep 4096 as measured candidate. | Reported length audit found 4.3% truncation; raw audit absent and storage accounting needs correction. |
| [#39][i39] | Closed/completed | Preserve ramp method and historical curve. | Final ramp/audit supports then-c256 recommendation; unpinned-engine drift and current capacity remain unresolved. |
| [#40][i40] | Closed/completed | Preserve explicit generation settings. | Final .7/.8/20, low, clear=true decision; cache prompt parity and max-effort relevance remain unresolved. |
| [#41][i41] | Closed/completed | Keep conscious-default review checklist. | Latest addendum transfers compiler correctness to #73; current micro-4/cadence-100 supersede stale values. |
| [#42][i42] | Closed/completed | Preserve final preregistration with clarifications. | Final same-image four-arm/schedule-end protocol supersedes older shorthand; execution and noted ambiguities remain. |
| [#43][i43] | Closed/completed | Keep experiment-name plumbing/tests. | Latest landing confirms PR #66; #70 subsequently disables cache export by default. Resolver still needs repair. |
| [#44][i44] | Closed/completed | Preserve recovery bundle and failure evidence. | Final drills/PR #78 supersede earlier durability assumptions; latest landing confirms integration, not complete integrity. |
| [#45][i45] | Open | Carry budget/retention decisions forward as unresolved. | Last comment adds ~4.2 TB checkpoint-retention concern; later #57 revises capture rate. No consolidated plan. |
| [#46][i46] | Open | Retain scoped preflight review intent. | No review report or explicit go; #47 clears liveness only. Old spending plan is not a new commitment. |
| [#47][i47] | Closed/completed | Salvage serving integration and failure ledger. | Final correction/PR #76 establishes whitelist, stacked override, fa4; external smoke artifacts do not establish correctness. |
| [#48][i48] | Open | Keep execution/reproducibility checklist. | No completed full-run ledger; #42 supersedes frozen-baseline-only wording and historical batch defaults. |
| [#49][i49] | Open | Keep future results-report fields. | No finished findings; #42 replaces “~5+ math-code” shorthand and keeps literature bands nongating. |
| [#51][i51] | Closed/completed | Keep corrected tools and dated analysis. | Latest correction supersedes original ELSI-C attribution; #52 limits five-spread claim. Post-fix raw results remain external. |
| [#52][i52] | Open | Retain parity → retrain → online comparison. | Latest/only comment adds deep-heavy alternative; no parity table, selected winner, or τ result. |
| [#56][i56] | Open | Keep alternative-checkpoint audit question. | No adopt/warm-start decision or run; public preview metadata does not establish compatibility. |
| [#57][i57] | Closed/completed | Salvage final padded packer/integration/tests. | Latest PR #84 addendum supersedes real-token-sum admission and pre-fix throughput; repair probe before rerunning. |
| [#60][i60] | Closed/completed | Preserve conservative optimization decision. | Final issue decision supersedes patch-first document advice; no matching measured micro-8 gain curve. |
| [#63][i63] | Closed/completed | Keep CPU import probes/negative-drill contract. | Final merge confirmation/PR #68 follows positive and broken-import drills; trim obsolete evaluator prerequisites. |
| [#65][i65] | Closed/completed | Keep recipe source register; refresh compiler claims. | PR #72 resolution transfers actionable compiler decision to still-open #73. |
| [#67][i67] | Closed/not planned | Discard accidental duplicate. | No comments; body identifies wrong-tool creation. Actual review is #41. |
| [#70][i70] | Closed/completed | Preserve no-cache-by-default backup policy. | PR #74 supersedes three-arm default; does not establish extant or restorable GLM backup. |
| [#73][i73] | Open | Retain runtime/compiler prerequisite. | Latest comment adds engagement logging; no recorded audit/version choice. #41/#65 closure does not resolve it. |
| [#81][i81] | Closed/completed | Salvage corrected math/search/tests. | Latest resolution reports corrected mixed/all-raw runs; supersedes surrogate and chance-floor claims, not #52's online gate. |

## Sources

The ledger links to final resolutions/latest relevant updates. Code/result links
are immutable at the review basis or explicitly pinned upstream revisions.

[old-main]: https://github.com/PhillipChaffee/GLM-5/tree/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18
[model]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py#L85-L297
[config]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/config.py#L39-L92
[trainer]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/trainer/glm5_trainer.py#L32-L131
[recipe]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/config/dspark/glm5_flash.py#L17-L62
[forward-tests]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/tests/test_glm5_drafter_forward.py#L55-L121
[parser]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/data/parser.py#L40-L70
[regen]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/generate_train_data_glm.py#L210-L276
[capture]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/prepare_target_cache_glm.py#L276-L382
[packer]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/token_budget_batcher.py#L19-L79
[merge]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/merge_cache_slices.py#L113-L241
[stages]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/glm_drafter.py#L493-L646
[recovery]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/lib/modal_pipeline.py#L442-L546
[probe]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/budget_probe.py#L157-L198
[probe-entry]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/budget_probe.py#L254-L261
[serve-adapter]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/sglang-patch/glm5_dspark.py#L26-L125
[serve-conversion]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/dspark_bringup.py#L90-L192
[compat]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/notes/dspark-compat-report.md#L251-L258
[proxy-vanilla]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/proxy-run/results/full_ours.json#L1-L541
[proxy-rnn]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/proxy-run/results/ab_rnn.json#L1-L553
[baseline-suite]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/dflash2-smoke/results/dflash.json#L258-L337
[baseline-analysis]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/dflash2-smoke/results/analysis.json#L183-L269
[baseline-probe]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/dflash2-smoke/results/probe2-dflash.json#L25-L97
[elsi-search]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/elsi_analyze.py#L275-L382
[elsi-tests]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/tests/test_glm5_elsi_math.py#L47-L244
[elsi-capture]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/elsi_capture_glm.py#L65-L212
[extractor]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/tests/test_extract_paper.py#L13-L39
[features]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/hidden_states.py#L36-L117
[teacher-logits]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py#L495-L512
[tf-final]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1494-L1511
[elsi-columns]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/elsi_analyze.py#L18-L27
[cache-format]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/target_cache_dataset.py#L283-L374
[cache-validation]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/target_cache_dataset.py#L203-L218
[regen-requests]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/generate_train_data_glm.py#L80-L176
[parser-render]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/data/parser.py#L113-L118
[target-template]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/chat_template.jinja#L1-L4
[dspark-common]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py#L78-L106
[training-report]: https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6081455333
[recovery-drills]: https://github.com/PhillipChaffee/GLM-5/issues/44#issuecomment-6150728773
[backup]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/lib/modal_pipeline.py#L363-L428
[restore]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/glm_drafter.py#L850-L875
[eval-pivot]: https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6091513682
[tf-cache]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/cache_utils.py#L1118-L1151
[old-gates]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/pyproject.toml#L610-L637
[old-image]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/glm_drafter.py#L79-L125
[redhat-card]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/README.md
[redhat-config]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/config.json
[dflash-card]: https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/README.md
[old-license]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/LICENSE#L89-L201
[ds-notice]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/NOTICE#L1-L38
[target-license]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/LICENSE
[i1]: https://github.com/PhillipChaffee/GLM-5/issues/1
[i2]: https://github.com/PhillipChaffee/GLM-5/issues/2#issuecomment-5861471333
[i3]: https://github.com/PhillipChaffee/GLM-5/issues/3#issuecomment-5861471414
[i4]: https://github.com/PhillipChaffee/GLM-5/issues/4#issuecomment-5861470466
[i5]: https://github.com/PhillipChaffee/GLM-5/issues/5#issuecomment-5861470703
[i6]: https://github.com/PhillipChaffee/GLM-5/issues/6#issuecomment-5868394057
[i7]: https://github.com/PhillipChaffee/GLM-5/issues/7#issuecomment-6021819935
[i8]: https://github.com/PhillipChaffee/GLM-5/issues/8#issuecomment-6054521571
[i9]: https://github.com/PhillipChaffee/GLM-5/issues/9#issuecomment-5901136526
[i10]: https://github.com/PhillipChaffee/GLM-5/issues/10
[i11]: https://github.com/PhillipChaffee/GLM-5/issues/11#issuecomment-6055393141
[i12]: https://github.com/PhillipChaffee/GLM-5/issues/12#issuecomment-6028863117
[i13]: https://github.com/PhillipChaffee/GLM-5/issues/13#issuecomment-5862764221
[i14]: https://github.com/PhillipChaffee/GLM-5/issues/14#issuecomment-6063603729
[i16]: https://github.com/PhillipChaffee/GLM-5/issues/16#issuecomment-5896322089
[i22]: https://github.com/PhillipChaffee/GLM-5/issues/22#issuecomment-6074676184
[i23]: https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6156252553
[i25]: https://github.com/PhillipChaffee/GLM-5/issues/25#issuecomment-6028869611
[i28]: https://github.com/PhillipChaffee/GLM-5/issues/28
[i29]: https://github.com/PhillipChaffee/GLM-5/issues/29
[i33]: https://github.com/PhillipChaffee/GLM-5/issues/33#issuecomment-6123350107
[i34]: https://github.com/PhillipChaffee/GLM-5/issues/34#issuecomment-6123670244
[i35]: https://github.com/PhillipChaffee/GLM-5/issues/35#issuecomment-6128548440
[i36]: https://github.com/PhillipChaffee/GLM-5/issues/36#issuecomment-6137540995
[i37]: https://github.com/PhillipChaffee/GLM-5/issues/37#issuecomment-6139001810
[i38]: https://github.com/PhillipChaffee/GLM-5/issues/38#issuecomment-6139830512
[i39]: https://github.com/PhillipChaffee/GLM-5/issues/39#issuecomment-6140763631
[i40]: https://github.com/PhillipChaffee/GLM-5/issues/40#issuecomment-6139938041
[i41]: https://github.com/PhillipChaffee/GLM-5/issues/41#issuecomment-6140758962
[i42]: https://github.com/PhillipChaffee/GLM-5/issues/42#issuecomment-6140175977
[i43]: https://github.com/PhillipChaffee/GLM-5/issues/43#issuecomment-6140354664
[i44]: https://github.com/PhillipChaffee/GLM-5/issues/44#issuecomment-6151569450
[i45]: https://github.com/PhillipChaffee/GLM-5/issues/45#issuecomment-6156251732
[i46]: https://github.com/PhillipChaffee/GLM-5/issues/46
[i47]: https://github.com/PhillipChaffee/GLM-5/issues/47#issuecomment-6144822966
[i48]: https://github.com/PhillipChaffee/GLM-5/issues/48
[i49]: https://github.com/PhillipChaffee/GLM-5/issues/49
[i51]: https://github.com/PhillipChaffee/GLM-5/issues/51#issuecomment-6169820153
[i52]: https://github.com/PhillipChaffee/GLM-5/issues/52#issuecomment-6157271732
[i56]: https://github.com/PhillipChaffee/GLM-5/issues/56
[i57]: https://github.com/PhillipChaffee/GLM-5/issues/57#issuecomment-6178912084
[i60]: https://github.com/PhillipChaffee/GLM-5/issues/60#issuecomment-6138546119
[i63]: https://github.com/PhillipChaffee/GLM-5/issues/63#issuecomment-6140502468
[i65]: https://github.com/PhillipChaffee/GLM-5/issues/65#issuecomment-6140753933
[i67]: https://github.com/PhillipChaffee/GLM-5/issues/67
[i70]: https://github.com/PhillipChaffee/GLM-5/issues/70#issuecomment-6140924483
[i73]: https://github.com/PhillipChaffee/GLM-5/issues/73#issuecomment-6140861458
[i81]: https://github.com/PhillipChaffee/GLM-5/issues/81#issuecomment-6169823400

## Verification boundary

No old code/tests were run, cloud artifacts recovered, or current GPU
compatibility proved. This research adds a report and navigation pointers;
the proposed implementation migration has not been performed.

The main risk is mistaking valuable historical implementation and smoke
evidence for current-pin correctness or a completed high-quality GLM drafter.
