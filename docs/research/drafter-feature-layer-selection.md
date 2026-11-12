# Choosing context-feature layers for the drafter

Web and source review completed 2026-11-12 by a multi-agent research run: 20
sources fetched, 25 claims checked by three independent verifiers each, 23
confirmed and 2 rejected. The main session then checked the numbering claims
against this repo's docs.

Evidence labels used below:

- **Published**: stated in a paper, config, or code that the verifiers read.
  No paper result was reproduced.
- **Heuristic**: reasoning from published results on other tasks or models.
- **Speculation**: no source.

## Summary

No published work gives a tested way to choose which target layers feed a
feature-conditioned drafter. Every design reviewed uses a fixed rule, and none
reports an ablation of layer position.

The published evidence supports two things only:

1. Features from several depths beat the top layer alone (EAGLE-3).
2. For concatenate-and-project fusion, the gain from more layers flattens
   between 5 and 7 (DFlash, DFlare).

Nothing covers hybrid linear/sparse attention, MoE layers, or mHC. For
GLM-5.3-Flash the choice has to be made by experiment. The recommended course
is a cheap offline analysis over all boundaries, followed by a small training
ablation with equal feature counts.

## Correction to the two candidate sets

The research question swapped the labels of the two sets. This repo's docs give
the correct ones (`first-drafter-design-comparison.md`, "Features: compare
meanings, not matching numbers").

| Set | Whose | Numbering | Completed blocks it reads |
| --- | --- | --- | --- |
| `[20,28,32,36,40,44]` | Public checkpoint; approved first feature control | Boundary `B_k` is the state after blocks `0..k-1` | 19, 27, 31, 35, 39, 43 |
| `[1,12,23,34,42,43,44]` | Old port | IDs 0-43 are completed blocks; ID 44 is the normalized final state | 1, 12, 23, 34, 42, 43, plus the final normalized state |

Two consequences follow from the numbering (checked by arithmetic against the
layer pattern in `glm-5.3-flash-architecture.md`):

- **All six control features are sparse-attention outputs.** Blocks 19, 27, 31,
  35, 39, and 43 all satisfy `i % 4 == 3`. No source says whether this was
  intended.
- **The control does not include the final block.** Boundary 45 is kept as the
  separate teacher state.

## What published designs do

All entries are **Published**. Layer IDs are zero-based block outputs for a
target with `N` layers.

| Design | Rule | Layers used | Position ablation |
| --- | --- | --- | --- |
| DFlash | Evenly spaced from ID 1 to ID `N-3` | 5 (8 for one 64-layer target) | None |
| DFlare | Same window as DFlash | 9 on Qwen3, 7 on GPT-OSS-20B | None |
| EAGLE-3 | Hard-coded IDs 2, `N//2`, `N-3` | 3 | Top layer alone vs these three |
| Whalen et al. | "four evenly-spaced target layers" | 4, held fixed | None |
| DeepSpec configs | Per-target list; Gemma4-12B uses `[5,17,29,41,46]` | 5 | None |

- No paper or repo gives a reason for the window or its endpoints.
- No published design is late-heavy, and none uses a run of adjacent top layers.
- The review found no DSpark paper text on layer choice, only the DeepSpec
  repository.

## What the published ablations show

| Result | Numbers | Limits |
| --- | --- | --- |
| DFlash, 5 features vs 3 | Acceptance length 5.64 vs 5.38 (Math500), 4.61 vs 4.47 (HumanEval), 3.18 vs 3.07 (MT-Bench) | Single runs. The paper does not say which 3 layers, so count and spacing are confounded. |
| DFlare, 5 / 7 / 9 features with a shared projection | Acceptance length 4.13 / 4.18 / 4.19 on Qwen3-4B | Single runs, reduced training regime, from the authors of a competing fusion method. |
| EAGLE-3, top layer vs low + mid + high | Acceptance length 5.37 to 6.13 (MT-bench), 5.22 to 6.23 (GSM8K) | One dense 8B target. The fused variant also adds a projection. EAGLE-3 is a one-layer autoregressive drafter. |

The differences between feature counts are 0.05 to 0.26 tokens of acceptance
length, with no seeds or error bars.

## Late-heavy, spread, or early-inclusive

There is no direct evidence. The points below are **Heuristic** unless marked.

- **Published practice** is evenly spread and includes a very early layer (ID 1
  or 2). This is convention without an ablation behind it.
- **For late layers:** trained per-layer probes decode the next token better at
  each successive layer (tuned-lens paper).
- **For spread:** probes transfer between nearby layers with little loss, so
  nearby layers add less to each other than distant ones do (tuned-lens paper).
- **Final layer:** representation studies find the final layer geometrically
  distinct from the layers before it (Gromov et al., tuned-lens paper). No
  drafter experiment shows that adding it helps.
- **DeepSpec's evaluator rejects the final decoder layer** as a feature. Its
  stated reason is a capture mismatch between raw hooked outputs and the
  normalized state from `output_hidden_states`. It is not a claim about
  redundancy. (**Published**, source inspection at the pinned revision.)
- A claim that adjacent late layers are the most redundant choice was rejected
  by all three verifiers and is not used here.

## Data-driven selection methods

None has been validated for choosing drafter features. All evidence comes from
dense transformers and from picking a single layer.

| Method | Support | Use here |
| --- | --- | --- |
| Trained affine probe per layer (tuned lens), fit by KL to the target's final distribution | Best supported. It measures the quantity a verified drafter must match. | Primary per-boundary score |
| Logit lens (no training) | The tuned-lens paper calls it "unreliable" | Do not use |
| CKA or covariance similarity between layers | Cheap. One paper finds that high similarity does not imply equal task usefulness. | Redundancy screen only, next to the probe |
| Label-free layer-quality metrics | Skean et al.: they pick a better layer on one model and a worse one on another | Do not rely on |
| Learned softmax weights over layers (DFlare) | Used in one drafter. Learned values are not reported and no pruning step is tested. | Untested shortcut |
| Greedy, submodular, or mutual-information selection; scalar mixes with top-k pruning; distillation layer mapping | Not covered by any verified claim in this run | Not reviewed |

## Architecture-specific evidence

No source covers Kimi Delta Attention, sparse attention, MoE feed-forward, mHC
streams, or a decoder without positional encoding.

- The one cross-architecture data point is Skean et al., which includes a
  state-space model on embedding tasks and finds flatter curves across depth.
- Whether to read features after sparse-attention layers is an open question
  that only an experiment on this target can answer.
- Averaging the four mHC streams may hide per-stream differences. No source
  addresses this. (**Speculation**)

## Recommended procedure

This is a **Heuristic** built from the evidence above. No published work has
validated it.

1. **Cache all boundaries on a small sample.** Store the stream mean for
   boundaries 0 through 45 with the same capture path the drafter will use. One
   fp16 vector per boundary is about 377 KB per token, so about 100k tokens fits
   in under 40 GB.
2. **Score each boundary.** Fit one affine probe per boundary by KL to the
   target's final distribution. Extending the probe to tokens 2 to 8 ahead is an
   addition of this review, since a block drafter predicts several tokens.
3. **Measure redundancy.** Compute pairwise linear CKA between boundaries. Check
   both curves for a period-4 pattern at sparse-attention outputs.
4. **Build equal-count candidate sets.** Hold the count at 6 so that position is
   the only variable.
5. **Train small drafters that differ only in the feature set.** Use the same
   data, steps, and seed in a reduced regime.
6. **Measure the noise floor first.** Train one arm with two or three seeds.
   Treat gaps smaller than that spread as ties.

Candidate sets, in this repo's boundary numbering:

| Set | Boundaries | Completed blocks | Sublayer type read | Purpose |
| --- | --- | --- | --- | --- |
| Control | `[20,28,32,36,40,44]` | 19, 27, 31, 35, 39, 43 | Sparse | Approved first feature control |
| DFlash rule | `[2,10,18,27,35,43]` | 1, 9, 17, 26, 34, 42 | Linear | The published convention applied to 45 layers |
| Spread, sparse-aligned | `[4,12,20,28,36,44]` | 3, 11, 19, 27, 35, 43 | Sparse | Separates spread from sublayer type |
| Data-driven | From steps 2 and 3 | | | Greedy pick by probe score, skipping near-duplicates |

The DFlash-rule set differs from the control in both spread and sublayer type.
The sparse-aligned spread set is this review's addition to tell those two
effects apart.

Metrics:

- **Primary:** mean acceptance length under greedy decoding with target
  verification, on held-out prompts from two or three task types. Every
  published ablation reports it.
- **Secondary:** acceptance by position within the block, and top-k agreement as
  an early training signal.

## Open questions

- At a fixed count of 6, does layer position change acceptance length on this
  target by more than seed noise?
- Do sparse-attention outputs condition a drafter better or worse than
  linear-attention outputs at similar depth?
- Does adding the final state help, given that the drafter already receives the
  target's sampled anchor token?
- Do the offline scores predict the acceptance-length ranking well enough to
  replace training ablations?

## Limits of this review

- No verified claim covers HASS, Kangaroo, other early-exit drafters, or
  Medusa/Hydra. That means not reviewed, not that they lack evidence.
- DFlash and EAGLE-3 are peer reviewed. DFlare and Whalen et al. are preprints.
- `upstream/deepspec` was not initialized in this checkout, so the DeepSpec
  statements rest on the verifiers' reading of the pinned revision, not on a
  local check.
- The field is moving quickly. A position ablation could appear at any time.

## Sources

| Source | Used for |
| --- | --- |
| [DFlash, arXiv 2602.06036](https://arxiv.org/abs/2602.06036) | Layer rule, 3 vs 5 feature ablation |
| [DFlare, arXiv 2606.02091](https://arxiv.org/abs/2606.02091) | 5 / 7 / 9 feature ablation, learned layer weights |
| [EAGLE-3, arXiv 2503.01840](https://arxiv.org/abs/2503.01840) | Top layer vs low + mid + high |
| [Whalen et al., arXiv 2606.11552](https://arxiv.org/abs/2606.11552) | Four evenly spaced layers, held fixed |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) | Shipped layer lists, final-layer guard |
| [z-lab/dflash](https://github.com/z-lab/dflash) | Spacing formula and shipped configs |
| [Tuned lens, arXiv 2303.08112](https://arxiv.org/abs/2303.08112) | Per-layer probes, probe transfer, logit-lens bias |
| [Skean et al., arXiv 2502.02013](https://arxiv.org/abs/2502.02013) | Layer-quality metrics across models |
| [Gromov et al., arXiv 2403.17887](https://arxiv.org/abs/2403.17887) | Final-layer distinctness |
| [Shi et al., arXiv 2605.07271](https://arxiv.org/abs/2605.07271) | Similarity is not sufficient |
