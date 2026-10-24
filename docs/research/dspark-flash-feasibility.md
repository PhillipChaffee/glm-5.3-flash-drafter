# DSpark / GLM-5.3-Flash feasibility review

Primary-source review completed 2026-10-24. Two collectors, five independent
researchers, and one synthesizer used `openai/gpt-6.1-sol`.

## Summary

The user clarified the goal on 2026-10-24: train our own drafter for learning
and experimentation. The earlier adoption-first recommendation is superseded.
Public checkpoints are optional comparison points. Their performance does not
decide whether custom training proceeds.

mHC does not make learning fundamentally impossible. A published GLM-5.3-Flash
drafter with reported acceptance provides substantially stronger feasibility evidence than
our old optimizer smoke test. It does not establish lossless serving, useful
max-effort coding acceleration, or affordable reproduction on our intended stack.

Preserve the old artifacts as evidence and diagnostic material, while withholding
unsupported quality and recovery claims. The old generation/replay mismatch was
unintended and inadequately documented, but it does not automatically make the
cache internally corrupt: a frozen teacher can supervise learning on off-policy
text, meaning text sampled under different conditions.

Start with explicit feature/data contracts and a small custom training run.
Keep state-recovery and sampling checks before making speculative correctness
claims. Use comparisons with target-only decoding, native MTP, and public
drafters to learn from the results.

## 1. What mHC does—and does not—prevent

Three different mechanisms must not be conflated:

| Component | What it carries | Relevant obligation |
| --- | --- | --- |
| mHC residual streams | Several representations through model depth, for each token | Capture completed states and choose an explicit reduction. |
| KDA recurrent attention and convolution | Memory of earlier tokens | Recover the accepted prefix after speculative rejection. |
| Sparse-attention caches | KV, compressed index pools, and raw indexer tails | Prevent rejected candidates from contaminating later execution. |

These distinctions are explicit in the target's [mHC implementation][mhc-code],
[KDA cache handling][kda-hf], and the serving engine's
[indexer-tail allocation][tail].

DSpark learns a mapping from selected teacher features to a proposal
distribution. It does not need to reproduce the teacher's internal computation.
Its learned context projection can consume averaged streams, concatenated
streams, or a learned reduction, provided training and serving use the same
semantics and context remains prefix-only. The [paper][dspark] and
[reference context mask][context-mask] support this interpretation.

Averaging is lossy, but loss is not automatically a bug. Different stream
arrangements can have the same mean and still produce different learned
readouts. The mHC paper's mean-preservation result concerns its doubly
stochastic residual mixing; it does not prove that intermediate means retain
everything useful for prediction. GLM's native final readout is indeed
`norm(mean(streams))`, but that establishes the final interface, not optimal
intermediate conditioning. [mHC paper][mhc-paper] · [Native final readout][final]

There is public precedent: SGLang's DeepSeek DSpark path completes deferred mHC
work before averaging, and current GLM capture reconstructs completed streams
before contraction. Therefore:

- Completed-stream mean is a defensible baseline.
- Unfinished sublayer output mistaken for a completed block is an implementation
  defect.
- Full-stream conditioning is a later quality/overhead experiment, not a
  prerequisite for learning.

[DSpark capture precedent][sg-capture] · [Current exact-target capture][v-capture]

Neither mHC itself nor intermediate averaging establishes infeasibility.
Conversely, native final averaging alone does not prove intermediate averaging
optimal.

## 2. Strongest empirical evidence, with its limits

Red Hat publishes an [exact-target DSpark checkpoint, configuration, and
evaluation card][release]. The reviewers verified its uploaded-file metadata,
not tensor contents or loadability. The publisher reports greedy aggregate
acceptance length 3.77, including 4.74 on HumanEval, using eight proposals. The
aggregate appears bonus-inclusive because `1 + 8 × 34.67% ≈ 3.77`; its
definition must be confirmed before comparing it with old τ measurements.

This is credible evidence that a trained drafter can learn useful predictions
for this target. HumanEval is relevant coding evidence, but not a max-effort,
multi-turn coding deployment test. The card supplies neither matched
target-only latency nor end-to-end stochastic correctness results. Its
substantial extraction/training setup provides no small-budget reproduction
guarantee.

There is a reproducibility gap: the public extraction commit cited by the
publisher lacks GLM auxiliary capture. Current support code cannot retroactively
establish which historical patch captured its training features. This weakens
independent reproducibility, not the existence of the released artifact or the
reported result. [Historical model source][historical-capture]

### The public and old feature contracts differ

| Contract | Public checkpoint / current source interpretation | Old port |
| --- | --- | --- |
| Context features | Six boundaries: `[20,28,32,36,40,44]` | Seven IDs: `[1,12,23,34,42,43,44]` |
| Numbering | Layer-entry boundary: 20 follows completed zero-based block 19 | IDs 0–43 denote completed blocks; ID 44 is special. |
| Final supervision | Separate boundary 45, before final norm; trainer normalizes once | Separate already-normalized final state, projected directly through the head. |
| Backbone/layout | Qwen3-style RoPE, block 8, anchor sampling | Custom NoPE, block 7. |
| Mask token | 154856 | 154821 |

The public data loader removes the final column before context projection: it
is six context features plus separate teacher supervision, not seven
conditioning features. [Released config][release-config] ·
[Context/teacher split][split] · [Teacher normalization][teacher-norm] ·
[Old feature semantics][old-features] · [Old recipe][old-recipe]

Current vLLM captures before the numbered block; HF tuple capture and old helper
IDs use different conventions. Matching integer names or tensor shapes is
insufficient. Store the semantic boundary, stream reduction, and normalization
status explicitly. Mean-of-normalized-streams differs from normalized-mean;
normalizing an already-normalized teacher state again changes its semantics.

The public RoPE drafter also demonstrates that mirroring the target's NoPE
attention is not necessary. Whether NoPE is better is an experiment, not an
architectural requirement.

## 3. What is actually wrong with the old data?

Historical generation used low effort, cleared earlier reasoning, and
temperature/top-p/top-k sampling. Replay omitted template options and therefore
used max effort with retained reasoning. This was an unintended distribution
mismatch with missing provenance. It is not approved new-quality data for
max-effort coding. [Generator][generation] · [Template policy][template]

However, the inspected builder computes features and final teacher states
together on the replayed token sequence. The inspected objective pairs each
draft slot with the corresponding next-token teacher state and excludes context
at/after the anchor. Reviewers found no inherent one-token shift or future-token
leak in that source contract; historical binary execution remains unverified.
[Builder][builder] · [Alignment][old-alignment] · [Context mask][context-mask]

A frozen teacher still defines valid next-token probabilities on off-policy
prefixes. The distinction is:

- Token cross-entropy follows the recorded continuations.
- Distribution matching follows the replay teacher.
- Those objectives need not prefer the same answer distribution.
- Neither the direction nor magnitude of practical harm is known without a
  controlled comparison.

[Loss implementation][loss]

If deployment uses max effort with retained history, some replay conditioning
may already match deployment; generation trajectories and workload coverage
still differ. Blanket regeneration or recaching does not follow from the
mismatch alone. This qualifies the earlier recommendation to rebuild everything.

### The repair is a per-turn contract, not two flags

For each turn, preserve the exact generation prefix, completion tokens,
effort/history settings, stop status, and revisions. To create deployment-matched
examples, training must use the same prefix and supervise only that turn's
completion.

Rendering an entire conversation once with clearing enabled can remove earlier
reasoning from its original training target. Retaining everything instead
changes later prefixes relative to generation. Separate per-turn examples
resolve that conflict. Max effort does not itself specify history policy.

The following user marker is token 154827, a configured GLM EOS token;
including it in assistant supervision is not inherently a mask bug. Conversely,
reconstructed strings without finish reasons cannot establish whether the final
completion ended naturally or was truncated.
[Parser][parser] · [EOS configuration][eos]

Changing only masks can preserve causal states. Changing earlier tokens
invalidates subsequent states; adding missing taps or recovering discarded
stream information requires recapture. Old low step-20 τ cannot identify the
cause: limited training, data mismatch, feature semantics, model choices,
numerical behavior, and evaluation definitions remain competing explanations.

## 4. Artifact and engineering verdict

Executed CPU checks, inspected source/tests, historical raw measurements, newly
recovered historical logs, and unvalidated tensor artifacts are distinct
evidence categories.

| Artifact / mechanism | Evidence and limitations | Recommendation |
| --- | --- | --- |
| Raw corpus | Recovered metadata reports 698 rows; all lack `input_index`. Stop/token provenance is incomplete. | Retain immutable. Establish stable source/turn identities. Reuse as declared off-policy or smoke data; generate deployment-matched data for quality claims. |
| Canonical/probe caches | Manifest declares 686 samples. Sizes are structurally consistent, but the 698→686 mapping and tensor integrity are unresolved. | Quarantine, not delete. Validate lineage, offsets, lengths, finiteness, masks, and teacher parity. Rebuild states invalidated by changed prefixes/features. |
| Step-20 weights and serving copy | Historical saves exist; weights were not loaded. Rank-file presence does not prove restore completeness. | Retain as regression/failure fixtures. Revalidate before warm-start use; keep original and serving configurations distinct. |
| ELSI captures/results | Captures contain pooled chunk sums, not tokenwise training features. Raw block 44 and post-norm column 45 differ. | Retain captures and corrected reports. Retire unsupported winning-recipe conclusions. Rescore existing sums if appropriate; recapture changed conditions or representations. |
| Feature/head helpers and patches | Mean-folding, separate final supervision, and final-norm capture have coherent intent. Hook-equivalence tests use a mock teacher. | Retain selected mechanisms and tests. Replace implicit ID translation with explicit semantics; require real-target parity. |
| Padded-token batcher | Review E executed 140,616 CPU configurations plus edge/invalid-input checks. Packing arithmetic passed on its intended domain. Negative-length validation remains incomplete; GPU safety was not tested. | Retain the algorithm. Harden exposed input validation. Repair the separate capacity-probe launcher's `.remote()`/decorator mismatch. |
| Generation resume and cache slicing/merge | Index-based resume fixes completion-order problems. Input/settings fingerprints are missing; legacy rows, retries, empty plans, and conflicting slice semantics are unsafe. Tests were inspected, not rerun. | Repair before reuse, especially before moving shards. Preserve interruption fixtures. |
| Checkpoint healing/export/restore | Validator accepts an empty step directory and does not establish complete rank/config state. Existing bad symlinks can bypass healing. Export omits resume state; restore does not reconstruct the full expected layout. | Retain design/fixtures, not guarantees. Require completed-save validation and a full round-trip test. Explicitly choose publication visibility instead of inheriting a public-dataset default. |
| Compiler audits | Newly recovered logs show compiled and two eager-labelled runs saving at step 20. Rounded final losses are close, but graph breaks and internal recompilation occurred. | Retain logs and separate checkpoints. They establish those historical runs, not output/gradient parity or qualification of the original micro-batch recipe. |
| Baseline/proxy results and protocol | Raw exact-target DFlash and Qwen Markov/RNN results exist. Workloads and environments limit transfer. | Retain measurements and controls. Reuse same-stack comparisons and prompt identities; replace unsupported conclusions and unsuitable workload/pass criteria. |

Sources: [Recovered inventory][inventory] · [ELSI capture][elsi] ·
[Helper tests][hook-tests] · [Batcher][batcher] · [Resume][resume] ·
[Merge][merge] · [Recovery/export/restore][pipeline] ·
[Baseline controls][controls] · [Proxy experiment][proxy]

### Scope of the executed batcher check

The independent Python 3.12.13 check imported only the historical stdlib
`token_budget_batcher.py` at old commit `5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18`.
It enumerated vectors of lengths 0–4 with 0–5 rows, budgets 4–12, and count caps
1–4: 140,616 configurations. An independent maximal-prefix oracle checked
packing, ordering, partition/no duplication, padded totals, bounds, and unchanged
inputs. Five additional edge cases passed; seven documented-invalid cases
raised `ValueError`.

Two gaps were reproduced: negative row lengths are accepted and can produce
negative padded totals; empty input accepts negative `max_row_tokens`. This is
limited arithmetic evidence, not execution of the cache pipeline or a GPU-memory
qualification.

### Corrections to earlier claims

- Compiler audits did occur after the earlier tracker decisions. Recovered logs
  establish later compiled/eager runs; similar losses do not establish numerical
  equivalence. Their final rounded losses were 2.7748 compiled and 2.7746 in
  both eager-labelled runs, under a micro-batch-2 audit recipe. [Inventory][inventory]
- Green old gates did not certify the port. Default discovery/coverage did not
  certify the experiment integration; some tests skip, and the CUDA-gated
  backward test does not move its model and inputs to CUDA.
  [Gate configuration][gates] · [Backward test][backward]
- Historical “losslessness PASS” was unsupported. Partial completion agreement
  plus a tie-flip explanation does not prove sampling correctness.
- The historical 1.9× README claim was against MTP, not plain decoding. Raw
  aggregate ratios are approximately 1.573× DFlash/plain at c1, 1.037× at c16,
  and 1.194× DFlash/MTP at c1. Preserve raw numbers, not the inflated summary.
  [Controls][controls] · [README][baseline-readme]
- The Qwen RNN experiment is a useful negative result, not proof of GLM
  superiority or statistical equivalence; both backbones were retrained.
  [Proxy results/protocol][proxy]

Keep a compact provenance index pointing to immutable originals. Preserve
useful evidence without importing the old project's entire context.

## 5. Recommended starting route

The public checkpoint's six-feature, RoPE, block-8 contract is a candidate
starting recipe and comparison point. The first custom recipe remains a choice
for this research project. Studying or benchmarking the public checkpoint can
proceed alongside custom training.
Current inspected vLLM is the first qualification candidate because it contains
exact-target capture, native DSpark loading, sequential Markov sampling, and
accepted-length KDA/convolution recovery. These are source-supported capabilities,
not a validated deployment pin. SGLang is an alternative; maintaining two
initial ports adds unnecessary variables.
[Capture][v-capture] · [Loader][loader] · [Proposal sampling][proposal] ·
[Recovery][recovery]

Start with fixed-length chain verification, eager execution, no
history-dependent penalties/custom processors, and no pipeline parallelism.
Add production-required features after the reference passes.

Choose one explicit recipe for the first run. Change one design choice at a
time in later experiments. Full-stream concatenation, learned reduction, RNN
heads, scheduling, and the old seven-tap NoPE design remain possible research
questions. A disappointing result can guide the next comparison.

Stock DeepSpec evaluation using generic Transformers `.crop(start)` is not a
correctness route for this hybrid recurrent cache. The pinned recurrent cache
explicitly cannot be cropped that way. Integrated accepted-prefix recovery—or
complete snapshot/replay—is required.
[Cache contract][crop] · [Generic evaluator caller][crop-caller]

## 6. Blockers versus repairable questions

| Class | Examples | Decision |
| --- | --- | --- |
| Findings that rule out a deployment claim | Rejected history affects continuation; sampler uses wrong proposal probabilities; target cannot run in the intended hardware envelope; no latency benefit even under favorable conditions. | Correct or narrow the claim and investigate the cause. These findings can guide further research. |
| Repair before scaling | Wrong boundary/normalization, unintended prefix mismatch, unknown lineage, buffer lifetime errors, incomplete restore. | Correct the contract and repeat qualification. |
| Optimization questions | Mean versus full streams, tap choice, NoPE versus RoPE, training scale, adaptive scheduling. | Investigate after correctness and useful baseline learning exist. |

## 7. Staged, falsifiable validation plan

These stages describe evidence to collect, not permission to continue the
research project. Training and reference benchmarking can proceed in parallel.
Correctness failures require investigation before interpreting measurements.
Performance losses remain valid experimental findings.

### Stage A — Freeze contracts and validate inexpensive invariants

Pin target/draft/tokenizer/runtime revisions, quantization, capture
implementation, head identity, token alignment, and normalization count. Specify
max-effort and history policy explicitly.

Use token-level fixtures for multiple turns, clearing/retention, truncation,
tools, and all stop IDs. Require generation/training prefix identity and
completion-only supervision for matched-policy data. Reconcile the old
corpus/cache mapping and validate cache structure.

Test resume/recovery against changed inputs, missing ranks, malformed files,
bad links, and conflicting slice manifests. A model-only export must not pass
as an exact-resume backup.

Refutation: incompatible data silently resumes, invalid state passes validation,
or historical samples cannot be traced. Quarantine affected reuse rather than
relabeling it.

### Stage B — Qualify public-checkpoint features, state, and probabilities

On identical prefixes, compare direct completed-block hooks, extraction/cache
round-trips, and engine captures. Check ordering, normalization, buffer reuse,
and reconstructed teacher-head logits. Establish target-versus-target numerical
variability first; choose justified bounds from that stack, not arbitrary
tolerances.

For every accepted-prefix length, force rejection/acceptance and compare
continuation logits and logical state with fresh target execution of the
committed prefix. Inspect KDA matrices, convolution windows, ordinary KV,
compressed index pools, and raw tails. In the reviewed layout, the old anchor
plus accepted proposals are consumed; the newly emitted correction/bonus
remains unconsumed until the next forward. Test pool/page boundaries, then
chunked prefill, prefix hits, graphs, mixed batches, retraction, and slot reuse.
Both engines contain recovery mechanisms, but none was executed in this review.
[Recovery][recovery] · [Tail regression][tail-test] · [SGLang caller][sg-recovery]

For stochastic serving, verify the actual conditional proposal after Markov
correction. Acceptance uses `min(1,p/q)` and rejection uses normalized positive
residual `max(p−q,0)`; backbone marginals or confidence scores cannot substitute
for `q`. Exhaustively test a tiny finite-vocabulary reference, including zero
support, deterministic proposals, and full acceptance. Then qualify engine
samplers and candidate-prefix grammar/stop handling. [Sampling contracts][proposal]

Adaptive verification-length selection is a separate correctness obligation:
DSpark Appendix A distinguishes non-anticipating scheduling from biased
retrospective prefix selection. Qualify fixed-length decoding first. [Paper][dspark]

Refutation: any reproducible rejected-history influence, alignment defect, or
probability counterexample fails correctness. High acceptance cannot rescue it.
Losslessness means the target distribution, not identical seeded stochastic text.

### Stage C — Measure performance and behavior

Compare target-only, native MTP where supported, and public DSpark on held-out
short/long, multi-turn, tool-using max-effort coding. Hold hardware, runtime,
target precision, sampling, context/concurrency envelope, cache policy, and
prompt/stop budgets fixed. Repeat and interleave controls.

Measure acceptance and request latency, first-token/inter-token latency,
throughput, memory, output lengths, and task/tool validity. Separate capture,
drafting, and verification overhead. Reconcile τ definitions; historical
custom-stack DFlash results are not a current DSpark baseline.

Interpretation: a latency loss rules out a speedup claim for that configuration.
Measure where time is spent to choose the next experiment. A correctness failure
requires a repair before comparing performance. Neither result cancels the
goal of learning through custom training.

### Stage D — Test reduced-scale learning and artifact reuse

Keep the first architecture fixed while measuring training behavior. Record
optimization, checkpoint reload, and held-out prediction/acceptance relative to
initialization. Falling training loss alone does not establish generalization.
Complete a save/export/restore/next-step comparison with optimizer, RNG, progress, and data
identity before scaling.

Compare old off-policy data against fresh deployment-matched data under equal
training budgets and otherwise fixed recipes. To attribute differences
specifically to conditioning, also control prompt composition. Measure
generation, extraction, and training throughput separately before extending the
run.

Refutation: failure to improve held-out prediction/acceptance leaves the proposed
recipe unsubstantiated; improvement without latency gain establishes learning,
not acceleration. Consider richer streams or architectural ablations afterward.

## 8. Remaining unknowns and verification boundary

Unknowns include the publisher's historical capture patch/environment, exact
effort/history data mix, stochastic correctness, matched latency, and
reduced-budget learning curve. The export/trainer Transformers-version
discrepancy also needs resolution. Target revision metadata shows identical
config, tokenizer, weight-index, and weight-file metadata between the compared
historical/current revisions; the chat template changed its tool-result sorting
implementation. That narrows the changed-weight concern without proving
identical tool-conversation rendering.
[Trainer environment][environment] · [Historical manifest][old-target-manifest] ·
[Current manifest][new-target-manifest]

The old cache subset, tensor integrity, exact historical overlay, and complete
checkpoint restore remain unvalidated.

This synthesis uses reviews A–E. Executed local checks included batcher
enumeration, synthetic template comparisons, and metadata/key-resolution checks.
Inspected test files were not thereby executed. Historical raw results and
newly recovered logs remain historical evidence. No weights/cache tensors were
loaded and no GPU, cloud, or serving tests were run for this review.

## Sources

All code links are pinned. The [artifact inventory][inventory] retains the
remote coordinates, sizes, and selected hashes behind recovered evidence.
The reviewer notes and CPU check remain in this session's feasibility scratch
directory; important findings and check scope are recorded above.

[dspark]: https://arxiv.org/html/2607.05147v1
[mhc-paper]: https://arxiv.org/html/2512.24880v1#S4.SS2
[mhc-code]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L218-L336
[kda-hf]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L687-L763
[final]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1494-L1511
[context-mask]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py#L78-L188
[sg-capture]: https://github.com/sgl-project/sglang/blob/961bcf481d184be65c2a2879e62faa3e9d2fdc65/python/sglang/srt/models/deepseek_v4.py#L4343-L4369
[v-capture]: https://github.com/vllm-project/vllm/blob/49d634bd6a3dc7433d8fb464821c95c76db85350/vllm/models/glm5next/common/model.py#L773-L861
[tail]: https://github.com/vllm-project/vllm/blob/49d634bd6a3dc7433d8fb464821c95c76db85350/vllm/models/glm5next/common/attention.py#L125-L179
[release]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/README.md
[release-config]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/config.json
[historical-capture]: https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py#L578-L704
[split]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/train/data.py#L350-L358
[teacher-norm]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dflash/core.py#L377-L407
[old-features]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/hidden_states.py#L36-L100
[old-recipe]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/config/dspark/glm5_flash.py#L17-L62
[generation]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/generate_train_data_glm.py#L80-L176
[template]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/chat_template.jinja#L132-L156
[builder]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/prepare_target_cache_glm.py#L85-L147
[old-alignment]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py#L443-L528
[loss]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py#L60-L161
[parser]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/data/parser.py#L113-L155
[eos]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/generation_config.json#L3-L7
[inventory]: artifact-inventory.md
[elsi]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/elsi_capture_glm.py#L179-L212
[hook-tests]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/tests/test_glm5_hook_equivalence.py#L18-L127
[batcher]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/token_budget_batcher.py#L19-L94
[resume]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/generate_train_data_glm.py#L210-L276
[merge]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/scripts/data/merge_cache_slices.py#L62-L241
[pipeline]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/lib/modal_pipeline.py#L71-L546
[controls]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/dflash2-smoke/results/analysis.json#L201-L269
[proxy]: https://github.com/PhillipChaffee/GLM-5/issues/14#issuecomment-6063603729
[gates]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/pyproject.toml#L610-L637
[backward]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/tests/test_glm5_drafter_forward.py#L101-L121
[baseline-readme]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/dflash2-smoke/README.md#L91-L141
[loader]: https://github.com/vllm-project/vllm/blob/49d634bd6a3dc7433d8fb464821c95c76db85350/vllm/model_executor/models/qwen3_dspark.py#L312-L372
[proposal]: https://github.com/vllm-project/vllm/blob/49d634bd6a3dc7433d8fb464821c95c76db85350/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L116-L209
[recovery]: https://github.com/vllm-project/vllm/blob/49d634bd6a3dc7433d8fb464821c95c76db85350/vllm/models/glm5next/common/kda.py#L527-L688
[crop]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/cache_utils.py#L1106-L1151
[crop-caller]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py#L407-L425
[tail-test]: https://github.com/vllm-project/vllm/blob/49d634bd6a3dc7433d8fb464821c95c76db85350/tests/kernels/test_kpool_decode_update_batched.py#L388-L463
[sg-recovery]: https://github.com/sgl-project/sglang/blob/961bcf481d184be65c2a2879e62faa3e9d2fdc65/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L1095-L1164
[environment]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/pyproject.toml#L62
[old-target-manifest]: https://huggingface.co/api/models/zai-org/GLM-5.3-Flash/revision/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a?blobs=true
[new-target-manifest]: https://huggingface.co/api/models/zai-org/GLM-5.3-Flash/revision/46a050bec13002ad29ee795c16e61cf4fa3c4a9c?blobs=true
