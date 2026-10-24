# GLM-5.3-Flash target contract

Read source revisions from `../UPSTREAMS.toml`. Local references are frozen
copies; their integrity is checked by `python -m glm_flash_drafter`.

## Target identity and features

- Hub ID: `zai-org/GLM-5.3-Flash`.
- Architecture: `Glm5NextForConditionalGeneration`; model type `glm5_next`.
- Nested text config: 45 layers, hidden size 4096, vocabulary size 154880.
- Hybrid attention: 34 linear-attention layers and 11 sparse-attention layers.
- mHC uses four streams. Intermediate decoder outputs are shaped
  `[batch, sequence, 4, 4096]`; the final hyper-head averages streams and
  normalizes to `[batch, sequence, 4096]`.
- Feature layer selection, stream reduction/projection, and normalization are
  open design decisions. Offline hooks and online features must agree.

Source: `reference/glm-5.3-flash/config.json` and
`reference/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py`.

## Prompt and output contract

Use the pinned tokenizer and `chat_template.jinja`. Record `reasoning_effort`
and `clear_thinking` explicitly in each data-generation recipe. Validate the
assistant-token mask before caching training examples. Preserve the configured
output-head vocabulary width rather than inferring it from tokenizer length.

## State recovery

DeepSpec's verifier crops caches after rejecting draft tokens. This target also
maintains linear-attention convolution and recurrent state; ordinary cache
cropping does not restore initialized recurrent state. Transformers marks the
model stateful and rejects its standard assisted-generation path.

Define and test state checkpoint/replay or a serving-backend-specific recovery
mechanism. After forced rejection, greedy continuation must agree with fresh
target-only decoding.

Read `reference/transformers/src/transformers/cache_utils.py`,
`reference/transformers/src/transformers/generation/utils.py`, and
`upstream/deepspec/deepspec/eval/base_evaluator.py` for this work.

## Compatibility status

The inspected source revisions are not a validated runtime combination.
DeepSpec pins Transformers 5.10.2 and explicitly integrates Qwen3/Gemma4.
Its pinned reference requirements also have known advisory and license-audit
failures. Define and audit a project-owned GPU lockfile before installing a
training environment; see `bootstrap-verification.md` for the observed results.
The Hub config's `deepseek_sparse_attention` label also needs validation against
the inspected Transformers `indexed_attention` cache/mask dispatch.

The checkpoint contains MTP-associated metadata and weights, but the inspected
Transformers implementation excludes the MTP layer. Native MTP is a separate
backend-dependent comparison baseline.

## First milestone

1. Choose the serving engine, GPU/sharding setup, and initial modality scope.
2. Validate configuration loading and prompt/loss-mask handling.
3. Prove offline/online feature parity and native/reconstructed logit parity.
4. Prove state recovery for rejection and partial/full draft acceptance.
5. Create a tiny cache and run one training step before scaling data collection.

DeepSpec's evaluator reports acceptance statistics, not latency. Measure actual
speedup separately against target-only decoding on the same hardware/workload.
The longer source review is in `research/repository-setup.md`.
