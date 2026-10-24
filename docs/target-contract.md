# GLM-5.3-Flash target contract

Read source revisions from `../UPSTREAMS.toml`. Local references are frozen
copies; their integrity is checked by `python -m glm_flash_drafter`.

Before implementing features, cache handling, or serving, read the
[prior-work assessment](research/glm5-salvage-assessment.md). The old project
contains a custom Flash port and SGLang integration with historical smoke
results; their contracts and known gaps should guide revalidation here.

Before selecting a drafter recipe or reusing its training cache, read the
[feasibility review](research/dspark-flash-feasibility.md) for evidence about
features, data, and state recovery. Use the public checkpoint as an optional
comparison. The [project goal](../README.md) is to train and study our own drafter.

## Target identity and features

- Hub ID: `zai-org/GLM-5.3-Flash`.
- Architecture: `Glm5NextForConditionalGeneration`; model type `glm5_next`.
- Nested text config: 45 layers, hidden size 4096, vocabulary size 154880.
- Hybrid attention: 34 linear-attention layers and 11 sparse-attention layers.
- mHC uses four streams. Intermediate decoder outputs are shaped
  `[batch, sequence, 4, 4096]`; the final hyper-head averages streams to
  `[batch, sequence, 4096]`, then a separate final RMSNorm normalizes that mean.
- The approved initial feature control uses completed-block boundaries
  `[20,28,32,36,40,44]`, in order, with unweighted stream means. A separate raw
  boundary-45 mean supplies teacher supervision through target final RMSNorm
  exactly once and the frozen output head. Offline hooks and online features
  must agree; six/seven-feature alternatives remain controlled comparisons.

Source: `reference/glm-5.3-flash/config.json` and
`reference/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py`.

The [approved first drafter baseline](research/first-drafter-recommendation.md)
records the selected architecture, feature and loss contracts, and mHC rationale.
Its combined runtime behavior still requires validation.

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

## First training milestone

1. Use the approved model contract; choose the dataset, training setup, and modality scope.
2. Make sure that configuration loading and prompt/loss-mask handling match the recipe.
3. Make sure that captured features and reconstructed teacher logits match the target.
4. Create a tiny cache and train a drafter with recorded configuration and results.
5. Save and reload the checkpoint, then test training recovery before scaling data collection.

Before interpreting speculative evaluation, make sure that training and serving
features agree. Test state recovery for rejection and partial/full acceptance.
DeepSpec's evaluator reports acceptance statistics, not latency. Measure speed
separately against target-only decoding on the same hardware/workload. Record
slow or unsuccessful experiments alongside successful ones.
The longer source review is in `research/repository-setup.md`.
