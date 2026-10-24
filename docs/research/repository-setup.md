# GLM-5.3-Flash DSpark drafter: repository setup

Source review: 2026-10-23. This is a setup recommendation, not a validated
training or serving recipe.

## Recommendation

Create a small Python integration repository with three pinned upstream inputs:
the exact GLM-5.3-Flash checkpoint metadata, DeepSpec, and Transformers. These
are the right starting sources. Give each a distinct role and a focused reading
index so agents can find the relevant code without searching entire model families.

The [Flash model card][model-card] describes a newly trained base, hybrid sparse
and linear attention, and Manifold-Constrained Hyper-Connections (mHC). The
[Transformers model documentation][model-doc] explicitly identifies `glm5_next`
as its implementation. This supports using those sources as the target's
specification rather than inferring the architecture from broader GLM-5 material.

## Upstream inputs

| Input | Role | How to include it |
| --- | --- | --- |
| `zai-org/GLM-5.3-Flash` on Hugging Face | Exact checkpoint, configuration, tokenizer, and prompt template | Download selected metadata at a commit revision; keep weights in the external Hugging Face cache. |
| `deepseek-ai/DeepSpec` | DSpark model, loss, confidence logic, data preparation, trainer, and evaluator | Pin a complete checkout as a submodule; give agents a DSpark-specific file index. |
| `huggingface/transformers` | Target implementation and shared feature/cache behavior | Pin the complete runtime dependency; keep a small, read-only source snapshot for agent inspection. |

The Transformers model directory is not an independent repository or a standalone
runtime package. It imports shared framework machinery, and its modular source
also imports other model implementations. A selective source snapshot serves as a
reference; the installed dependency must provide the full framework.

Likewise, DeepSpec's DSpark reading subset is not a standalone executable package:
package initialization imports other model families and algorithms. Retain its
complete checkout for execution. If GLM integration requires changes to its
dispatch or trainers, record those in a project fork and update the submodule pin.

Preserve upstream licenses and notices in any copied source snapshots.

### Inspected revisions

| Upstream | Revision |
| --- | --- |
| GLM-5.3-Flash Hub repository | `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` |
| DeepSpec | `005e03b81cec38b7da6399833d609ee89a2587f2` |
| Transformers | `50e35cc0ca18ec19b80653690de553e6decf74f5` |

These are provenance pins, not a tested compatible dependency set. In particular,
[DeepSpec's requirements][requirements] pin Transformers 5.10.2 and depend on
framework internals. Resolve and test that dependency boundary before creating
the project's runtime lockfile.

## Suggested layout

The following is the proposed layout. The bootstrap implements the source and
verification portions; add model-integration modules when their milestone needs them.

```text
glm-5.3-flash-drafter/
  README.md                    # Goal, current milestone, reproduction commands
  AGENTS.md                    # Target identity, source authority, reading order
  UPSTREAMS.toml               # URLs, revisions, selected paths, snapshot hashes
  pyproject.toml               # Project package and canonical Python gates
  requirements-lock.txt        # Tested dependency set
  glm_flash_drafter/           # GLM-specific feature/cache/training integration
  configs/                     # Small, explicit experiment configurations
  tests/                       # Feature, token-mask, logit, and rejection checks
  scripts/                     # Thin source-fetch and experiment entry points
  docs/
    target-contract.md         # Verified shapes, feature semantics, cache behavior
    research/
      repository-setup.md
  reference/
    glm-5.3-flash/             # Selected Hub metadata and prompt template
    transformers/             # Selected implementation sources and upstream docs
  upstream/
    deepspec/                 # Pinned complete submodule
```

Use the [canonical Python baseline][python-baseline] when bootstrapping the code
repository: lint, formatting, strict type checks, branch coverage at least 95%,
and the baseline's hygiene and dependency gates. Scope first-party code checks
to project-owned code; treat unchanged upstream snapshots as third-party inputs.

Use a Git worktree for implementation after the repository has an initial commit.

### Model metadata snapshot

Preserve `config.json`, `generation_config.json`, `tokenizer_config.json`,
`tokenizer.json`, `chat_template.jinja`, `processor_config.json`, `README.md`,
and `LICENSE` from the pinned Hub revision. The weight index is useful optional
evidence when inspecting checkpoint structure. Tokenizer files are runtime assets;
agents should usually read a short summary rather than the entire token table.

Use an explicit filename allowlist with `snapshot_download(revision=...,
allow_patterns=...)`, as documented in the [Hub download guide][hub-download].
This avoids accidentally fetching weight shards. Load the tokenizer through a
resolved download, not a raw Git LFS pointer. Keep weights, training data, feature
caches, checkpoints, and run logs outside the tracked source tree.

### Agent context boundaries

The project's agent instructions should establish:

- Target: exactly `zai-org/GLM-5.3-Flash`; Transformers model type `glm5_next`.
- Read `docs/target-contract.md` and the relevant experiment config first.
- Use the pinned checkpoint metadata for dimensions and prompt formatting;
  use the pinned implementation for execution semantics.
- Use DeepSpec as the DSpark reference; its existing Qwen/Gemma assumptions
  require explicit adaptation for GLM.
- Search project-owned code first, then explicitly selected upstream paths.
- Follow shared framework imports when a task requires them; directory boundaries
  are a reading aid, not a claim that the model implementation is self-contained.
- Treat broader GLM-5 sources as supplementary references with explicit relevance.

A short reading index should point to Transformers' `modeling_glm5_next.py`,
`configuration_glm5_next.py`, `modular_glm5_next.py`, and relevant portions of
`cache_utils.py`, `utils/output_capturing.py`, and `generation/utils.py`.

For DeepSpec, prioritize `config/dspark/`, `deepspec/modeling/dspark/`,
`deepspec/trainer/dspark_trainer.py`, `deepspec/trainer/base_trainer.py`,
`deepspec/data/parser.py`, `scripts/data/prepare_target_cache.py`,
`deepspec/eval/dspark/`, and `deepspec/eval/base_evaluator.py`.

## Integration facts that should shape the first milestone

### Hidden features have a GLM-specific shape contract

The [checkpoint configuration][model-config] specifies 45 text layers, hidden
size 4096, vocabulary size 154880, and four mHC residual streams. In the
[target implementation][modeling], intermediate decoder features have shape
`[batch, sequence, 4, 4096]`; the final hyper-head averages streams and normalizes
to `[batch, sequence, 4096]`.

DeepSpec's [cache builder][cache-builder] collects raw decoder outputs, while
its [evaluation feature extractor][draft-common] uses returned hidden states.
The GLM adapter must define feature layer selection, stream reduction or
projection, and normalization consistently across both paths. Averaging
intermediate streams is a possible experiment, not an established DSpark recipe.

### Rejected drafts require recurrent-state recovery

DeepSpec's [verifier][verifier] crops target caches after rejection. Flash's
linear-attention layers maintain convolution and recurrent state in addition
to sparse-attention cache state. The inspected [cache implementation][cache-utils]
does not support ordinary cropping of initialized recurrent state, and
Transformers marks this model stateful and rejects it in its standard assisted
generation path. Recurrent-state checkpoint/replay or a backend-specific recovery
mechanism therefore needs a correctness test.

### The current DeepSpec port is substantive

DeepSpec's [released targets][deepspec-readme] are Qwen3 and Gemma4. Its
[trainer dispatch][dspark-trainer] and [template registry][parser] explicitly
enumerate those families. The port must cover draft configuration/backbone,
assistant-token masks, target feature extraction, embedding/output-head handling,
evaluation dispatch, and target state recovery. Changing a model ID is insufficient.

### Native MTP is a separate baseline

The checkpoint config declares an MTP layer, and its weight index contains
associated extra-layer entries. However, [Transformers explicitly documents][model-doc]
that its implementation excludes MTP. Those checkpoint entries are not an exposed
DSpark implementation. Record native MTP support as a backend-dependent baseline
to compare against when available.

## First executable milestone

Before substantial data generation or training:

1. Validate target configuration loading and the agreed dependency set. The
   inspected Hub config labels sparse layers `deepseek_sparse_attention`, while
   the inspected Transformers cache/mask dispatch uses `indexed_attention`;
   confirm normalization or resolve the mismatch before claiming compatibility.
2. Verify GLM chat rendering and assistant-token loss masks, including explicit
   `reasoning_effort` and `clear_thinking` settings.
3. Verify that offline hooks and online feature extraction produce matching
   features under the chosen mHC handling.
4. Verify reconstructed target logits against native target logits.
5. Force a speculative rejection, recover target state, and compare the resumed
   greedy continuation with fresh target-only decoding. Exercise partial and
   complete acceptance as well.
6. Build a tiny feature cache and run one training step; then evaluate a small
   draft before scaling up.

Use tiny architecture-level tests where practical, followed by real-checkpoint
GPU checks. Full target loading and sharding need attention: DeepSpec's inspected
cache/evaluation loaders assume a complete target per worker/device. Its
[data guide][data-guide] already estimates about 38 TB for a default Qwen3-4B
feature cache; do not reuse that data recipe at Flash scale without a measured budget.

## Serving decision

Choose the intended serving engine before optimizing end-to-end performance.
Then add its exact GLM-5.3-Flash implementation and speculative state-management
code as another pinned, selectively indexed source input.

DeepSpec's OpenAI-compatible answer-generation endpoint is only one data pipeline
stage. It does not provide the hidden features and cache controls required for
this integration. The public evaluator measures acceptance statistics rather
than end-to-end latency, so serving integration must separately measure target-only
and speculative latency/throughput under the same workload and hardware.

Open decisions: serving engine and GPU/sharding environment; whether the first
milestone covers text prompts only or also image/video conditioning; the mHC
feature contract; and the recurrent-state recovery mechanism.

## Verification boundary

Two GPT-6.1 Sol researchers inspected the target and DSpark source groups.
No model tensors were downloaded, and no runtime, GPU, training, or latency
checks were performed. The inspected revisions are not validated as a working set.

[model-card]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/README.md
[model-config]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json
[model-doc]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/docs/source/en/model_doc/glm5_next.md
[modeling]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py
[cache-utils]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/cache_utils.py
[deepspec-readme]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md
[requirements]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/requirements.txt
[cache-builder]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/prepare_target_cache.py
[draft-common]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py
[verifier]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py
[dspark-trainer]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/trainer/dspark_trainer.py
[parser]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/parser.py
[data-guide]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/README.md
[hub-download]: https://huggingface.co/docs/huggingface_hub/guides/download
[python-baseline]: https://github.com/PhillipChaffee/67-sus-95-clean/tree/main/python
