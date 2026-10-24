# First drafter: published design versus our old plan

Source review: 2026-10-24. Supports
[Choose the first drafter recipe and feature contract][decision].
The user requested this comparison before selecting a recipe. The subsequent
decision is recorded in the [approved baseline](first-drafter-recommendation.md).

## Main finding

The published Red Hat drafter and our old implementation share much of their
design: five dense transformer layers, the same hidden and attention dimensions,
averaged target streams, a rank-256 vanilla Markov head, and a confidence head.
Both fresh-training paths construct a new backbone and copy frozen target
embeddings and output weights.

The differences extend beyond six versus seven features and RoPE versus NoPE.
The published configuration selects **causal sliding-window attention**, while
our old implementation uses **bidirectional attention within each draft block**
and the full earlier target prefix. Their losses also differ in both hard-label
choice and distribution-distance scale. These are separate experimental variables.

The published checkpoint has reported acceptance after substantial training.
Our old result was a 20-step smoke run. Those results do not establish which
architecture is better under matched training.

## How both designs work

A context feature is an internal representation from the frozen target model.
The drafter reads several such representations for each earlier token,
concatenates them, and projects them to its own hidden width. Each draft layer
uses this projected context to form attention keys and values.

The drafter also receives an anchor token followed by mask tokens. The anchor
is the known token immediately before the continuation being proposed. One
transformer pass produces a base prediction for each row. A small sequential
head then adjusts each prediction using the preceding token: during training,
that preceding token comes from the recorded sequence; during generation, it
is the token just sampled. The backbone does not rerun for each sampled token.

The vanilla Markov head implements this adjustment as a rank-256 token embedding
followed by a vocabulary-wide linear projection. A separate confidence head
uses the backbone state and the previous-token embedding to predict acceptance.
Having that head does not establish that adaptive scheduling was used or helped
in a particular evaluation. [Published forward path][public-forward] ·
[Published sequential head][public-heads] · [Old forward path][old-forward] ·
[Old sequential head][old-heads]

## Architecture comparison

The old column describes the committed seven-feature recipe, which superseded
the original five-feature training plan provisionally. Serialized configuration
and inspected constructors establish these values; saved tensors were not loaded.

| Component | Published Flash DSpark | Our old implemented recipe |
| --- | --- | --- |
| Backbone | Five dense Qwen3-style layers | Five custom dense layers |
| Hidden / MLP width | 4,096 / 12,288 | 4,096 / 12,288 through the target-config builder |
| Attention dimensions | 64 query heads, 64 KV heads, head dimension 64 | Same effective dimensions |
| Shared details | SiLU MLP, Q/K RMSNorm, no attention dropout | Same |
| Position handling | RoPE: relative-position rotations; theta 10,000 | NoPE: no explicit positional encoding; discards position IDs |
| Target-context visibility | Same-document features before the anchor, limited to the preceding 2,048 positions | Features before the anchor, without that sliding-window bound |
| Within-block visibility | Causal: a row sees itself and earlier synthetic rows | Bidirectional: every row sees the whole synthetic block |
| Block input | Anchor plus seven masks: eight rows | Anchor plus six masks: seven rows |
| Proposed tokens | Eight; anchor row predicts the first | Seven; anchor row predicts the first |
| Context features | Six, concentrated in later target layers | Seven, spanning early and later layers |
| Context projection | 24,576 to 4,096, then draft RMSNorm | 28,672 to 4,096, then draft RMSNorm |
| Sequential correction | Vanilla Markov, rank 256 | Same |
| Confidence head | Enabled; 4,352 inputs to one scalar | Same dimensions; enabled |
| Vocabulary width | 154,880 | 154,880 |
| Mask token | 154856 | 154821, the tokenizer's `[MASK]` token |
| Target stopping IDs | 154820, 154827, 154829 | Same IDs in the inspected target metadata |
| Target identity | Release configuration embeds snapshot `eb9eb208…` in a publisher-local path | Old loader did not pin the target revision |

Sources: [Published configuration][public-config],
[published attention mask][public-mask], [old recipe][old-recipe],
[old configuration builder][old-builder], [old model][old-model],
[old shared attention mask][old-common], and [target generation configuration][stops].
The old config class's MLP default of 10,240 is not the builder-selected value;
the old attention implementation also resolves `head_dim=0` to 64.

### Attention: an important correction to the generic description

The public configuration sets all five layers to `sliding_attention`, window
2,048, and `sliding_window_non_causal=false`. The inspected mask therefore
restricts each synthetic row to itself and earlier rows. Generic descriptions
of DSpark's bidirectional block do not describe this selected configuration.

Both designs can still compute all their base predictions in parallel: the
synthetic input rows are already known. In the old design, seeing later mask
rows does not expose the actual future tokens. [Published mask][public-mask] ·
[Old mask and block construction][old-common]

There is a useful source-derived consequence of the old combination. With no
positional encoding, identical mask embeddings, zero dropout, and identical
visibility, its six mask rows remain identical through deterministic backbone
operations. Thus they have the same base logits. The previous-token Markov
correction can still produce different final predictions. RoPE and the causal
mask give the published backbone ways to distinguish draft positions.
This is a structural inference, not a measured explanation of the old low
acceptance, and not a claim that NoPE cannot learn. [Old forward][old-forward] ·
[Old model][old-model] · [Old common code][old-common]

## Features: compare meanings, not matching numbers

Let boundary `B_k` mean the completed state after blocks `0` through `k-1`,
before block `k`. GLM has four mHC residual streams. Both designs average those
streams, but the capture must include the completed block's residual updates.

| Contract | Published design / current capture interpretation | Old implemented recipe |
| --- | --- | --- |
| Serialized IDs | `[20,28,32,36,40,44]` | `[1,12,23,34,42,43,44]` |
| Context boundaries | Mean of `B_20,B_28,B_32,B_36,B_40,B_44` | Mean of `B_2,B_13,B_24,B_35,B_43,B_44`, then normalized mean of `B_45` |
| Context normalization | Raw stream means, followed by normalization after the learned context projection | First six are raw means; final feature already has target final normalization; projection has its own norm |
| Separate teacher state | Mean of `B_45`, before target final norm | Mean of `B_45`, already target-final-normalized |
| Teacher logits | Apply copied target final norm once, then copied output head | Apply copied output head directly; no extra target norm |

For example, public boundary 20 follows completed zero-based block 19. Old ID
23 follows completed block 23, so it corresponds to boundary 24. Old ID 44 is
exceptional: it means the normalized final readout, not raw block-44 output.
Both recipes store a separate teacher state; the public recipe has six context
features plus teacher supervision, not seven conditioning features.
[Current public capture][public-capture] · [Public dataset split][public-data] ·
[Public teacher projection][public-backbone] · [Old feature helper][old-features] ·
[Old teacher alignment][old-forward]

The publisher's cited historical extraction commit lacks the GLM auxiliary
capture path present in the newer inspected implementation. The table records
the current source-supported interface; it does not reconstruct the missing
historical extraction patch. [Historical target implementation][historical-capture]

## What we originally planned

The old [architecture decision][old-architecture-decision] selected five dense
layers, block seven, and confidence prediction as starting choices informed by
the reference implementation and available research. It proposed training with
five spread-out features, `[1,12,23,34,44]`, considering the top-three set later,
and caching their seven-feature union in one target pass.

The [later feature decision][old-feature-decision] recognized that the code and
smoke run actually used all seven features. It kept seven provisionally, pending
feature-parity checks and accepted-length comparisons. Corrected offline layer
analysis generated more candidates; the [online comparison][old-online-decision]
remained open. Neither five-spread nor seven-union became a measured online winner.
Changing a configuration alone also could not select five columns from the old
seven-column cache: its reader required exact feature-list equality.

The [vanilla-versus-RNN decision][old-head-decision] retained the simpler vanilla
head after a Qwen proxy comparison. That was a useful prior for the first run,
not a GLM-specific proof. Our plan was not to copy GLM's full hybrid
attention/MoE/mHC backbone into the drafter.

## Initialization: similar intent, different details

Both inspected fresh-training paths construct new draft backbone weights and
copy frozen target token embeddings and output-head weights. Neither copies
the target decoder blocks. Using the published architecture for fresh training
is distinct from loading and fine-tuning the released drafter checkpoint.

The old constructor creates the context projection, backbone, Markov head, and
confidence head before framework `post_init()`. Its declared initialization
path applies the framework initialization to them together. The public DSpark
constructor creates Markov and confidence modules after its parent backbone's
`post_init()`, leaving those later modules with their ordinary PyTorch
constructor initialization in the inspected fresh path. Therefore identical
head dimensions do not imply identical initialization.

The public training path also loads a frozen target final norm for its
pre-norm teacher states. The old teacher states are already normalized and
need no copied teacher norm. The publisher reports a common fresh initialization
but does not provide its exact initialization artifact; the export's recorded
Transformers version also falls outside the cited training project's dependency
range. Exact historical initialization remains unverified.
[Public constructor][public-constructor] · [Public parent constructor][public-backbone] ·
[Public target-weight loading][public-loading] · [Public card][public-card] ·
[Public dependencies][public-dependencies] · [Old constructor][old-model] ·
[Old target-weight loading][old-trainer]

## The training objectives are not identical

Both recipes advertise a small cross-entropy term and a larger distribution
matching term, plus confidence training. The inspected loss definitions differ:

| Term | Published Speculators recipe and loss definition | Old DeepSpec recipe and loss definition |
| --- | --- | --- |
| Cross-entropy, weight 0.1 | Hard label is the teacher's highest-logit token | Hard label is the recorded continuation token |
| Distribution term, weight 0.9 | Total variation: `0.5 * sum(abs(p - q))` | Full probability L1: `sum(abs(p - q))` |
| Confidence | Predict distribution overlap, `1 - TV` | Same overlap target; explicit weight 1.0 |
| Position weighting | Implementation defaults exist; full historical overrides are not recorded in the card | Explicit exponential decay `exp(-position / 4)` |

Here `p` and `q` are teacher and draft token probabilities. For the same
distributions, the old distance is twice TV, so using the same 0.9 coefficient
does not give the same weighting relative to cross-entropy. Recorded sampled
tokens also need not equal the teacher's most likely token. These distinctions
matter when comparing learning curves or migrating a recipe.
[Public reference losses][public-loss] · [Old losses][old-loss] ·
[Public recipe][public-card] · [Old recipe][old-recipe]

The old generation/replay conditioning mismatch is a separate data-contract
issue: generation used low effort with earlier reasoning cleared; replay
omitted those options and used the inspected template's max-effort,
retained-reasoning defaults. That does not prove the backbone or averaged
features cannot learn. See the [existing feasibility review](dspark-flash-feasibility.md)
for its off-policy-data interpretation.

## What the results establish

| Evidence | Published checkpoint | Old project |
| --- | --- | --- |
| Training scale | Publisher reports 1,723,733 prepared examples, length cap 8,192, and a three-epoch run; release is epoch 2, step 270,979 | Intended large-data, ten-epoch run with length cap 4,096; historical smoke used 686 cached examples and 20 optimizer steps |
| Training settings | Publisher reports Muon, learning rates 0.001, and regenerated data with low/high/max effort | Committed learning rate 0.0006, global batch 512, and low-effort generation |
| Acceptance | Publisher reports aggregate length 3.77 and HumanEval 4.74 with eight proposals | Historical small serving smoke reports about 1.05 greedy and 1.01 sampled |
| Meaning | Reported learning and acceptance for a substantially trained whole recipe | Reported optimizer and serving liveness; not a completed quality experiment |

The public aggregate appears bonus-inclusive from `1 + 8 * 0.3467`, but metric
definitions must be reconciled before comparing values. Training exposure,
data, masks, objective, runtime, and evaluation also differ. Neither the public
result nor our old smoke isolates the effect of one architectural choice.
[Public training and evaluation card][public-card] · [Old training report][old-smoke] ·
[Old serving report][old-serving]

## Implications considered in the decision

The public recipe is a stronger empirical starting reference than our old
smoke, but selecting it would change several variables at once. It would be
reasonable to establish one complete starting recipe and then vary one choice
at a time. That recommendation informed the subsequently approved baseline.

The first differences worth discussing are position/attention behavior, feature
selection and final-state normalization, and the training objective. Backbone
depth and width, vanilla-head rank, and confidence-head dimensions are already
largely shared. Full-stream features and RNN heads are possible later
experiments, rather than prerequisites to understanding these two candidates.

## Verification boundary

Two `openai/gpt-6.1-sol` researchers inspected the published and historical
source groups. Synthesis cross-checked the published configuration and mask,
the old recipe and forward path, and both loss definitions. Evidence consists
of inspected source, publisher claims, historical issue reports, and explicitly
labelled structural reasoning. No model tensors, cache tensors, training,
serving, CPU model checks, or GPU checks were executed for this comparison.

Missing historical extraction and initialization details limit exact public-run
reproduction. A matched experiment is still needed to attribute quality to
RoPE, attention layout, tap selection, block length, or any other individual choice.

## Sources

[decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3
[public-config]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/config.json
[public-card]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/README.md
[public-mask]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dflash/attention.py#L42-L100
[public-forward]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dspark/core.py#L151-L211
[public-constructor]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dspark/core.py#L38-L94
[public-heads]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dspark/model_definitions.py#L12-L91
[public-backbone]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dflash/core.py
[public-loading]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/model.py#L144-L199
[public-data]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/train/data.py#L337-L359
[public-capture]: https://github.com/vllm-project/vllm/blob/49d634bd6a3dc7433d8fb464821c95c76db85350/vllm/models/glm5next/common/model.py#L773-L856
[historical-capture]: https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py#L578-L704
[public-loss]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/losses/eager.py#L92-L142
[public-dependencies]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/pyproject.toml#L58-L62
[stops]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/generation_config.json
[old-recipe]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/config/dspark/glm5_flash.py#L13-L62
[old-builder]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/config.py#L39-L92
[old-model]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py#L35-L334
[old-forward]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py#L419-L569
[old-trainer]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/trainer/glm5_trainer.py#L60-L131
[old-features]: https://github.com/PhillipChaffee/GLM-5/blob/5e87e16dfc1fd7c2f1e4b02b70ef85ba09796a18/experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/hidden_states.py#L36-L100
[old-common]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py#L78-L294
[old-heads]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py#L8-L90
[old-loss]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py#L60-L252
[old-architecture-decision]: https://github.com/PhillipChaffee/GLM-5/issues/8#issuecomment-6054521571
[old-feature-decision]: https://github.com/PhillipChaffee/GLM-5/issues/33#issuecomment-6123350107
[old-online-decision]: https://github.com/PhillipChaffee/GLM-5/issues/52#issuecomment-6157271732
[old-head-decision]: https://github.com/PhillipChaffee/GLM-5/issues/14#issuecomment-6063603729
[old-smoke]: https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6081455333
[old-serving]: https://github.com/PhillipChaffee/GLM-5/issues/47#issuecomment-6144822966
