# DeepSeek's current DSpark design choices

Source review: 2026-10-24. Supports
[Choose the first drafter recipe and feature contract][decision].
The user asked why bidirectional attention helps with masked future inputs and
what DeepSeek chooses for its latest drafters. Two direct subagents inspected
the native releases and public reference recipes after the user requested
cancelling the workflow-based investigation.

## Answer

DeepSeek's native V4 Flash and Pro DSpark releases use **partial RoPE and
bidirectional draft-block attention**, three MoE draft layers, three late target
features, and a direct target-prefix window of **128 positions**. Their native
inference configs and production paper use **five proposals**.

DeepSeek's public Qwen3/Gemma4 reference checkpoints use **RoPE and bidirectional
blocks**, five dense draft layers, five target features, **seven proposals**,
and the full supplied earlier prefix. They are different recipes, not complete
public reproductions of native V4 training.

The earlier [Red Hat GLM comparison](first-drafter-design-comparison.md) describes
a third recipe: RoPE, causal sliding attention, six features, eight proposals,
and a 2,048-position direct-prefix window. Its choices do not define DSpark as
an algorithm.

## Why let mask positions attend bidirectionally?

A mask token is the initial input to a position, not a description of every
hidden vector that position will have throughout the network.

1. Future positions start with the same mask-token embedding.
2. RoPE gives their attention queries and keys different positional rotations.
3. They can attend differently to the real anchor and known target context,
   producing different context-conditioned representations.
4. Later layers can exchange those representations in both directions across
   the block. Information computed at a later position can influence an
   earlier one.
5. After the parallel backbone pass, the sequential head samples proposals and
   adjusts each next distribution using the actual previous sampled token.

Thus bidirectional attention permits communication between contextualized
future-position representations. It does **not** provide the actual future
sampled tokens to the backbone. The paper explicitly identifies missing
sampled-token conditioning as the reason for the sequential correction head.
[Paper's parallel-backbone discussion][parallel] ·
[Native draft attention][native-attention] · [Reference block mask][common]

This is a mechanical explanation, not an isolated measured benefit. Neither
source review found a causal-versus-bidirectional attention ablation in the
DSpark paper. Its reported ablations cover depth, proposal length, sequential
heads, and confidence scheduling.

For the old custom GLM combination of NoPE, identical mask embeddings, identical
visibility, and deterministic shared operations, the mask rows remain
structurally symmetric. Native partial RoPE and reference RoPE remove that
enforced symmetry. This does not prove RoPE or bidirectionality caused any
particular acceptance result. A causal mask can also compute all known
anchor/mask input rows in parallel.

## Release identity and recency

The current official Hub author listing returned native Flash/Pro DSpark and
the four Qwen/Gemma checkpoints below. Native repositories were last modified
July 4; reference checkpoints June 28. DeepSpec HEAD was still
`005e03b81cec38b7da6399833d609ee89a2587f2`, dated July 9; its GitHub releases
endpoint returned no releases. These are the latest official releases found
by this check, not a claim about unpublished internal work.

| Official repository | Inspected revision |
| --- | --- |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | `62af8fffb2f7030cac4de2f0169f5b8d1101b646` |
| `deepseek-ai/DeepSeek-V4-Pro-DSpark` | `7c09739fd136abfb70a49ec334157f65f45b52cd` |
| `deepseek-ai/dspark_qwen3_4b_block7` | `3457dff1417cb84927f6098a5fcb7cee85c934b7` |
| `deepseek-ai/dspark_qwen3_8b_block7` | `03326e5043815da1f81b109078b2889737c26017` |
| `deepseek-ai/dspark_qwen3_14b_block7` | `83207b416acf99f41c2184648923632fccea6dd0` |
| `deepseek-ai/dspark_gemma4_12b_block7` | `2fa72e765eec2965fc4d86a8663ce6769eba6218` |

The native cards cite the [DeepSeek-V4 technical report][v4-paper], which
describes the target and its original MTP strategy. The separate
[DSpark paper][dspark-paper], especially section 5, describes native DSpark
deployment. Target configuration fields and original MTP fields must not be
read as a complete description of the attached DSpark module.

## Native Flash and Pro

| Choice | V4 Flash DSpark | V4 Pro DSpark |
| --- | --- | --- |
| Draft layers | Three MoE layers, four mHC streams | Same |
| Hidden width / expert MLP width | 4,096 / 2,048 | 7,168 / 3,072 |
| Experts per draft layer | 256 routed, top six, plus one shared | 384 routed, top six, plus one shared |
| Attention heads / head dimension | 64 / 512, shared KV representation | 128 / 512, shared KV representation |
| Position encoding | RoPE on 64 of 512 head dimensions; theta 10,000 | Same |
| Within-block attention | Bidirectional | Bidirectional |
| Direct target context | Up to 128 earlier positions plus the synthetic block | Same |
| Configured proposals / inputs | Five proposals; anchor plus four masks | Same |
| Mask token | Shared embedding at ID 128799 | Same |
| Feature IDs | `[40,41,42]` | `[58,59,60]` |
| Feature meaning | Completed last three target blocks, before final target readout | Same |
| Feature reduction | Mean across four completed mHC streams | Same |
| Concatenated context projection | 12,288 to 4,096, then RMSNorm | 21,504 to 7,168, then RMSNorm |
| Sequential head | Markov, rank 256 | Markov, rank 512 |
| Confidence linear input / output | 4,352 / 1 | 7,680 / 1 |
| Production verification | Adaptive confidence/load-aware scheduling | Same |

Sources: [Flash inference config][flash-config], [Pro inference config][pro-config],
[native model][native-model], and [paper section 5.1][deployment].

### Details that can otherwise be misread

**Three DSpark layers, not one.** The top-level target configs retain
`num_nextn_predict_layers: 1`, but the supplied inference configs select
`n_mtp_layers: 3`, and the implementation constructs three `DSparkBlock`s.
The paper independently confirms three layers. The historical `mtp.*` name
does not make this the original one-layer MTP model.

**128 is a direct-feature window, not the target's context capacity.** All
draft layers have zero attention compression. Their explicit gathered indices
include the valid 128-slot target-feature cache and every synthetic block row;
they do not additionally read the target's compressed historical attention
pools. The target itself supports a much larger context, and its recent
features can already carry information from earlier tokens.
[Native indices and attention][native-attention] · [Gathered attention kernel][native-kernel]

**The native draft uses positional information.** Zero compression selects
ordinary RoPE rather than the target's compressed-layer YaRN configuration.
Each synthetic row has its own position. There are no separate per-position
vocabulary heads or position-indexed mask embeddings in the supplied path.

**Five versus seven has a documentation discrepancy.** Native inference
configs and the production paper use five proposals, but both model cards
show vLLM examples requesting `num_speculative_tokens: 7`. That command is not
evidence that native training used seven. The review did not establish the
integration's override behavior or a seven-proposal quality result.
[Native card example][native-card]

**Raw final context is not the teacher readout.** Native feature capture
averages completed mHC streams after selected blocks. The target's final
prediction instead uses its learned mHC readout, final normalization, and LM
head. The paper describes communicating a separate pre-LM-head teacher state
for training; the full native training cache schema is unpublished.
[Feature capture][native-capture] · [Native training description][deployment]

**Confidence scheduling is a production-paper claim.** The minimal supplied
generation demonstration is not the full production verifier/scheduler.
Native code returns raw confidence logits, whereas the paper describes
sigmoid probabilities, post-hoc calibration, and load-aware scheduling.
The confidence head consumes the reduced draft state before its final RMSNorm,
alongside the previous-token embedding. [Native heads][native-heads] ·
[Production scheduler][scheduler]

## Public Qwen3 and Gemma4 reference recipes

All four checkpoints have five dense draft layers, bidirectional synthetic
blocks, seven input rows producing seven proposals, five target features,
rank-256 vanilla Markov correction, and an enabled confidence head.

| Target | Draft hidden / MLP width | Position encoding | Ordered target feature IDs |
| --- | --- | --- | --- |
| Qwen3-4B | 2,560 / 9,728 | RoPE, theta 1,000,000 | `[1,9,17,25,33]` |
| Qwen3-8B | 4,096 / 12,288 | RoPE, theta 1,000,000 | `[1,9,17,25,33]` |
| Qwen3-14B | 5,120 / 17,408 | RoPE, theta 1,000,000 | `[1,10,19,28,37]` |
| Gemma4-12B | 3,840 / 15,360 | Partial RoPE, 128 of 512 head dimensions, theta 1,000,000 | `[5,17,29,41,46]` |

Sources: released [4B][q4-config], [8B][q8-config], [14B][q14-config], and
[Gemma][gemma-config] configs, with [Qwen][qwen-builder] and
[Gemma][gemma-builder] builders and implementations.

Every draft layer is full/global attention. The training mask permits only
target positions strictly before the anchor and all synthetic rows in the same
block. Gemma's saved `sliding_window=1024` is inherited metadata: its custom
full-attention draft path does not apply that window. The 4,096 training
sequence cap and configured maximum positions are also not attention windows.
[Reference mask][common] · [Gemma attention][gemma-attention]

Feature IDs denote completed zero-based decoder blocks: ID 1 corresponds to
boundary B2, for example. Features are concatenated, projected, and draft-
normalized, without an mHC stream-reduction step for these target families.
None of the selected sets includes the final decoder block. The evaluator
rejects that block because raw cache hooks and final-normalized returned hidden
states would otherwise disagree. The separate teacher state is already
normalized and goes directly through the copied output head; Gemma additionally
applies its configured output-logit softcap.
[Reference capture][reference-capture] · [Final-block guard][final-guard]

The reference trainer constructs a fresh draft from target configuration and
copies/freezes target token embeddings and LM-head weights. It does not copy
target decoder blocks. Backbone, context projection, Markov and confidence
modules exist before framework initialization. Fixed seven-proposal
verification is the default; optional confidence-threshold truncation is not
the native paper's load-aware production scheduler. Main paper Table 1 disables
confidence scheduling. [Reference initialization][reference-init] ·
[Proposal and threshold logic][reference-proposals] · [Paper experiments][experiments]

## Losses, feature selection, and unpublished native details

All four public recipes use cross-entropy against recorded continuation tokens
with weight 0.1, full probability L1 with weight 0.9, and confidence BCE with
weight 1.0. Confidence targets are distribution overlap, `1 - 0.5 * L1`.
Position weights are `exp(-j / 4)` for zero-based position `j`; training samples
up to 512 anchors per sequence. [Reference loss][reference-loss] ·
[Reference recipe][reference-recipe]

The paper calls its equation-10 distance TV but writes full L1, consistent with
the reference implementation. Its general decay equation uses proposal length,
while public configs explicitly choose decay constant four. The native
checkpoint-specific optimizer, learning rate, data composition, duration,
sequence cap, initialization/copying strategy, and overrides to general loss
defaults remain unpublished in the inspected sources. The Qwen/Gemma
Open-PerfectBlend recipe must not be attributed to native V4 without evidence.
[Paper training equations][training] · [Native training description][deployment]

Native releases select the last three target blocks. Public reference tap lists
are hard-coded; Qwen's are evenly spaced. Neither review found a published
ELSI procedure or an empirical layer-search account establishing these as
optimal sets. The user's proposed broader capture and six/seven-feature search
remain legitimate experiments, not something these sources resolve in advance.

## Consequence for our baseline decision

**RoPE plus bidirectional block attention has direct DeepSeek precedent.**
The causal mask in the published Red Hat GLM recipe is a distinct implementation
choice. DeepSeek also supplies a production precedent for much shorter direct
context than either our old full-prefix design or Red Hat's 2,048-position window.
Those precedents do not establish the best window or feature count for GLM.

The user subsequently approved an eight-proposal baseline and the broad-capture,
six/seven-feature comparison direction. The selected position encoding, block
attention, direct-context window, and feature contract are recorded in the
[approved baseline](first-drafter-recommendation.md). Detailed data, stack, and
analysis-protocol decisions remain open.

## Verification boundary

Two direct subagents inspected official pinned release metadata, inference
source, reference recipes, and papers. Synthesis also checked both native
inference configs and the Qwen3-8B release config. The abandoned workflow was
cancelled; these findings came from the replacement direct-subagent work.

No weights, cache tensors, model executions, GPU jobs, or causal-versus-
bidirectional experiments were run. Source behavior, author-reported production
choices, and structural reasoning are distinguished above. Native training
details and the five-versus-seven serving-example discrepancy remain open.

## Sources

[decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3
[dspark-paper]: https://arxiv.org/html/2607.05147v1
[v4-paper]: https://arxiv.org/html/2606.19348v1#S2.SS1
[parallel]: https://arxiv.org/html/2607.05147v1#S3.SS1
[training]: https://arxiv.org/html/2607.05147v1#S3.SS3
[experiments]: https://arxiv.org/html/2607.05147v1#S4.SS2
[deployment]: https://arxiv.org/html/2607.05147v1#S5.SS1
[scheduler]: https://arxiv.org/html/2607.05147v1#S5.SS2
[flash-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/config.json
[pro-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/blob/7c09739fd136abfb70a49ec334157f65f45b52cd/inference/config.json
[native-model]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py
[native-attention]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py#L743-L786
[native-heads]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py#L795-L904
[native-capture]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py#L913-L925
[native-kernel]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/kernel.py#L277-L350
[native-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/README.md#L220-L235
[q4-config]: https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/blob/3457dff1417cb84927f6098a5fcb7cee85c934b7/config.json
[q8-config]: https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/blob/03326e5043815da1f81b109078b2889737c26017/config.json
[q14-config]: https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7/blob/83207b416acf99f41c2184648923632fccea6dd0/config.json
[gemma-config]: https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7/blob/2fa72e765eec2965fc4d86a8663ce6769eba6218/config.json
[common]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py#L52-L294
[qwen-builder]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/config.py#L9-L56
[gemma-builder]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/gemma4/config.py#L52-L102
[gemma-attention]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/gemma4/modeling.py#L34-L168
[reference-capture]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/prepare_target_cache.py#L83-L134
[final-guard]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py#L100-L112
[reference-init]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/trainer/base_trainer.py#L251-L282
[reference-proposals]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py#L22-L153
[reference-loss]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py#L25-L252
[reference-recipe]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_8b.py#L10-L57
