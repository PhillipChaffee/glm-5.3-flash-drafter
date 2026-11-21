# What makes a target boundary valuable to the drafter

Source review: 2026-11-21, revised the same day. Supports
[Define the broad-capture and ELSI comparison protocol][elsi-decision].
The user asked which questions and statistics can inform the choice of target
layers for the drafter. The user then asked for a note that starts from the
GLM-5.3-Flash architecture and does not lean on earlier project analysis.

This note describes how the target model computes. It then asks what gives one
boundary value to the drafter, and how to measure that value. It selects no
feature set and changes no approved contract.

## Answer

The target model is not a stack of equal layers. It repeats a four-block
pattern with two different kinds of attention. That pattern gives each
boundary a different job.

A boundary has value to the drafter for eight separate reasons, listed below.
Depth alone explains only some of them. The kind of block that wrote the
boundary, and the kind that reads it next, explain others.

The target model also exposes direct evidence that needs no trained drafter.
It shows where it reads from earlier tokens, how far back it reads, and how
much each block writes. A study of the target itself can answer several
questions before any drafter training.

## Terms in this note

This note uses the canonical terms in the [glossary](../../CONTEXT.md). It uses
`B_k` and `M_k` as the [approved baseline](first-drafter-recommendation.md)
defines them. The glossary does not define the terms below. They are local to
this note until the project adopts them.

| Term | Meaning in this note |
| --- | --- |
| Block | One decoder layer of the target model, numbered from zero. |
| Boundary | The hidden state after a completed block. `B_k` follows block `k-1` and feeds block `k`. |
| Sublayer input | The single vector that the input mapping builds for one sublayer. |
| Anchor | The newest known token, at position `a`. |
| Row | One of the eight drafter inputs. Row `j` predicts token `a+j+1`. |
| Lookahead | The count of positions from a context feature to the token to predict. |
| Probe | A small model trained to read one quantity from stored features. |
| Phase | The index of a boundary modulo four. |

## How the target model computes

Every fact in this section comes from the pinned configuration and the pinned
Transformers source. [Target configuration][glm-config] · [Model source][glm-model]

### Four streams and two sublayers

The target model copies each token embedding into four residual streams. Each
of the 45 blocks then runs an attention sublayer and an MLP sublayer.

Around each sublayer, mHC does three things. The input mapping combines the
four streams into one sublayer input. The output mapping spreads the sublayer
result over the four streams. The residual mapping mixes the old streams.

The target model computes all three sets of weights again for each token, from
the four streams of that token. A sublayer reads a weighted sum, and the
weights change from token to token. [mHC module][glm-mhc] · [Block][glm-block]

### Two kinds of attention

Only attention moves information between positions. The target model uses two
kinds, and they read the past in different ways.

| Property | Linear attention | Sparse attention |
| --- | --- | --- |
| Blocks | 34 blocks | 11 blocks: `[3,7,11,15,19,23,27,31,35,39,43]` |
| How it reads the past | One memory matrix per head. Each token updates it. | Direct attention to selected earlier tokens. |
| Size limit | 64 heads, each a 128 by 128 matrix, at any prefix length | 512 groups of four tokens, plus up to three newest tokens |
| Forgetting | A forget gate decays the memory at each token | None. A selected token is read exactly. |
| Local context | A causal convolution over four tokens | None beyond the selection |
| Position signal | The order of the memory updates | None. The configuration sets `mla_use_nope`. |

A linear-attention block compresses the whole prefix into a fixed memory. It
writes each token with a learned strength and reads the memory with a query.
A sparse-attention block fetches exact content. An indexer scores groups of
four earlier tokens and keeps at most 512 groups.
[Linear attention][glm-kda] · [Indexer][glm-indexer] · [Sparse attention][glm-attn]

One consequence matters for any measurement. A prefix under about 2,048 tokens
fits inside the selection. In that case, each sparse-attention block reads
every earlier token. The selection only starts to matter on longer prefixes.

### Two kinds of MLP

Blocks 0 to 2 use one dense MLP of width 12,288. Blocks 3 to 44 use a mixture
of experts. A router scores 288 experts from the sublayer input and keeps
eight. One shared expert runs for every token. [Router and experts][glm-moe]

The router choice is a discrete record. For each token and block, it names the
eight experts that the target model used.

### The readout and the unused module

After block 44, the target model averages the four streams without weights.
It applies the final RMSNorm and the output head. No block reads an unweighted
mean before that point. [Final readout][glm-final]

The pinned checkpoint index also lists a block at index 45 with the tensors
`eh_proj`, `enorm`, and `hnorm`. The configuration sets
`num_nextn_predict_layers` to 1. The pinned Transformers source excludes this
module. [Checkpoint index][glm-index] · [Transformers note][tf-mtp]

DeepSeek-V3 describes a module with matching parts. It combines the last
representation of token `i` with the embedding of token `i+1`, and it predicts
token `i+2`. This note cannot show that this target uses the same design.
[DeepSeek-V3 report, section 2.2][v3]

### The four phases

Each boundary is the output of one block and the input of the next. The
four-block pattern gives four cases.

| Phase | Written by | Read next by | Boundaries |
| --- | --- | --- | --- |
| 0 | Sparse attention block | Linear attention block | `B4`, `B8`, and so on to `B44` |
| 1 | Linear attention block | Linear attention block | `B1`, `B5`, and so on to `B45` |
| 2 | Linear attention block | Linear attention block | `B2`, `B6`, and so on to `B42` |
| 3 | Linear attention block | Sparse attention block | `B3`, `B7`, and so on to `B43` |

`B45` is phase 1, and the final readout reads it. The six-boundary control
`[20,28,32,36,40,44]` contains only phase 0.

## What the drafter gets and lacks

The drafter sees no earlier tokens. It embeds only the anchor and the mask
rows. Every earlier position reaches it only as context features.
[DeepSpec forward][ds-forward]

The drafter concatenates the context features of one position. One linear
projection reduces them to width 4,096. Each draft layer turns that one vector
into attention keys and values. [DeepSpec projection][ds-fc]

Three limits follow from the approved contract and the DeepSpec mask:

- The drafter reads at most 2,048 earlier positions. The target model accepts
  up to 1,048,576 positions.
- The drafter has five layers. The target model ran up to 45 blocks to build
  each feature.
- The newest context feature is at position `a-1`. Row 0 predicts token
  `a+1`, so its lookahead is 2. Row 7 has a lookahead of 9.

The drafter has its own position encoding. It does not need the context
features to carry position. [DeepSpec mask][ds-mask]

## What makes a boundary valuable

Each property below is a separate reason for value. One boundary can be strong
on one property and weak on another.

### 1. It predicts the tokens that the rows must propose

At position `a-1`, the last block computed the distribution for token `a`. The
drafter already holds token `a`. Value comes from what the boundary tells
about tokens `a+1` to `a+8`, after token `a` is known.

Two studies on other models report that this differs from next-token
prediction. One finds a middle-depth peak for lookahead 2 on GPT-J. The other
finds that linear probes fall to a unigram baseline at lookahead 8. Both read
one position with a small reader. [Future Lens][future-lens] · [Lookahead probes][plan]

If the unused module follows the DeepSeek-V3 design, training shaped `B45` for
lookahead 2. That is the task of row 0.

### 2. It holds work that the drafter cannot repeat

A deep boundary holds the result of many blocks. A five-layer drafter cannot
compute that result again. A shallow boundary holds less work, and the drafter
can build more of it alone.

The sparse-attention blocks do a specific kind of work. Each one fetches exact
content from selected earlier tokens. A phase 0 boundary holds that fetched
content before later blocks mix it.

### 3. It carries context from outside the drafter window

The drafter reads 2,048 positions. Older context reaches it only inside the
features of recent positions.

Two paths bring old context to a recent position. The linear-attention memory
holds a decayed summary of the whole prefix. The indexer can select groups
from any earlier point. A boundary has value when it holds what those paths
delivered.

### 4. It works as an attention key

The drafter uses context features as keys. A row must find the earlier
positions that matter. A feature that is almost equal at every position is a
weak key.

The target model trains some boundaries for this job. A phase 3 boundary feeds
a sparse-attention block. That block builds its keys, its values, and its
indexer scores from the sublayer input. No other phase feeds a direct lookup.

### 5. It keeps the identity of the token at its position

The drafter needs earlier tokens for copied names, numbers, and code. It has
no token embeddings for them. At least one context feature must keep which
token stood at each position.

The first blocks are the likely place. They are close to the embedding, and
blocks 0 to 2 use a different MLP from the other blocks.

### 6. One linear projection can read it

The projection reads a concatenation of raw stream means. A boundary with
large values gets more weight at the start of training. A boundary whose
content sits in stream differences loses that content in the mean.

The target model never reads `M_k` for `k` below 45. It reads a weighted sum
with weights for each token. The sublayer input is the vector that the target
model uses, and the target model computes it in every pass.

### 7. It adds what the other chosen features lack

Two boundaries with one block between them differ by what that block wrote.
Two distant boundaries differ by many writes. The value of a boundary depends
on the set that already contains it.

### 8. The serving engine can supply the same values

A context feature has value only if online capture equals offline capture. A
stream mean after a block is one capture point. A sublayer input or a router
choice is another, and each needs its own engine support.

## How to measure that value

The measures are in order of cost. The first three groups need no trained
drafter.

### Read the target model during a normal pass

These measures record what the target model already computes.

| ID | Measure | Property | What it shows |
| --- | --- | --- | --- |
| R1 | Write size: norm of the attention result and of the MLP result, against the norm of the streams, for each block | 2, 7 | Which blocks change the state most |
| R2 | Reading distance: for each sparse-attention block, attention weight by distance back | 3, 4 | At which depth the target model fetches far content, and the share beyond 2,048 positions |
| R3 | Memory gates: write strength and forget-gate decay for each linear-attention block | 3 | How long each block keeps a token in memory |
| R4 | Router record: the experts chosen for each token and block | 1, 5, 7 | A discrete summary. Its link to later tokens can be counted directly. |
| R5 | Stream spread: share of variance between streams, and input-mapping weights by depth | 6 | How much an unweighted mean discards |
| R6 | Scale and sameness: root mean square of `M_k`, and mean cosine between positions | 4, 6 | Whether a boundary can dominate the projection, and whether it can work as a key |

R4 has a rare advantage. The record is discrete, so counts give its mutual
information with later tokens. No probe is needed.

### Change the target model and watch its outputs

These measures test what the target model itself uses. They need forward
passes with hooks, and no training.

| ID | Measure | Property | What it shows |
| --- | --- | --- | --- |
| C1 | Write removal: replace the result of one sublayer with its mean at positions up to `t`, then measure the change in the target distributions at `t+1` to `t+8` | 1, 2 | Which writes the later predictions depend on |
| C2 | Far edit: change tokens at a chosen distance back, then measure the change in each boundary at the newest position | 3 | How far back each depth reaches |
| C3 | Selection limit: repeat any measure on prefixes below and above 2,048 tokens | 3, 4 | Whether a result holds once the indexer must choose |

C1 ranks sublayers by use and not by readability. A probe can read information
that the model does not use. One of the cited studies reports this gap.
[Lookahead probes][plan]

### Train small readers on stored features

These measures need features for each token and the teacher state.

| ID | Measure | Property | What it shows |
| --- | --- | --- | --- |
| P1 | Depth trajectory: decode each boundary and compare with the teacher distribution at the same position | 1 | Where the target model settles its next token |
| P2 | Lookahead regression: predict the teacher state at `t+d-1` from features at `t` plus the embedding of token `t+1` | 1, 7 | What a boundary adds about later tokens once the anchor is known |
| P3 | Identity probe: predict the token at `t` from `M_k` at `t` | 5 | Which boundaries keep token identity |
| P4 | Key test: one attention head over `M_k` must find the earlier position of a repeated token | 4 | Whether a boundary works as a key |
| P5 | Reduced-rank form of P2 with a fixed number of directions | 6 | How much one 4,096-wide projection can keep |

P2 scores a boundary by its gain over the anchor embedding alone. The same
regression on a set scores what one more boundary adds.

### Train drafters

| ID | Measure | Property | What it shows |
| --- | --- | --- | --- |
| D1 | One-layer drafter with the approved inputs, trained on one boundary or a small set | 1 to 7 | Acceptance for each row at low cost |
| D2 | Matched five-layer runs with repeated seeds | All | The quantity that the choice is for |
| D3 | Feature removal on a trained drafter | 7 | Which features one trained drafter relies on |

D2 needs the spread between seeds before it can rank two sets.

## Hypotheses that the architecture suggests

Each hypothesis follows from the structure above. None is tested. Each names
the measure that can refute it.

| ID | Hypothesis | Refuted if |
| --- | --- | --- |
| H1 | Phase 0 boundaries hold fetched content and lead on lookahead for tokens that depend on far context | P2 and C1 show no phase effect at equal depth |
| H2 | Phase 3 boundaries make the best keys | P4 and R6 show no advantage for phase 3 |
| H3 | Boundaries after blocks 0 to 2 keep token identity best | P3 peaks deeper |
| H4 | `B45` gives the most for row 0 | P2 at lookahead 2 ranks another boundary higher |
| H5 | The sublayer input beats the unweighted mean of the same boundary | R5 shows small stream spread, or P2 shows no gain |
| H6 | Results from prefixes under 2,048 tokens do not transfer to long prefixes | C3 shows equal rankings on both |
| H7 | Deep sparse-attention blocks read nearer tokens than shallow ones | R2 shows equal distances by depth |

H1 and H2 pull in opposite directions. A set can need both a phase 0 boundary
and a phase 3 boundary at similar depth. The six-boundary control cannot test
either hypothesis, because it holds one phase.

H6 affects every measurement plan. A calibration set of short samples never
makes the indexer choose.

## Knowledge that is out of reach today

| Knowledge | Why it helps | Why it is out of reach | Nearest stand-in |
| --- | --- | --- | --- |
| Accepted length of every feature set after full training | Gives the true ranking | 45 boundaries give 8,145,060 sets of six | D1, then D2 on a shortlist |
| What the unused module reads, and how training used it | Settles H4 without a probe | The pinned source excludes the module | Inspect a serving engine that runs it |
| Why the target designers chose this pattern, and which jobs they meant each block kind to do | Turns H1, H2, and H7 into facts | The model card states the design and no reasons | R1 to R3 and C1 to C3 |
| Exact information that a boundary holds about later tokens | Removes the dependence on any probe | No reliable estimate exists for wide vectors | P2 for the usable part, R4 for a countable part |
| The ceiling for a drafter that reads every stream of every boundary | Bounds what the choice can gain | Storage and projection width | A wide-pool drafter on a reduced set |

## Verification boundary

This review inspected the pinned target configuration, the checkpoint index,
and the Transformers model source. It read the mHC module, both attention
kinds, the indexer, the router, the block, and the final readout. It inspected
the DeepSpec mask and projection at the pinned revision.

It ran arithmetic only: the phase of each boundary, the phase of the control,
and the count of sets. It loaded no weights. It ran no capture, probe,
intervention, or training.

Every statement about what a boundary holds is reasoning from structure. The
hypotheses are untested. Every measure in this note is a proposal.

A fetch helper returned the text of the three cited papers. It returned no
values from figures. This review did not consult the earlier ELSI analysis of
this project.

## Sources

[elsi-decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/9
[glm-config]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json
[glm-index]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/model.safetensors.index.json
[glm-model]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py
[glm-moe]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L85-L206
[glm-mhc]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L218-L336
[glm-kda]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L463-L770
[glm-indexer]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L773-L915
[glm-attn]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1101-L1236
[glm-block]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1279-L1349
[glm-final]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1492-L1511
[tf-mtp]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/docs/source/en/model_doc/glm5_next.md?plain=1#L26
[ds-mask]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py#L78-L107
[ds-fc]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py#L240-L245
[ds-forward]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py#L361-L407
[future-lens]: https://arxiv.org/html/2311.04897
[plan]: https://arxiv.org/html/2605.07984
[v3]: https://arxiv.org/html/2412.19437
