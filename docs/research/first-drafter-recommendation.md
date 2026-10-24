# Approved first drafter baseline and mHC rationale

Approved by the user on 2026-10-24 after reviewing the complete proposal.
This is the selected model-and-loss contract for
[Choose the first drafter recipe and feature contract][decision].
The first feature control uses six boundaries; six- and seven-feature candidates
will be compared through broad capture and corrected ELSI/ELSI-C analysis.
Approval records a planning decision, not successful model execution.

## Approved baseline

Train a fresh five-layer dense drafter with ordinary single-stream residuals,
RoPE, bidirectional eight-row anchor/mask blocks, and a 2,048-position direct
target-feature window. Use averaged completed GLM mHC streams as conditioning.
Start with six explicit public feature boundaries as an initial feature control,
then compare six- and seven-feature candidates from corrected layer analysis.
Keep the vanilla rank-256 sequential head and confidence prediction; begin with
fixed verification length. This is the approved experimental baseline, not a
validated combination or a reproduction of the public checkpoint's training run.

## Why DeepSeek chose MoE: what is actually known

The DSpark paper describes its native V4 drafter as three MoE layers with mHC
and a 128-position window. It does not state why MoE was chosen over dense,
why mHC was retained, or why exactly three layers were selected. No isolated
native dense-versus-MoE or target-architecture-matching ablation was found.
[Native DSpark description][native-dspark]

The supplied native implementation reuses target block machinery, including
expert routing and mHC, but replaces the attention path with draft-specific
uncompressed attention. Reuse could help integration, and sparse experts could
offer a useful capacity/active-compute tradeoff. Those are engineering inferences,
not documented author reasons or evidence of a target-matching advantage.
[Native implementation][native-model]

The drafter should model the target's conditional output probabilities well.
That does not require reproducing its internal architecture. Under standard
stochastic verification, per-position acceptance equals the overlap of the
target and proposal distributions, `1 - 0.5 * sum(abs(p - q))`. The comparison
is between distributions, not expert routers or residual connections.
[Acceptance and sequential correction][dspark-method]

A dense drafter can learn from expensive representations the target has already
computed. Red Hat's dense GLM-5.3-Flash drafter provides reported exact-target
feasibility evidence, though not a matched dense-versus-MoE comparison.
[Public GLM card][public-card] · [Public GLM configuration][public-config]

The dense recommendation reuses that concrete starting size and avoids needing
to establish a separate expert-routing/mHC training and serving design for the
first run. It does not claim dense is always faster, cheaper, or more accurate.
A later controlled MoE or mHC comparison can answer those questions if useful.

## mHC from first principles

### Ordinary residual connections

A token's hidden state is its current numerical representation. With an ordinary
residual connection, a sublayer computes an update and adds it to the existing
representation:

```text
next representation = current representation + computed update
```

A decoder block normally does this around attention and again around its
feed-forward computation. The existing representation has a bypass route, so
each sublayer can add a change rather than replace everything.

### Four residual streams in GLM Flash

GLM carries four 4,096-wide representations per token through its text blocks.
The token embedding starts as four identical copies. Learned updates let those
copies diverge; the streams do not have predefined jobs such as syntax or reasoning.

At each attention or feed-forward sublayer:

```text
four incoming streams
   |
   +-- learned weighted combination --> one sublayer computation --+
   |                                                              |
   |                                                distribute its update
   |                                                              |
   +-- constrained mixing of the existing streams -----------------+
                                                                  |
                                                           four updated streams
```

The learned pre-combination produces one vector for the attention or feed-forward
operation. Learned post coefficients distribute its result back to four streams.
A separate learned matrix mixes the old streams on the bypass path, and the
two contributions are added. GLM computes fresh coefficients around attention
and around the feed-forward sublayer. Four residual streams do not imply four
independent copies of the core sublayer computation.
[GLM mHC implementation][glm-mhc] · [Completed decoder block][glm-block]

### What the constraint means

mHC means **Manifold-Constrained Hyper-Connections**. The residual-mixing matrix
is restricted toward nonnegative matrices whose rows and columns each sum to
one. This is a controlled remix of existing streams, intended to keep that
residual path stable across depth. GLM approximates the constraint using
iterative row/column normalization.

The restriction applies to the residual-mixing matrix, not every learned gate:
GLM's pre coefficients do not sum to one, and post coefficients can exceed one.
It is not a guarantee about the stability or information preservation of the
entire network. [mHC paper, revised version][mhc-paper] · [GLM coefficients][glm-mhc]

### Independent concepts

- **mHC:** how a token's representations flow and mix through depth.
- **MoE:** which feed-forward expert networks process a token inside a sublayer.
- **RoPE:** how token positions affect attention calculations.
- **Inference state:** the cached attention or recurrent information about earlier tokens.

These mechanisms can be combined in different ways. Feeding target mHC features
to a dense, single-stream drafter is a valid interface; its quality is empirical.

### GLM's actual final readout

The exact pinned GLM implementation ends with:

```text
four completed final streams
        -> unweighted mean
        -> final target RMSNorm
        -> target vocabulary head
        -> next-token logits
```

The hyper-head only averages; final normalization is a separate operation.
Native DeepSeek V4 instead has a learned final stream reduction. These two
targets must not be given the same final-readout contract by analogy.
[GLM hyper-head][glm-hyper-head] · [GLM final normalization][glm-final]

Averaging intermediate streams loses their differences. Those differences can
matter to later target computation even when the mean is unchanged. At GLM's
final boundary, however, the mean contains everything its immediate norm/head
readout consumes. Averaging is a defensible initial feature reduction, not proof
that intermediate full-stream or learned-reduction inputs would offer no benefit.

## Approved model contract

These choices are approved. Dimensions and several mechanisms have public
precedent; their complete combination has not been trained or served here.
Let `B_k` denote the completed state after zero-based block `k-1`, and let `M_k`
be its unweighted mean over four mHC streams.

| Component | Approved choice |
| --- | --- |
| Target and tokenizer | Exactly `zai-org/GLM-5.3-Flash`, revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` |
| Draft backbone | Five dense transformer layers; hidden width 4,096; gated SiLU MLP width 12,288 |
| Draft residual structure | Ordinary single stream; pre-normalized blocks; no draft MoE or mHC for this initial recipe |
| Attention dimensions | 64 query heads, 64 KV heads, head dimension 64; bias-free attention projections |
| Draft norms and dropout | Q/K RMSNorm; draft RMSNorm epsilon `1e-5`; attention dropout zero |
| Position encoding | Full-head RoPE; theta 10,000; default scaling |
| Positions | Context position `t` stays `t`; anchor at `a` gives synthetic positions `a` through `a+7` |
| Draft input and output | Anchor plus seven copies of the internal mask embedding; eight input rows and eight proposals |
| Block attention | Bidirectional among all eight synthetic rows; exclude other synthetic blocks |
| Target-feature attention | Same document; `a-2048 <= t < a`, shared by every row in the block, in all five layers |
| Initial feature control | Ordered `[M20,M28,M32,M36,M40,M44]`; completed blocks `[19,27,31,35,39,43]` |
| Context projection | Concatenate six vectors; bias-free `24576 -> 4096`, then draft RMSNorm; each draft layer forms its context K/V |
| Seven-feature candidates | Same model except context projection input width becomes 28,672 and its ordered feature definitions change |
| Stream reduction | Unweighted mean of completed mHC streams; no separate target-final-normalization operation on the initial six context taps |
| Separate teacher payload | `M45`, before final target normalization; not an extra conditioning tap |
| Teacher logits | Copied frozen target final RMSNorm applied once to `M45`, then copied frozen output head |
| Token alignment | Synthetic row `j` predicts token `a+j+1`; teacher state comes from position `a+j` |
| Sequential correction | Vanilla Markov, rank 256; previous-token embedding then vocabulary-wide additive logit correction |
| Confidence prediction | Enabled; post-draft-final-norm state plus previous-token rank-256 embedding; biased `4352 -> 1` linear head |
| Initial verification policy | Fixed eight proposals, subject to normal stopping limits; learn confidence without adaptive verification initially |
| Vocabulary | Full 154,880 outputs; separate input embedding and output-head matrices |
| Internal mask ID | 154856, matching the public drafter; a configured embedding row, not the tokenizer's `[MASK]` string |
| Target stop IDs | 154820, 154827, 154829 |
| Fresh initialization | New linear/embedding weights normal with mean zero and standard deviation 0.02; new biases zero; trainable RMSNorm scales one |
| Frozen target components | After new-module initialization, copy and freeze target token embeddings, output head, and teacher final norm; do not copy target decoder blocks |

Sources for component choices: [public configuration][public-config],
[dense model definitions][model-definitions], [forward and teacher alignment][backbone],
[sequential/confidence heads][heads], and [GLM target readout][glm-final].

Bidirectional windowed attention is a deliberate variation from Red Hat's
causal block. The inspected Speculators mask can express it with five
`sliding_attention` layers, `sliding_window=2048`, and
`sliding_window_non_causal=true`. Its window is anchor-relative, not shifted
independently for each synthetic query. Serving must preserve that contract;
the complete export/runtime path remains unvalidated. Stock DeepSpec's common
mask has no window parameter. [Speculators mask][mask]

The target tokenizer calls ID 154821 `[MASK]`. The public drafter's ID 154856
is inside the configured model vocabulary but has no identified token-string
mapping in the inspected tokenizer metadata. It is an internal synthetic input,
not a prompt-rendering token. [Tokenizer][tokenizer] · [Public configuration][public-config]

The explicit new-module initialization avoids accidentally inheriting different
Markov/confidence initialization from DeepSpec versus Speculators constructor
ordering. It is the selected policy, not a claim to recover the public model's
historical initialization artifact. [Constructor comparison](first-drafter-design-comparison.md)

## Approved loss contract

For untempered teacher probabilities `p` and corrected draft probabilities `q`:

```text
TV = 0.5 * sum(abs(p - q))
loss = 0.1 * teacher-argmax cross-entropy
     + 0.9 * TV
     + 1.0 * confidence binary cross-entropy
```

Confidence targets are detached `1 - TV`. Weight position `j` by `exp(-j/4)`
for `j=0...7`; divide weighted sums by the valid-position count. Markov training
still uses the recorded previous tokens, even though the cross-entropy label
is the replay teacher's most likely next token.

The rationale is to make both token and distribution supervision follow the
frozen replay teacher. This is not measured superiority over recorded-token CE.
It follows the inspected Speculators CE/TV meanings; DeepSpec's `0.9 * full-L1`
has twice the distribution-term scale, and its weighted-mean reduction differs.
Those definitions must remain explicit through implementation.
[Loss definitions][loss] · [Loss reduction][loss-reduction]

## Broad capture and six/seven-feature comparisons

1. Capture all 45 completed raw stream means, `M1...M45`, on a representative
   analysis set. Keep normalization status and token alignment explicit.
2. Keep the final normalized readout `R45 = target_norm(M45)` separately labelled.
   A mixed analysis pool may include it as an additional candidate; the primary
   raw-only pool and any mixed-pool rankings must remain distinguishable.
3. Run corrected ELSI/ELSI-C at cardinalities six and seven, with common valid-token
   pooling grids and explicit normalization/granularity settings.
4. Compare shortlisted candidates against the six-feature control under matched
   architecture, loss, training exposure, and initialization of shared modules.
   A different feature count necessarily changes context-projection shape.

Pooled diagnostic statistics are not tokenwise training features. Once candidate
sets are selected, obtain or retain the full per-token features needed for their
training. If `R45` is an analysis candidate, normalize each token before pooling;
normalizing an already-pooled mean does not produce the same representation.
Historical pooled captures also cannot recover discarded stream differences.
[ELSI paper][elsi-paper] · [Corrected analysis record][elsi-fixes]

The pinned Transformers hidden-state tuple replaces the raw last-block entry
with the final normalized state. Capture completed block 44 directly when
`M45` or its full streams are needed; do not relabel the tuple's final entry as
an unnormalized block output. [Output capture behavior][output-capture]

## Remaining planning decisions

Dataset composition, effort/history settings, per-turn supervision masks,
analysis splits, optimizer, learning rate, batch size, sequence cap, duration,
precision, runtime dependencies, sharding, and hardware budget belong to the
remaining data and training-stack decisions. A 2,048-position draft-feature
window does not determine the training sequence length or target context limit.

Next is [Define the first dataset and per-turn training contract][data-decision].
[Choose the training stack and compute setup][stack-decision] owns the runtime
and resource decisions; [Define the broad-capture and ELSI comparison protocol][elsi-decision]
owns the detailed analysis and matched candidate runs.

## Verification boundary

Three direct subagents inspected author rationale, exact mHC behavior, and the
proposed configuration. Synthesis also read the pinned local mHC implementation.
This document contains inspected facts, author-reported results, and the approved
planning choices; it is not an executable experiment configuration.

No model tests, feature captures, learning runs, or GPU comparisons were performed.
The dense-versus-MoE advantage, intermediate stream information loss, best feature
set, and quality/speed of the combined proposal remain empirical questions.

## Sources

[decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/3#issuecomment-6383354884
[data-decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/4
[stack-decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/5
[elsi-decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/9
[native-dspark]: https://arxiv.org/html/2607.05147v1#S5.SS1
[dspark-method]: https://arxiv.org/html/2607.05147v1#S3.SS2
[native-model]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py
[mhc-paper]: https://arxiv.org/html/2512.24880v2#S4
[glm-mhc]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L218-L329
[glm-block]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1279-L1349
[glm-hyper-head]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L332-L336
[glm-final]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1494-L1511
[output-capture]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/utils/output_capturing.py#L234-L305
[public-card]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/README.md
[public-config]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview/blob/1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4/config.json
[model-definitions]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dflash/model_definitions.py#L56-L207
[backbone]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dflash/core.py#L336-L445
[heads]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dspark/model_definitions.py#L12-L91
[mask]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/models/dflash/attention.py#L42-L100
[tokenizer]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/tokenizer.json#L15-L23
[loss]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/losses/eager.py#L92-L142
[loss-reduction]: https://github.com/vllm-project/speculators/blob/b9c51bcfc134923eae2b6a6f8246ae3500d5c42f/src/speculators/losses/utils.py#L290-L304
[elsi-paper]: https://arxiv.org/abs/2610.12939v1
[elsi-fixes]: https://github.com/PhillipChaffee/GLM-5/issues/81#issuecomment-6169823400
