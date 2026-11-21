# Questions and statistics for choosing context features

Source review: 2026-11-21. Supports
[Define the broad-capture and ELSI comparison protocol][elsi-decision].
The user asked which questions to ask before choosing the target layers that
the drafter reads. The user also asked which statistics or knowledge can inform
that choice. The request includes knowledge that nobody can get today.

This note lists questions and statistics. It selects no feature set and changes
no approved contract.

## Answer

The choice has more parts than a list of layer numbers. Three choices define a
context feature: the capture point, the stream reduction, and the
normalization. Published recipes vary only the capture point, and only at
block boundaries.

Four findings carry most of the weight:

1. No DFlash or DSpark recipe read in this review explains its layer list. The
   six-boundary control is a precedent from one public drafter.
2. ELSI and the similar selectors score how widely feature values spread. None
   of them uses the tokens that the drafter must predict.
3. Reported gains from layer selection are small. A comparison of feature sets
   needs the spread between repeated runs first.
4. The pinned sources show three facts that the current protocol does not test.
   The sections below call them phase, lookahead, and stream reduction.

The most useful addition is a statistic that uses the prediction target. It
measures how well each boundary predicts the next eight target distributions
after the drafter knows the anchor token.

## Terms in this note

This note uses the canonical terms in the [glossary](../../CONTEXT.md). It uses
`B_k`, `M_k`, and `R45` as the [approved baseline](first-drafter-recommendation.md)
defines them. The glossary does not define the terms below. They are local to
this note until the project adopts them.

| Term | Meaning in this note |
| --- | --- |
| Block | One decoder layer of the target model, numbered from zero. |
| Boundary | The hidden state after a completed block. `B_k` follows block `k-1`. |
| Capture point | The place in the target computation where a hidden state is recorded. |
| Anchor | The newest known token, at position `a`. |
| Row | One of the eight drafter inputs. Row `j` predicts token `a+j+1`. |
| Lookahead | The count of positions from a context feature to the token to predict. |
| Accepted length | Tokens gained per verification. The count includes the one token that the target model adds. |
| Probe | A small model trained to read one quantity from stored features. |
| Phase | The position of a block inside the repeating four-block attention pattern. |

## What the choice controls

A feature set matters only through the acceptance that a trained drafter
reaches with it. Three inspected facts define that link.

Context features come only from positions before the anchor. DeepSpec builds
the mask with `kv_idx < anchor_pos`, and the approved contract uses
`a-2048 <= t < a`. Row `j` predicts token `a+j+1`.
[DeepSpec mask][ds-mask] · [DeepSpec labels][ds-labels]

The drafter embeds only the anchor and the mask rows. Each earlier token
reaches the drafter only as context features. DeepSpec concatenates the
features of one position and applies one bias-free linear projection, then
RMSNorm. Every draft layer builds its keys and values from that one vector.
[DeepSpec projection][ds-fc] · [DeepSpec forward][ds-forward]

The DSpark paper gives the acceptance rule in section 3.2.1, equation 8. Under
sampling, verification accepts a proposal with chance
`1 - 0.5 * sum(abs(p - q))`, given that it accepted every earlier proposal.
The first rejection discards the later proposals. Call that chance `c_j` for
row `j`. Then:

```text
accepted length = 1 + c_0 + c_0*c_1 + ... + c_0*c_1*...*c_7
```

The public drafter card agrees with this form. For each dataset row, one plus
the sum of the eight position rates equals the reported accepted length. The
card reports those rates at temperature 0. [Public drafter card][public-card]

| Card dataset | Position 0 rate | Later rates, each divided by the rate before it |
| --- | --- | --- |
| `math_reasoning` | 92.9% | 86.4% to 91.5% |
| `qa` | 73.1% | 60.3% to 68.5% |

The ratios are arithmetic on the card numbers, not measurements from this
project. They show that a loss at an early row reduces every later term.

## Findings that shape the questions

### Published layer lists come without reasons

| Recipe | Target features | Stated rule or reason |
| --- | --- | --- |
| DFlash | Default count equals draft depth | Even spacing from block 1 to block `N-3`. The paper tests 3 against 5 features. It does not test which blocks. |
| DeepSpec Qwen3 | `[1,9,17,25,33]`, `[1,10,19,28,37]` | None stated. Both lists equal the DFlash rule output. |
| DSpark paper | Not named | "we use the same target-model feature layers for all drafters". No layer ablation. |
| DeepSeek native V4 | Last three blocks | None found in the [earlier review](deepseek-dspark-design-choices.md). |
| Speculators default | `[2, num_layers//2, num_layers-3]` | The documentation states the default and gives no reason. |
| Public GLM drafter | `[20,28,32,36,40,44]` | The card gives no reason. |

Sources: [DFlash paper][dflash] · [DFlash rule][dflash-code] ·
[DSpark paper][dspark] · [Speculators options][spec-cli] ·
[Public drafter card][public-card]

Neither documented rule produces the public GLM list. The DFlash rule gives
blocks `[1,9,17,26,34,42]` for 45 blocks and six features. The Speculators
default gives `[2,22,42]` for 45 blocks. Only the publisher knows the reason
for the list. No inspected source records it.

### The control reads one phase of the attention pattern

The pinned configuration repeats three linear-attention blocks and then one
sparse-attention block. The sparse-attention blocks are
`[3,7,11,15,19,23,27,31,35,39,43]`. [Target configuration][glm-config]

The control boundaries `[20,28,32,36,40,44]` follow blocks
`[19,27,31,35,39,43]`. All six are sparse-attention blocks. Every identifier
is a multiple of four, so the list keeps one phase under either reading of
the identifiers.

The control cannot separate the effect of depth from the effect of phase. No
inspected source says that the publisher chose this on purpose.

### The newest context feature is two tokens behind the first proposal

The newest context feature sits at position `a-1`. At that position, the last
block computed the distribution for token `a`. The drafter already holds token
`a` as the anchor. Row 0 predicts token `a+1`, so its lookahead is 2. Row 7
has a lookahead of 9.

Two probing studies report that depth acts differently at lookahead 2 and
beyond:

- Future Lens reads one hidden state of GPT-J. Its best method peaks at
  middle layers for lookahead 2. It reports that next-token accuracy "peaks at
  the last layer". [Future Lens][future-lens]
- A 2026 study trains one linear probe per layer and per lookahead on three
  models. Accuracy falls as lookahead grows. At lookahead 8 on general text,
  the probe overlaps a unigram baseline at every layer.
  [Lookahead probes][plan]
- The same study finds that a probe can read information that the model does
  not use. The authors write that "encoding and use turn out to be
  dissociable."

Both studies read one position with a small reader. The drafter reads up to
2,048 positions with five layers. Public recipes do not settle the question.
The native DeepSeek list reads only late blocks, and the DeepSpec lists read
every depth. These results are hypotheses for this target. They are not
findings about it.

### The checkpoint holds a multi-token prediction module

The pinned checkpoint index lists a block at index 45 with the tensors
`eh_proj`, `enorm`, `hnorm`, and `shared_head.norm`. The configuration sets
`num_nextn_predict_layers` to 1. The pinned Transformers implementation
excludes this module. [Checkpoint index][glm-index] · [Transformers note][tf-mtp]

DeepSeek-V3 describes a module with matching parts. Its first module combines
the main model's representation of token `i` with the embedding of token `i+1`,
and it predicts token `i+2`. [DeepSeek-V3 report, section 2.2][v3]

Row 0 has the same task: features up to `a-1`, plus token `a`, predict token
`a+1`. If this target follows that design, training shaped its last hidden
state for row 0. This is an inference from tensor names. No inspected source
shows what the module of this target reads.

The approved contract keeps `M45` as the teacher state and not as a context
feature. DeepSpec also rejects the last block as a feature, for a
capture-mismatch reason. Neither decision rests on a measurement of what `M45`
adds. [DeepSpec guard][ds-guard]

### The target does not read the unweighted mean between blocks

Each sublayer of the target reads a weighted sum of the four streams. The
input mapping computes those weights again for each token. Only the final
readout uses the unweighted mean. [Input mapping][glm-mhc] · [Final readout][glm-final]

For `k` below 45, `M_k` is a reduction that the target itself does not use.
The approved baseline already records that averaging discards stream
differences. The target computes each sublayer input vector during its normal
pass. That vector is a candidate context feature at no added target compute.

### One shared projection limits what more features add

DFlare repeats the DFlash design on Qwen3-4B with 5, 7, and 9 target layers.
Accepted length moves from 4.13 to 4.18 to 4.19. DFlare then gives each draft
layer its own weighted mixture and reports 4.38 to 4.49. [DFlare][dflare]

DFlash reports a gain from 3 to 5 features with a three-layer drafter. On
Math500, accepted length moves from 5.38 to 5.64. [DFlash paper][dflash]

Both results come from other targets. They suggest that a seventh feature adds
little under one shared projection. They also show that the best feature set
depends on the fusion design.

### Label-free selectors report small and mixed gains

ELSI averages hidden states over 32-token blocks from 200 ShareGPT samples. It
keeps the 30 percent of positions with the highest predictive entropy. It then
searches for the layer set whose concatenation has the highest matrix entropy.
ELSI-C adds a filter from CKA transitions. [ELSI paper][elsi]

| Selector | Statistic | Reported effect of the selection on accepted length |
| --- | --- | --- |
| ELSI | Matrix entropy of concatenated layers | Qwen3-4B: 1.6% to 2.2% above the fixed three-layer rule. Llama-3.1-8B: 1.0% below it. |
| HAWK | Linear CKA regions and coverage | 3.24 against 3.17 for an even grid, with ten layers against nine. |
| EDSD | Curvature and LiDAR per layer | Not extracted in this review. |

Sources: [ELSI paper][elsi] · [HAWK][hawk] · [EDSD][edsd]

The ELSI tests use EAGLE-style drafters with one to three features. The paper
includes one low-entropy control, which scores lowest. This review found no
test of rank agreement between the statistic and trained acceptance. No source
tests ELSI with a DSpark-style drafter, six or seven features, or an mHC
target.

HAWK reports a caution for every CKA statistic. Before it removed 32
high-variance dimensions, adjacent layers scored 1.000 across a long range of
one model.

## Question list

Each question names the statistic that answers it. The next section defines
the statistics. Notes on single questions follow the table.

| ID | Group | Question | Statistic |
| --- | --- | --- | --- |
| Q1 | Task | Which rows lose the most accepted length with the control? | S16, S17 |
| Q2 | Task | After the drafter knows the anchor token, how much does each boundary still tell about tokens `a+1` to `a+8`? | S8, S11 |
| Q3 | Task | How fast does the value of a context feature fall with its distance from the anchor? | S11 with shorter windows |
| Q4 | Task | Does the drafter need the exact identity of earlier tokens, and which boundaries keep it? | S7 |
| Q5 | Content | At which depth does the target settle its next-token choice? | S6 |
| Q6 | Content | Does this target show a middle-depth peak for lookahead 2 and beyond? | S8, S11 |
| Q7 | Content | Do boundaries after sparse-attention blocks differ from boundaries after linear-attention blocks? | S2 to S8, split by phase |
| Q8 | Content | Where do representations change sharply between neighboring blocks? | S3 |
| Q9 | Content | Does one boundary dominate the projection because its values are larger? | S4 |
| Q10 | Content | What does the final boundary add as a context feature? | S8, S11 |
| Q11 | Content | Which boundaries predict how uncertain the target is at later positions? | S10 |
| Q12 | Streams | How different are the four streams at each depth? | S9 |
| Q13 | Streams | How far is `M_k` from the vector that the next sublayer reads? | S9 |
| Q14 | Streams | Does a probe gain from four streams, or from the sublayer input, over the mean? | S8, S11 |
| Q15 | Sets | What does one more boundary add to a set that is already chosen? | S2, S8, S11 |
| Q16 | Sets | Do two neighboring boundaries beat two distant boundaries? | S8, S11 |
| Q17 | Sets | How many features can one 4,096-wide projection use? | S8 in reduced-rank form |
| Q18 | Sets | Which features does a trained drafter use? | S1, S14, S15 |
| Q19 | Data | Do rankings change between code, chat, and reasoning text, or between reasoning-effort settings? | Any, split by workload |
| Q20 | Data | Do the tokens that a statistic keeps match the tokens where the drafter loses acceptance? | S2, S16 |
| Q21 | Data | Do rankings change between temperature 0 and sampling? | S16, S17 |
| Q22 | Experiment | How much does accepted length move between repeated runs of one feature set? | S16, S17 |
| Q23 | Experiment | Does a low-cost statistic rank feature sets in the same order as trained drafters? | Rank agreement of S2 to S11 with S16 |
| Q24 | Experiment | Does a short training run favor boundaries that are easy to use early? | S16 learning curves |
| Q25 | Experiment | How much can any feature set add over the control? | S12 |
| Q26 | Experiment | Does a selector that works on a small target also work on this target? | Q23 on both targets |
| Q27 | Serving | Which capture points can the serving engine supply with values equal to the offline hooks? | Engine inspection |
| Q28 | Serving | What must the capture store so that probes and subsets stay possible? | Storage arithmetic |

Notes:

- Q4 matters because the recipes disagree. The DeepSpec lists include block 1.
  The public GLM list starts at `B20`. The native DeepSeek list reads only the
  last three blocks.
- Q9 follows from the approved contract. It concatenates raw stream means and
  normalizes only after the projection.
- Q11 serves the confidence head.
- Q16 asks whether the drafter gains more from what one block wrote than from
  two distant states.
- Q20 applies to ELSI, which keeps only high-entropy positions.
- Q22 comes before Q23. The reported gains from selection are small, so an
  unknown spread makes any ranking unreadable.
- Q23 needs a set that the statistic ranks low, a random set, and a
  phase-shifted copy of the control.
- Q28 has a known size. All 45 stream means cost 368,640 bytes for each token
  at two bytes for each element, which is 45 × 4,096 × 2. Four full streams
  cost four times that. Probes need features for each token, so pooled sums
  are not enough.

## Statistics catalog

The tiers are in order of cost. Tier 1 needs files only. Tier 2 needs pooled
activations. Tier 3 needs activations for each token with teacher states.
Tier 4 needs small training runs. Tier 5 needs full runs.

| ID | Tier | Statistic | Needs | Main limit |
| --- | --- | --- | --- | --- |
| S0 | 1 | Structure map: attention type and MLP type of each block, and the phase of each boundary | Pinned configuration | Describes structure, not content. Computed in this review. |
| S1 | 1 | Share of each feature in the context projection of a public drafter | Public weights. Feature scale needs activations. | Shows what one trained recipe uses. A weight norm without feature scale misleads. |
| S2 | 2 | Matrix entropy of each boundary and of concatenated sets, as ELSI defines it | Pooled means | Ignores the prediction target. Depends on scale and pooling. |
| S3 | 2 | CKA between boundaries and its transition scores | Pooled means | A few high-variance dimensions can dominate it. |
| S4 | 2 | Scale profile: root mean square and variance per dimension at each boundary | Pooled first and second moments | Describes a risk. It does not rank sets. |
| S5 | 2 | Curvature, LiDAR, DiME, and effective rank | Pooled means, or tokens in order for curvature | Layer by Layer tests them on embedding tasks. EDSD applies two of them to drafting. |
| S6 | 3 | Depth trajectory: decode each boundary and measure the distance to the teacher distribution at the same position | `M_k` and `M45` for each token | The plain decode assumes that `M_k` fits the final readout. A fitted map for each boundary removes that assumption. |
| S7 | 3 | Token identity probe: predict the token at position `t` from `M_k` at `t` | `M_k` and tokens | A large probe can memorize. It needs a control. |
| S8 | 3 | Linear lookahead regression, defined below | Candidate features and `M45` for each token | Linear only. It reads one position. |
| S9 | 3 | Stream statistics: share of variance between streams, input-mapping weights, and the distance from `M_k` to the sublayer input | Full streams for a small set | Describes loss of detail. It does not show that the drafter needs the detail. |
| S10 | 3 | Entropy probe: predict teacher entropy at later positions from each boundary | Candidate features and `M45` | Serves the confidence head only. |
| S11 | 4 | Drafter-shaped probe: a one-layer drafter with the approved inputs, trained on one boundary or one small set, and scored for each row | Feature cache for a small corpus | A shallow probe can rank sets differently from five layers. |
| S12 | 4 | Superset drafter: one drafter trained on a wide capture pool | Wide feature cache | Wide projection. Gives a ceiling, not a set. |
| S13 | 4 | Feature-dropout drafter: train on a superset and drop features at random, then score many subsets without new training | Wide feature cache | Shared weights rank subsets with error. |
| S14 | 4 | Learned scalar mixture over the capture pool for each draft layer, as DFlare defines it | Wide feature cache | Changes the approved fusion. DFlare does not report its learned weights. |
| S15 | 4 | Feature ablation on a trained drafter: replace one feature with zero or its mean | A trained drafter and a target runtime | Measures reliance of one drafter, not the value of the feature. |
| S16 | 5 | Matched training runs, with offline distance to the teacher for each row | Full feature cache for each candidate | Cost limits the number of sets and seeds. |
| S17 | 5 | Online acceptance for each row and workload, with repeated seeds | Serving with state recovery | Depends on the serving work in the target contract. |

S8 predicts the teacher state at position `t+d-1` from candidate features at
position `t`, for lookahead `d` from 2 to 9. It also receives the embedding of
token `t+1`, which stands for the anchor. The score is the distance between
the predicted and the true distribution through the frozen target readout.
The gain over the anchor embedding alone is the part that the features add.

A reduced-rank form of S8 answers Q17. It limits the regression to a fixed
number of directions. It then reports how the score falls as that number
falls.

S15 runs on the public GLM drafter without new training. It needs a target
runtime with matching capture.

Sources for S5: [Layer by Layer][layer-by-layer]. Sources for the tier 3
methods: [tuned lens][tuned-lens] · [usable information][v-info] ·
[probe controls][control-tasks] · [CKA reliability][cka-reliability] ·
[feature selection by information][brown].

## Knowledge that is out of reach today

| Knowledge | Why it helps | Why it is out of reach | Nearest stand-in |
| --- | --- | --- | --- |
| Accepted length of every six-boundary set after full training | Gives the true ranking and the share of each boundary | 45 boundaries give 8,145,060 sets of six | S13, then S16 on a shortlist |
| Exact information that a feature set holds about the next eight target distributions, given the anchor | Removes the dependence on any probe | Estimates of this quantity have [formal limits][mi-limits] | S8 and S11 measure the usable part |
| How the publishers chose `[20,28,32,36,40,44]` and the last three blocks | Shows which alternatives they already rejected | Unpublished | Ask the publisher. Run S1 and S15 on their weights. |
| Which information about later tokens the target uses at each depth | Separates used information from readable information | Needs activation patching inside this target | S8, with the caution from [the probing study][plan] |
| What the multi-token prediction module of this target reads, and how its training used it | Tells whether training shaped `M45` for lookahead 2 | The pinned Transformers source excludes the module | Inspect a serving engine that runs native MTP |
| The ceiling for a drafter with every stream of every boundary | Bounds what layer choice can gain | Storage and projection width | S12 on a reduced pool |
| The feature-set ranking after training to convergence | Removes the bias of short runs | Compute | Learning curves in S16 |

## What each answer changes

| If the evidence shows | Then |
| --- | --- |
| Lookahead scores peak at middle depth after the anchor is given | Add boundaries below `B20` to the candidates. The control has none. |
| Late boundaries lead at every lookahead | Keep a late set. Test neighboring late boundaries and the final boundary. |
| Phase changes the scores at equal depth | State the phase of every candidate. Add a phase-shifted control. |
| The streams are close, and `M_k` is close to the sublayer input | Keep the unweighted mean and close Q12 to Q14. |
| The streams differ, and probes gain from them | Add the sublayer input or the full streams to the capture pool. |
| No late boundary keeps token identity, and copied spans fail | Add one early boundary. |
| The spread between seeds exceeds the gap between sets | Stop ranking feature sets at this scale. Vary draft depth, fusion, or data. |
| A low-cost statistic disagrees with trained ranking | Use that statistic only to describe features, not to select them. |

## Verification boundary

This review inspected the pinned target configuration, checkpoint index, and
Transformers model source. It inspected DeepSpec at the pinned revision after
initializing the submodule. It ran arithmetic only. That arithmetic covers
five items:

- The block types of the control.
- The outputs of the two layer rules.
- The ratios from the public card.
- The count of sets.
- The storage sizes.

This review loaded no weights. It ran no capture, probe, training, or serving.
Every statistic in the catalog is a proposal.

A fetch helper returned the text of the web sources. It returned no values
from figures. This review read the DSpark paper through the start of section
5.4. It read the ELSI and EDSD papers from extracted PDF text, and the EDSD
body only in part. It read the public card and the DFlash code from their main
branches on the review date, not from pinned revisions.

This review did not read six cited method sources again. They are the tuned
lens, usable information, probe controls, CKA reliability, feature selection,
and mutual information papers.

This session had no access to the corrected ELSI record in the old project or
to the uncommitted ELSI review. This note does not repeat or test their
results.

## Sources

[elsi-decision]: https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/9
[ds-mask]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py#L78-L107
[ds-labels]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py#L429-L445
[ds-fc]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py#L240-L245
[ds-forward]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py#L361-L385
[ds-guard]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py#L100-L112
[glm-config]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json#L79-L183
[glm-index]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/model.safetensors.index.json
[glm-mhc]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L300-L329
[glm-final]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L332-L336
[tf-mtp]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/docs/source/en/model_doc/glm5_next.md?plain=1#L26
[public-card]: https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview
[spec-cli]: https://docs.vllm.ai/projects/speculators/en/latest/cli/train/
[dspark]: https://arxiv.org/html/2607.05147v1
[dflash]: https://arxiv.org/html/2602.06036
[dflash-code]: https://github.com/z-lab/dflash/blob/main/dflash/model.py
[dflare]: https://arxiv.org/html/2606.02091v1
[elsi]: https://arxiv.org/abs/2610.12939v1
[hawk]: https://arxiv.org/html/2610.00623v1
[edsd]: https://aclanthology.org/2026.acl-long.2145/
[future-lens]: https://arxiv.org/html/2311.04897
[plan]: https://arxiv.org/html/2605.07984
[layer-by-layer]: https://arxiv.org/html/2502.02013
[v3]: https://arxiv.org/html/2412.19437
[tuned-lens]: https://arxiv.org/abs/2303.08112
[v-info]: https://arxiv.org/abs/2002.10689
[control-tasks]: https://arxiv.org/abs/1909.03368
[cka-reliability]: https://arxiv.org/abs/2210.16156
[brown]: https://jmlr.org/papers/v13/brown12a.html
[mi-limits]: https://arxiv.org/abs/1811.04251
