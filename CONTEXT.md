# GLM-5.3-Flash drafter research

Shared terms for understanding, training, and evaluating a speculative drafter
with a frozen target model.

## Language

### Numerical representations and operations

**Vector**:
An ordered list of numbers representing a model quantity. [Source][linear-algebra]

**Matrix**:
A rectangular arrangement of numbers in rows and columns. [Source][linear-algebra]

**Element**:
One number at a specified position in a vector or matrix. [Source][linear-algebra]

**Dot product**:
The sum of products of matching elements of two equal-length real vectors.
[Source][attention]

**Matrix multiplication**:
A matrix product formed from dot products of left-matrix rows and right-matrix
columns. The left matrix's column count must equal the right matrix's row count.
[Source][linear-algebra]

**Transpose**:
The matrix obtained by exchanging the rows and columns of another matrix.
[Source][linear-algebra]

**Identity matrix**:
A square matrix with ones on its diagonal and zeros elsewhere.
Multiplication by it leaves a compatible vector or matrix unchanged.
[Source][identity-matrix]

**Mean**:
The sum of values divided by their count.
For vectors, the mean is computed separately for each element position.
[Source][mean]

**RMSNorm**:
Root mean square layer normalization, which scales a representation using its
root mean square with numerical stabilization.
A learned elementwise scale can follow that normalization. [Source][rmsnorm]

**Epsilon**:
A small positive constant used to stabilize a normalization denominator.
[Source][rmsnorm]

**Softmax**:
A function that exponentiates each input and divides each result by the sum of
those exponentials. [Source][softmax]

### Models and token representations

**Target model**:
The frozen model whose next-token distribution the drafter learns to approximate.
The target model also verifies proposed tokens during speculative decoding.

**Drafter**:
A trained model that proposes candidate tokens for the target model to verify.

**Token**:
A vocabulary unit represented by an identifier, such as a word or part of a word.
[Source][tokens]
_Avoid_: Word when a token is not a whole word.

**Input sequence**:
The ordered tokens supplied to a model computation. [Source][transformer]

**Token embedding**:
The vector retrieved from an embedding table using a token's vocabulary
identifier. [Source][embedding]

**Hidden state**:
The internal representation of a token at a particular point in a model's computation.
With mHC, that representation can contain several residual-stream vectors.
[Source][glm-hc]

**Input hidden state**:
The hidden state supplied to a specified layer or sublayer update.
[Source][mhc-update]

**Output hidden state**:
The hidden state produced by a specified layer or sublayer update.
In an mHC attention update, it combines the residual contribution and attention
contribution. [Source][glm-attention]
_Avoid_: Attention output.

**Residual stream**:
A token representation carried between model layers and updated along the way.
[Source][glm-hc]

**Manifold-Constrained Hyper-Connections (mHC)**:
A residual-connection design that carries multiple streams and constrains how
their existing contents are mixed between layers.
[Source][mhc-constraints]

**Mixture of experts (MoE)**:
A network design that routes tokens to selected expert networks for processing.

### Attention and mHC mappings

**Query vector**:
A position's vector used with key vectors to compute attention scores.
[Source][attention]

**Key vector**:
A position's vector used with query vectors to compute attention scores.
[Source][attention]

**Value vector**:
The vector of information from an attended position that attention weights
combine. [Source][attention]

**Query projection**:
The linear transformation that produces query vectors from input
representations. [Source][attention]

**Key projection**:
The linear transformation that produces key vectors from input
representations. [Source][attention]

**Value projection**:
The linear transformation that produces value vectors from input
representations. [Source][attention]

**Key matrix**:
For one attention head, the key vectors arranged as rows in sequence order.
[Source][attention]

**Value matrix**:
For one attention head, the value vectors arranged as rows in sequence order.
[Source][attention]

**Attention scores**:
In scaled dot-product attention, the query-key dot products after scaling and
before softmax. Scaling divides by the square root of the query vector's element
count. [Source][attention]
_Avoid_: Attention weights.

**Attention weights**:
The softmax-normalized attention scores used to combine value vectors in scaled
dot-product attention. [Source][attention]
_Avoid_: Attention scores, next-token probabilities.

**Attention output projection**:
The linear transformation applied to attention's weighted-value results,
combining heads when attention has multiple heads. [Source][attention]
_Avoid_: Output mapping.

**Attention output**:
The attention sublayer's result after the attention output projection.
[Source][attention]
_Avoid_: Output hidden state, attention contribution.

**Attention sublayer**:
The attention computation within a Transformer layer, distinct from its
surrounding residual update. [Source][transformer]

**Input mapping**:
The mHC mapping that combines residual streams into the input vector for a
sublayer. [Source][mhc-update]

**Residual mapping**:
The mHC mapping that mixes input residual streams along the residual path.
Its matrix targets the doubly stochastic constraint. [Source][mhc-constraints]

**Output mapping**:
The mHC mapping that distributes a sublayer's output across residual streams,
with a separate scale for each stream. [Source][mhc-update]
_Avoid_: Attention output projection.

**Doubly stochastic matrix**:
A square matrix of nonnegative elements whose rows and columns each sum to one.
[Source][mhc-constraints]

**Attention path**:
In an mHC attention update, the path through the input mapping, normalization,
attention sublayer, and output mapping. [Source][glm-attention]

**Residual path**:
In an mHC update, the path from the input hidden state through the residual
mapping to the residual contribution. [Source][glm-attention]

**Residual contribution**:
The contribution from the input hidden state after mHC's residual mapping mixes
its residual streams. [Source][mhc-update]

**Attention contribution**:
The contribution to the output hidden state from mHC's output mapping applied to
the attention output. [Source][glm-attention]
_Avoid_: Attention output.

### Drafter features and evaluation

**Context feature**:
A target-model representation supplied as an input to the drafter.

**Capture pool**:
The collection of target representations recorded as candidates for feature selection.

**Feature set**:
The ordered selection of context features supplied to a drafter.

**Teacher state**:
The target representation immediately before its output head, used to compute
teacher probabilities for training.

**Feature cache**:
Stored training examples with tokens, supervision masks, and target representations.
_Avoid_: Unqualified cache when inference state is also discussed.

**Inference state**:
The accumulated model state for an already-processed token prefix.
_Avoid_: Feature cache.

**Verification**:
The target-model step that determines which proposed tokens can be accepted.

**Training checkpoint**:
Saved drafter parameters and configuration from a training run.
An exact-resume checkpoint also includes the state needed to continue that run.

[linear-algebra]: https://dlmf.nist.gov/1.2#v
[identity-matrix]: https://docs.pytorch.org/docs/2.14/generated/torch.eye.html
[mean]: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
[rmsnorm]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html
[softmax]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Softmax.html
[tokens]: https://huggingface.co/docs/transformers/tokenizer_summary
[transformer]: https://arxiv.org/html/1706.03762v7#S3
[embedding]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html
[attention]: https://arxiv.org/html/1706.03762v7#S3.SS2
[mhc-update]: https://arxiv.org/html/2512.24880v2#S1.E3
[mhc-constraints]: https://arxiv.org/html/2512.24880v2#S4.SS1
[glm-hc]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L218-L329
[glm-attention]: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py#L1313-L1338
