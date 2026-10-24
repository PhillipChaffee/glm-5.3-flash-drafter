# GLM-5.3-Flash drafter research

Terms for training and evaluating a speculative drafter with a frozen target model.

## Language

**Target model**:
The frozen model whose next-token distribution the drafter learns to approximate.
The target model also verifies proposed tokens during speculative decoding.

**Drafter**:
A trained model that proposes candidate tokens for the target model to verify.

**Hidden state**:
The internal representation of a token at a particular point in a model's computation.

**Residual stream**:
A token representation carried between model layers and updated along the way.

**Manifold-Constrained Hyper-Connections (mHC)**:
A residual-connection design that carries multiple streams and constrains how
their existing contents are mixed between layers.

**Mixture of experts (MoE)**:
A network design that routes tokens to selected expert networks for processing.

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
