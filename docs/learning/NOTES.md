# Teaching notes

## 2026-10-24

- The user explicitly requested more explanation, code examples, and better visuals.
- Four mHC streams and the nonnegative/unit-sum matrix constraint remain unclear to them.
- They know queries, keys, and values approximately, but not the head-count distinction.
- Clarify gated **SiLU** versus **ELSI-C**, and **position weighting** versus numerical precision.
- Keep each explanation grounded in a concrete numerical example before adding terminology.
- Explanations and retrieval questions have been provided; do not record mastery without feedback.
- The scalar toy, matrix invariants, RoPE frequencies, shared-KV example, and loss arithmetic
  were executed successfully. These were not model runs.
- The corrected Excalidraw matrix checkpoint is `fe86f8a4f24c44c8a1`;
  the four-stream update diagram checkpoint is `0627795f5de8494888`.

## Baseline approval later in the same session

The user approved the complete restated first-run baseline: "This sounds good to me."
The selected contract is in `../research/first-drafter-recommendation.md`.
This approval does not establish mastery of every concept or validate model behavior.

## Session close-out

The configuration reference now lives at `guides/drafter-config.html` rather than
`reference/drafter-config.html`, keeping it inside first-party search and checks.
The lesson and repository README link to the current location.
