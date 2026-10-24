# Mission: understand and design our GLM Flash drafter

## Why

Build and train our own GLM-5.3-Flash drafter for learning and enjoyment.
Understanding its representations and configuration should let us choose
controlled experiments and interpret their results.

## Success looks like

- Trace one token through mHC and explain how identical streams can diverge.
- Distinguish target features, draft state, attention heads, and position signals.
- Explain what a training loss measures and what an experiment can establish.

## Constraints

- Ground examples in the pinned GLM-5.3-Flash implementation.
- Use small numerical examples, runnable code, and visual explanations.
- Record design choices explicitly; explanations alone do not imply approval.
- This teaching workspace is under `docs/learning`; upstream `reference/` files
  remain pinned source evidence.
- Reference guides use `guides/` so first-party checks do not exclude them as
  upstream snapshots under a directory named `reference`.

## Out of scope

- Production guarantees or a requirement to beat public drafters, as set by the map.
