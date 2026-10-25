# Teaching notes

Use the [authoring rules](AUTHORING.md) when creating or revising a lesson.
The notes below record the choices and feedback for this lesson.

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

## Lesson terminology convention

Use one accepted term for each concept across the HTML lesson, animation, and
Python example. Define the term in plain English, then keep its name unchanged.
The lesson's [terminology glossary](lessons/0001-follow-one-token-through-mhc.html#terms)
records the names and cites the mHC paper, Transformer paper, and PyTorch docs.

Use input mapping, output mapping, and residual mapping for the three mHC
mappings. Distinguish attention scores, attention weights, attention output,
attention contribution, residual contribution, and output hidden state.
Keep following `sat` in the same `The cat sat` input sequence throughout.

Use `/simple-english` in Plain mode for lesson prose and video captions.
Preserve the glossary terms when simplifying sentences.

## Approved token embeddings

Use `[1, 2, 3]` for `The`, `[-4, 5, 6]` for `cat`, and `[7, -8, 9]` for `sat`.
The distinct numbers make the token embeddings easier to track.
The user chose these signed token embeddings after comparing the alternatives.

## Visual focus

The user found the persistent path map and simultaneous calculations too busy.
Show one operation at a time. Move its result into the next operation.
Remove completed arithmetic instead of keeping it beside later results.
Use one short caption and large numbers. Keep the input sequence visible.
Gold marks the current focus. Teal marks a computed result.

## Purpose revision brief

The user approved the visual presentation. They said that the lesson explains
how the computation works well. Their remaining concern was that it did not
explain why the steps are needed.

The requested revision must explain the purpose of every step in both the written
lesson and the video. Keep the approved visual focus and the terminology
convention above. First establish why a model needs a hidden state that depends
on the input sequence. Connect this to the model's goal of predicting the next
token. Keep the toy's numerical endpoint at the output hidden state `X1`.

For each step, explain the need it serves before showing the operation.
Connect its result to the next step. A useful explanation states what the
operation enables, rather than restating its arithmetic.

These questions guide the writing. Answer them using the primary sources
already linked from the lesson.

| Step | Purpose question to answer |
| --- | --- |
| 1 | Why represent tokens with token embeddings instead of using token labels directly? |
| 2 | What do four residual streams enable? Why initialize all four from the same token embedding? |
| 3 | Why combine four residual streams into one vector before attention? |
| 4 | What problem does RMSNorm address before the projections and attention scores? |
| 5 | Why use a value projection to form the information that attention combines? |
| 6 | Why compare query vectors and key vectors? Why divide the dot product by the square root of the query vector’s element count? |
| 7 | Why convert attention scores into positive attention weights that sum to one? |
| 8 | Why combine the value vectors with those attention weights? What can the attention output add for sat? |
| 9 | Why retain a residual contribution? Why mix residual streams, and why constrain the residual mapping? |
| 10 | Why apply different amounts of one attention output to the residual streams? |
| 11 | Why add the residual contribution and attention contribution? What is the output hidden state for? |

Keep the purpose visible when the learner reaches each step. Use short video
captions before the current motion, and put longer explanations beside the
corresponding written step. Allow enough reading time without adding simultaneous
text blocks.

Follow the toy-versus-GLM distinctions in `animations/README.md`.
Describe the purpose of an operation separately from the chosen example numbers.
Explain attention weights as weights on the input sequence. Distinguish them
from the model's later next-token prediction.
Do not assign invented semantic meanings to the token-embedding elements or
fixed jobs to the residual streams. Add the explanations at the relevant steps.
The approved example and visual presentation remain the baseline.

## Purpose revision completed

The HTML lesson and video now explain why all eleven steps exist.
The introduction connects a hidden state that depends on the input sequence to
next-token prediction. The numerical example still ends at `X1`.
The written explanations link the primary sources beside the claims.

The video gives each purpose caption six seconds of reading time.
Captions replace each other in the existing caption area.
Each chapter now lasts 40 seconds, for a total of 7 minutes 20 seconds.
The HTML chapter starts, pause duration, render instructions, and chapter test
use the new timing.

Verification results:

- All 17 Python tests and all 21 repository gates passed.
- All 18 displayed matrices matched the numerical trace.
- The numerical computation and stylesheet matched their pre-edit file hashes.
- All 20 glossary headings passed the terminology check.
- The video contains 13,200 H.264 frames at 1920 × 1080 and 30 frames per second.
- The full video decoded without errors.
- Visual review covered 36 purpose-caption frames and 24 calculation or transition frames.
- All eleven chapter-end numerical displays matched the approved version in side-by-side review.
- All eleven chapter buttons paused at their new endpoints.
- Playback speeds, keyboard controls, and full-video playback across chapter boundaries passed.
- Desktop and 320/390-pixel layouts passed, including the expanded glossary.
- The written lesson worked with JavaScript disabled, with no browser page errors during the checks.

Visual review used sampled frames, not continuous viewing of every second.
The two video encodes have small pixel differences from compression.
The prior approval covers the earlier presentation, not this revision’s explanations.
The user’s understanding of each step’s purpose still needs feedback.

## Epsilon clarification

The user found “keeps the denominator nonzero” confusing.
They read it as the divisor used to average the squared elements.
Name RMSNorm’s final divisor explicitly, and state that epsilon enters before
the square root. The averaging divisor stays equal to the element count.

## Renderer choice

The user requested [Manim Community](https://github.com/ManimCommunity/manim)
for future animation work. The current mHC scene and video use ManimGL 1.7.2.
The user requested publication with this renderer and deferred conversion.
Before its next animation revision, convert the scene to Manim Community.
Preserve the approved numerical example and presentation during that conversion.
