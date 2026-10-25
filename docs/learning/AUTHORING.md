# Learning walkthrough authoring rules

Use these rules for lesson prose, numerical examples, and animations.
They record the requirements and lessons learned while building the mHC
walkthrough. Read the affected lesson or guide's notes before revising it.
Use the user's latest feedback to choose the next change.

For new material, use the [document roles and layout](#document-roles-and-layout)
to choose where its files belong.

## 1. Define the purpose and endpoint

Read the [mission](MISSION.md) for the learning area's goals and constraints.
State what the learner will understand and why that knowledge matters.
Introduce the concrete input and the intended result before explaining the
operations between them. Connect the worked example to the larger task it serves.

For an existing lesson, preserve the parts the user already approved.
Target the remaining learning gap. Record approval of presentation, understanding
of mechanics, and understanding of purpose as separate observations.

Complete this step when the starting point, endpoint, and learning goal are
explicit. A list of operations alone does not establish the purpose.

## 2. Follow one concrete example throughout

Build the whole lesson around one example:

- Introduce all required inputs before using them.
- Keep the same inputs, names, and example through the final result.
- Carry each computed result into the operation that uses it.
- Identify saved quantities when returning to an earlier calculation path.
- Use the same example for explanations and comparisons whenever possible.
- Finish at the declared endpoint before introducing another topic.

Choose a small example that exposes the structure being taught. When shapes
matter, show actual vectors and matrices. Label what rows and columns represent.
Make changes in those meanings visible, including a transpose.

Choose numbers that the reader can distinguish. Inspect the intermediate results
before settling on the inputs. Convenient arithmetic can hide the distinction
the lesson needs to teach. Preserve identical copies when the algorithm requires
them, even when other inputs use distinct numbers.

Label chosen teaching numbers as illustrative. Keep them distinct from measured
model outputs, learned parameters, and approved experiment settings.

Complete this step when one runnable example reaches the declared result and
accounts for every numerical quantity needed along the way.

## 3. Fix the terminology, then write plain English

Read the [root glossary](../../CONTEXT.md) before writing a lesson or guide.
It defines the project's shared technical vocabulary for lessons, guides, and
code examples. Follow the [domain documentation rules](../agents/domain.md)
when changing it. Use one term for each concept.

Use this process for shared terms:

1. Find each concept in `CONTEXT.md`. Reuse its established name and meaning.
2. For a missing shared concept, confirm its meaning in primary sources.
   Use the [resources](RESOURCES.md) as a starting point.
   Add a concise definition and source link to `CONTEXT.md` before using the term.
3. If existing definitions conflict, resolve the intended meaning before writing
   the explanation. Use explicit qualifiers for different concepts.
4. When a shared name or meaning changes, find affected first-party uses.
   Update their explanations, captions, and named code outputs in the same change.

Apply the same names to all learner-facing material:

- Headings and prose.
- Captions, diagrams, and axis labels.
- Written equations and mathematical symbols.
- Runnable examples and their named outputs.

Define each unfamiliar term when it first appears. Use ordinary language to
explain it, while preserving the shared name and meaning. Different lessons can
use different examples or analogies. Link local glossary summaries to `CONTEXT.md`.
Distinguish related concepts with explicit qualifiers, such as attention scores
and attention weights.
Define local mathematical symbols once and keep their meanings stable.
Keep derivations, toy values, and example assumptions with their lesson or guide.

Use `/simple-english` in Plain mode for prose and captions. Preserve the glossary
terms while simplifying the surrounding sentences:

- Use short sentences, active voice, and simple tenses.
- Put the condition before the instruction.
- Give one instruction per procedural sentence.
- Keep procedural sentences within 20 words and descriptive sentences within 25.
- Keep each paragraph on one topic.
- Remove filler, vague claims, and repeated explanations of the current task.

During a prose-only edit, preserve code, commands, paths, quoted text, numbers,
and factual meaning. Change them only when the task requires it.

Complete this step when shared concepts have entries in `CONTEXT.md`.
Names and meanings must match across affected lessons, guides, animations, and
runnable examples. Plain language must clarify the terms rather than replace
them with informal alternatives.

## 4. Explain why before showing how

Give every numbered step a visible purpose. Explain the need it serves, how its
operation serves that need, and why the following step needs its result.
When useful, explain what is lost if the operation is omitted.

Keep these distinctions explicit:

- The purpose of the operation versus the mechanics of its arithmetic.
- The real model's learned behavior versus the toy's fixed assumptions.
- The role of a parameter versus the role of the data passing through it.
- What the example demonstrates versus what requires evidence from trained models.

Support architectural and mathematical claims with primary sources.
Explain chosen numerical settings as teaching choices. Assign semantic meanings
to elements or fixed jobs to internal components only when evidence supports them.

Put the purpose at the relevant step in both the written lesson and the video.
A short purpose caption can precede the operation. Put longer explanations beside
the written step, and introduce them in the same order as the computation.

Complete this step when every operation has a purpose explanation that does more
than restate the operation. The learner must be able to find each explanation
without searching a distant glossary or conclusion.

## 5. Give each visual moment one focus

For a video or animation, plan a sequence of small visual moments:

- Show one active operation at a time.
- Keep one compact cue for the example's identity.
- Use large, readable numbers and enough empty space around them.
- Highlight the active elements, row, or column and dim unrelated content.
- Show one short explanation at a time.
- Remove completed arithmetic when its inputs leave the scene.
- Replace completed work with its result instead of accumulating panels.
- End each step on a stable result that the reader can inspect while paused.

Show an overview when it helps orientation. Reveal its detail when needed rather
than keeping a full path map beside every calculation. Separate distinct
operations into different steps when combining them increases visual load.

Preserve identity through motion. Move copies of the same numbers between layouts.
Reveal changed numbers cleanly so their digits stay readable. For a complex
rearrangement, move a small group at a time. The mHC transpose improved when only
one element pair moved at once.

Give each color one consistent role. The approved reference uses gold for the
current focus and teal for a computed result. Keep labels readable without
requiring the reader to identify objects by color alone.

A silent video still needs explanatory text. Time each caption with its operation
and leave enough time to read it. Use the written lesson for full derivations
rather than placing every equation on screen simultaneously.

Complete this step when the active operation is evident from the rendered frame
without reading several competing text blocks.

## 6. Keep the numbers and formats connected

Use one authoritative numerical computation. Reuse its outputs in the renderer
and compare the written tables against it. When inputs change, recompute every
affected table, equation, caption, and numerical expectation.

Keep unrounded numbers for arithmetic. Round only for display. Use approximation
signs where appropriate, and explain any effect of display rounding.
Omit unnecessary trailing zeros when they make the animation harder to read.

Keep the written and animated versions synchronized:

- Use the same example, names, numerical results, and step order.
- Keep chapter start times, pause behavior, and timing descriptions consistent.
- Allow enough time for explanations rather than treating a previous duration as fixed.
- Re-render assets when scene data, text, or layout changes.
- Keep the written explanation usable without video playback or JavaScript.

Provide pause, seeking, speed controls, and keyboard access for video lessons.
Keep sources and render instructions with the lesson.
Use [Manim Community](https://github.com/ManimCommunity/manim) for new animations
and future scene revisions. Follow its [official documentation](https://docs.manim.community/en/stable/)
and record the tested version with the render command.

Before revising a ManimGL scene, convert it to Manim Community.
Compare its numbers, timing, and rendered presentation with the approved version.
Keep existing render instructions accurate for the current scene source.

Complete this step when the text, runnable example, and rendered assets describe
the same computation and their navigation matches the actual media.

## 7. Inspect the result and record what was learned

Review the artifact that the learner receives:

- Run the numerical checks and relevant repository checks.
- Compare names and definitions against `CONTEXT.md` across prose, captions, and
  named code outputs.
- Inspect actual rendered frames, including intermediate calculations and transitions.
- Inspect moving numbers for overlap, distortion, or lost identity.
- Test chapter pauses, playback controls, and keyboard behavior.
- Inspect the page at normal player size and on a narrow screen.
- Make sure that every step explains its purpose as well as its operation.

Source inspection and passing tests do not establish visual clarity.
Use rendered evidence to find presentation failures. Use learner feedback to
establish understanding, and record the specific parts they approve or still
find unclear.

When studying a reference, read its actual teaching material and inspect its
visuals or animation sequences. Cite what you used and report access gaps.
Distinguish sampled frames from complete video viewing.

For requested code cleanup, follow the actual
[Cursor deslop instructions](https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/deslop/SKILL.md).
Keep factual, numerical, and terminology reviews as explicit checks alongside
code cleanup.

Complete this step when the relevant checks pass and the rendered artifact has
been inspected. Record remaining learning gaps in the repository so the next
session can target them without rebuilding approved work.

## Document roles and layout

The repository-level [glossary](../../CONTEXT.md) owns shared terms across lessons
and guides.

The learning root holds shared context:

- `MISSION.md`: Goals and constraints for the whole learning area.
- `RESOURCES.md`: A source index that lessons and guides can reuse.
- `AUTHORING.md`: Rules for writing, explaining, organizing, and verifying material.

These documents can focus on this project while serving more than one lesson.
Keep feedback about a particular lesson in that lesson's notes.

A lesson teaches understanding through an ordered progression and a worked
example. A guide answers a specific question or helps complete a task.
Readers can open a guide at the needed section without following a full lesson.
Guides reuse the terminology, evidence, readability, and verification rules.
Give each worked explanation in a guide its own coherent example.

`animations/` contains editable scene code and render instructions.
`assets/` contains files the page displays or loads, such as videos, images,
and stylesheets. Rendered animation output belongs in `assets/`.
Both lessons and guides can use these directories.

For new material, use one directory per lesson or guide:

```text
CONTEXT.md
docs/learning/
  MISSION.md
  RESOURCES.md
  AUTHORING.md
  assets/
    lesson.css
  lessons/
    0001-topic/
      index.html
      NOTES.md
      example.py
      animations/
        scene.py
        README.md
      assets/
        walkthrough.mp4
        poster.jpg
  guides/
    topic/
      index.html
```

Add notes, runnable examples, assets, and animation sources only when needed.
A guide can use the same optional files and directories shown for a lesson.
Keep its examples, explanations, and media together under the topic directory.
Link to primary sources beside the claims they support, even when the shared
resource index lists those sources.

Keep files used by several lessons or guides in shared directories at the learning root.
The existing `assets/lesson.css` serves both the lesson and the configuration
guide. A video or scene used by only one lesson belongs inside that lesson.
Extract shared animation code when several lessons or guides need it.

For existing material, adopt this layout in a separate file-move change.
Update incoming links, imports, render commands, and path-based tests together.
Keep file moves separate from changes to the lesson's behavior or content.

## Reference artifacts

The [mHC lesson](lessons/0001-follow-one-token-through-mhc.html) contains the
project's written walkthrough. The [teaching notes](NOTES.md) record
lesson-specific choices and user feedback.

The current lesson's dimensions, exact numbers, timing, and typography are
example choices. Use the rules above to choose appropriate settings for a new
lesson rather than treating those choices as universal requirements.
