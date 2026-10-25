# Render the mHC walkthrough

The [HTML lesson](../lessons/0001-follow-one-token-through-mhc.html) plays a local
MP4. It needs no Python installation or network access to play.

`mhc_toy.py` computes the numbers with the Python standard library.
`mhc_walkthrough.py` imports those numbers and animates them with
[3Blue1Brown's ManimGL](https://github.com/3b1b/manim), version 1.7.2.
The renderer is separate from the training environment.

Use [Manim Community](https://github.com/ManimCommunity/manim) for future animation work.
Before revising this scene, convert it to Manim Community.
Compare the rendered result with the approved presentation.
The command below reproduces the current ManimGL video.

## Read the numerical trace

From the repository root, run:

```sh
python3.12 docs/learning/animations/mhc_toy.py
```

The example follows `sat` in the input sequence `The cat sat` through an attention
sublayer with mHC. Its output hidden state is `X1`. The input sequence stays visible
in every video chapter.

The example has four residual streams and three elements per vector.
It uses one attention head for `sat`. The query, key, and attention output
projections are identity matrices. The value projection is fixed.
RMSNorm uses scale `1` and epsilon `1e-6`.

The input mapping, output mapping, and residual mapping are fixed.
Every displayed matrix comes from this computation.

GLM computes its mappings from the input hidden state. This untrained example
follows the attention path and residual path from the same input hidden state.
The output hidden state is the sum of the residual contribution and attention
contribution. The video and text use the same eleven-step order.

The video shows one operation at a time. It highlights the active elements,
then moves the result into the next operation. Gold marks the current focus.
Teal marks a computed result. Token names and residual stream indices label
the matrix rows. The full arithmetic remains in the written lesson.

Each step first explains its purpose. The captions connect the operations to
next-token prediction, while the numerical example stops at `X1`.
Purpose captions appear one at a time in the existing caption area.
Each stays visible for six seconds before the next caption or operation.
The [written steps](../lessons/0001-follow-one-token-through-mhc.html#example)
give the longer explanations and link their primary sources.

The [visual-design study](../../research/3blue1brown-visual-teaching.md) records
the reference lessons and animation techniques used for this revision.

Use the lesson's [terminology glossary](../lessons/0001-follow-one-token-through-mhc.html#terms)
for labels, captions, and Python identifiers. Each concept keeps one term.

## Render the animation

Install [FFmpeg](https://ffmpeg.org/download.html) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) first.
ManimGL also needs a working OpenGL context. This scene uses Pango text and drawn
brackets, so it does not need LaTeX.

From the repository root, run:

```sh
PYTHONPATH=docs/learning/animations uv run --no-project --python 3.12 \
  --with manimgl==1.7.2 --with setuptools==80.9.0 \
  --with numpy==2.5.4 --with manimpango==0.7.1 \
  manimgl docs/learning/animations/mhc_walkthrough.py MHCTokenWalkthrough \
  -w --hd --prerun -q -c '#101b2a' \
  --video_dir docs/learning/assets --file_name mhc-walkthrough
```

ManimGL 1.7.2 imports `pkg_resources`. The pinned setuptools version supplies it.
The scene uses Arial and Menlo. Both fonts are available on the macOS host used
for this render. If you use another host, install these fonts before rendering.
This render uses Python 3.12.13 and FFmpeg 9.0.1 on Apple Silicon.

Each of the eleven chapters lasts 40 seconds, for a total of 7 minutes 20 seconds.
The render checks the frame bounds at each caption and chapter ending.
It fails if content extends past the frame or a chapter exceeds 40 seconds.
If you change the timing, update the HTML chapter controls and pause duration too.

Make the poster from the first chapter:

```sh
ffmpeg -y -ss 2 -i docs/learning/assets/mhc-walkthrough.mp4 \
  -frames:v 1 -q:v 2 docs/learning/assets/mhc-walkthrough-poster.jpg
```

## Run the tests

From the repository root, run the test suite:

```sh
pytest
```

The tests follow the token embedding through to the output hidden state.
They compare the computed matrices with expected numbers and test the residual
mapping properties. Every matrix in the numerical trace must appear in the HTML
with matching numbers. Rendering uses the same numerical trace.

Open the HTML file locally. Play each chapter. Make sure that it pauses at its end.
Make sure that the speed selector and full-video button work.
Make sure that the page fits a mobile screen.
The written walkthrough also works with JavaScript disabled.
Inspect purpose captions, intermediate calculations, transitions, and final results.
Make sure that the captions have enough reading time and do not overlap the numbers.
