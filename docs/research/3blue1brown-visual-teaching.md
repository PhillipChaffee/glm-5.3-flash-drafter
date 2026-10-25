# Visual teaching study for the mHC lesson

The user asked for a simpler animation, informed by
[3Blue1Brown's Neural Networks topic](https://www.3blue1brown.com/?topic=neural-networks&lesson=neural-networks).
This study covers all nine available written articles, four inline animation
clips, and eleven diagrams. The two remaining topic entries have no written
article. The full-length YouTube videos were not watched in their entirety.

## Written material reviewed

The inventory follows the site's
[topic definition](https://github.com/3b1b/3Blue1Brown.com/blob/main/app/data/topics.json).

| Lesson | Reading scope |
| --- | --- |
| [But what is a Neural Network?](https://www.3blue1brown.com/lessons/neural-networks/) | Full article and available answers |
| [Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent/) | Full article and available answers |
| [Analyzing our neural network](https://www.3blue1brown.com/lessons/neural-network-analysis/) | Full article; interactive behavior not assessed |
| [Backpropagation](https://www.3blue1brown.com/lessons/backpropagation/) | Full article and available answers |
| [Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus/) | Full article and reveal answer |
| [Large Language Models explained briefly](https://www.3blue1brown.com/lessons/mini-llm/) | Full article |
| [Transformers](https://www.3blue1brown.com/lessons/gpt/) | Full article and available answers |
| [Attention](https://www.3blue1brown.com/lessons/attention/) | Full article, sidebars, and available answers |
| [How might LLMs store facts](https://www.3blue1brown.com/lessons/mlp/) | Full article, corrections, and available answers |
| [Cross-Entropy](https://www.3blue1brown.com/lessons/cross-entropy/) | Video listing only; no written article |
| [AI images and videos](https://www.3blue1brown.com/lessons/diffusion-models/) | Video listing only; no written article |

## Animation observations

I inspected sequential frames across these public clips, sampled twice per second.
The durations below come from their media metadata.

| Clip | Duration | Visible technique |
| --- | ---: | --- |
| [Pixels to neurons](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2017/neural-networks/pixels-to-neurons.mp4) | 9.31 s | The same pixel objects move from a grid into the input sequence. Their identity survives the change in layout. |
| [Activation vector](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2017/neural-networks/matrix-math-videos/activation-vector.mp4) | 5.07 s | Existing quantities move into a vector. One element receives attention while other elements become dim. |
| [Weight matrix](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2017/neural-networks/matrix-math-videos/weight-matrix.mp4) | 9.11 s | The matrix forms gradually. A highlighted row connects to the corresponding computation. |
| [Weighted sum](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2017/neural-networks/matrix-math-videos/weighted-sum.mp4) | 13.51 s | One row-vector product produces one output element. The focus then moves to the next row. |

The images support the same approach:

- [Dot product](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/gpt/HowToDot.jpg)
  connects matching elements to their products and sum.
- [Attention scores](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/attention/Score.jpg)
  and [attention weights](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/attention/Normalized.jpg)
  keep token positions fixed while the numbers change.
- [Value vectors](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/attention/ValueVector.jpg)
  and [the resulting vector](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/attention/DeltaE.jpg)
  dim unrelated columns and isolate the selected token.
- [The row view](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/mlp/Step12.jpg)
  and [column view](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/mlp/Column1.jpg)
  explain matrix multiplication through one vector operation.
- [Parameters and data](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/gpt/distinction.jpg),
  [embeddings](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/gpt/embedding.jpg),
  [the output matrix](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/attention/OutputMatrix.jpg),
  and [the MLP summary](https://storage.googleapis.com/3blue1brown-website-bucket/lessons/2024/mlp/MLPEnd.jpg)
  distinguish objects by their role and place in the computation.

These scenes can contain many numbers. Dimming, motion, and narration direct
attention to a small part of the scene. Our silent video needs less simultaneous
reading than a narrated video.

## Changes for our lesson

The previous video kept an eight-node path map, input sequence, heading,
subheading, matrices, arithmetic, and captions on screen together. Several scenes
showed three full matrices at once. Chapter transitions cleared the scene, so
viewers had to find the same quantities again.

The revised scene follows these rules:

1. Keep one compact input sequence and one operation title.
2. Show one operation at a time, with one short explanation.
3. Enlarge the numbers and omit unnecessary trailing zeros in the video.
4. Highlight the active row or column, and dim the rest.
5. Carry computed vectors into the next operation with visible motion.
6. Replace completed inputs with the result instead of accumulating panels.
7. Give RMSNorm and softmax separate steps.
8. Label token rows and residual-stream rows explicitly.
9. Reserve gold for the current focus and teal for a computed result.
10. Move copies of identical numbers, but reveal changed numbers without morphing
    their glyphs. Swap one matrix-element pair at a time during the transpose.

The input sequence, signed token embeddings, numerical computation, and glossary
remain the basis of the example. In particular, the attention contribution is
still `A = b @ y`. Individual weighted value vectors are not renamed as attention
contributions. The source lessons' temporary “Value-down” and “Value-up” names are
not introduced into our glossary.

The written lesson retains the full arithmetic. The video must show the actual
vectors and matrices, but it does not need every equation visible at once.

## Verification

Review frames during calculations and transitions, as well as chapter endings.
Make sure that the active operation is identifiable without reading multiple
text blocks. Compare displayed results with the existing numerical trace, and
test the revised chapter controls. User feedback is still needed to establish
whether the presentation is easier to follow.
