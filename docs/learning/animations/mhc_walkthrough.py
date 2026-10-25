"""Follow sat through an attention sublayer, one operation at a time.

The renderer is an optional documentation tool. See README.md in this folder.
ManimGL has no usable typing interface in the bootstrap environment.
"""

# mypy: disable-error-code="import-not-found,no-any-return,misc"

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import manimlib as mn
from mhc_toy import INPUT_SEQUENCE, trace

if TYPE_CHECKING:
    from mhc_toy import Matrix

INK = "#edf3fc"
MUTED = "#a7bacf"
GOLD = "#ffce7b"
TEAL = "#73dfbf"
CHAPTER_SECONDS = 40
FRAME_HALF_WIDTH = 7.1
FRAME_HALF_HEIGHT = 3.96
CHAPTERS = (
    "Start with the token embeddings",
    "Initialize four residual streams",
    "Apply the input mapping",
    "Apply RMSNorm",
    "Compute the value matrix",
    "Compute the attention scores",
    "Compute the attention weights",
    "Compute the attention output",
    "Apply the residual mapping",
    "Apply the output mapping",
    "Compute the output hidden state",
)
STREAM_LABELS = ("1", "2", "3", "4")


def label(message: str, size: int = 28, color: str = INK) -> mn.Text:
    """Make a lesson label without a LaTeX dependency.

    Args:
        message: Visible text.
        size: Font size in Manim units.
        color: Text color as a hexadecimal string.

    Returns:
        A Pango-rendered text object.
    """
    return mn.Text(message, font="Arial", font_size=size, color=color)


def number(value: float) -> str:
    """Show up to three decimals without unnecessary trailing zeros.

    Args:
        value: An element of the numerical trace.

    Returns:
        The rounded display number.
    """
    return f"{value:.3f}".rstrip("0").rstrip(".")


def matrix_table(values: Matrix) -> mn.VGroup:
    """Draw large numerical elements and line-segment brackets.

    Args:
        values: Numeric rows to display.

    Returns:
        A group containing the rows and brackets.
    """
    rows = mn.VGroup()
    for row_index, values_row in enumerate(values):
        row = mn.VGroup()
        for column_index, value in enumerate(values_row):
            cell = mn.Text(number(value), font="Menlo", font_size=36, color=INK)
            cell.move_to([column_index * 1.4, -row_index * 0.66, 0])
            row.add(cell)
        rows.add(row)
    brackets = mn.VGroup()
    for direction in (mn.LEFT, mn.RIGHT):
        x_position = rows.get_corner(direction)[0] + direction[0] * 0.2
        top = rows.get_top()[1] + 0.14
        bottom = rows.get_bottom()[1] - 0.14
        brackets.add(
            mn.Line([x_position, top, 0], [x_position, bottom, 0], color=MUTED),
            mn.Line([x_position, top, 0], [x_position - direction[0] * 0.13, top, 0], color=MUTED),
            mn.Line(
                [x_position, bottom, 0], [x_position - direction[0] * 0.13, bottom, 0], color=MUTED
            ),
        )
    return mn.VGroup(rows, brackets).center()


def panel(
    name: str,
    values: Matrix,
    row_labels: tuple[str, ...] = (),
    column_labels: tuple[str, ...] = (),
) -> mn.VGroup:
    """Label a numerical matrix and, when needed, its token or stream axis.

    Args:
        name: Quantity name or symbol.
        values: Numeric rows to display.
        row_labels: Token names or residual stream indices.
        column_labels: Token names after transposing the key matrix.

    Returns:
        Matrix, title, row labels, and column labels, in that order.
    """
    matrix = matrix_table(values)
    rows = matrix[0]
    row_names = mn.VGroup()
    column_names = mn.VGroup()
    if row_labels:
        for row, text in zip(rows, row_labels, strict=True):
            row_names.add(label(text, 25, MUTED).next_to(row, mn.LEFT, buff=0.4))
    if column_labels:
        for index, text in enumerate(column_labels):
            column = mn.VGroup(*(row[index] for row in rows))
            column_names.add(label(text, 25, MUTED).next_to(column, mn.UP, buff=0.25))
    title = label(name, 27).next_to(mn.VGroup(matrix, column_names), mn.UP, buff=0.3)
    return mn.VGroup(matrix, title, row_names, column_names).center()


class MHCTokenWalkthrough(mn.Scene):
    """Carry the same example through eleven focused calculation stages."""

    def construct(self) -> None:
        """Render fixed-duration chapters with persistent numerical results."""
        self.clear()
        self.data = trace()
        self.heading = mn.VGroup()
        self.annotation = mn.VGroup()
        self.carry = mn.VGroup()
        self.input_sequence = (
            mn.VGroup(
                *(
                    label(token, 23, GOLD if token == INPUT_SEQUENCE[-1] else MUTED)
                    for token in INPUT_SEQUENCE
                )
            )
            .arrange(mn.RIGHT, buff=0.45)
            .move_to([0, 3.65, 0])
        )
        self.input_sequence.add(
            mn.SurroundingRectangle(self.input_sequence[-1], color=GOLD, buff=0.09)
        )
        self.add(self.input_sequence)
        steps = (
            self.token_embeddings,
            self.residual_streams,
            self.input_mapping,
            self.rmsnorm,
            self.value_matrix,
            self.attention_scores,
            self.attention_weights,
            self.attention_output,
            self.residual_mapping,
            self.output_mapping,
            self.output_hidden_state,
        )
        for index, step in enumerate(steps):
            start = self.time
            self.remove(self.heading, self.annotation)
            self.annotation = mn.VGroup()
            self.heading = label(f"{index + 1}. {CHAPTERS[index]}", 36).move_to([0, 2.85, 0])
            self.add(self.heading)
            step()
            self.check_frame()
            # Rounding avoids an extra frame from floating-point wait durations.
            remaining = round(CHAPTER_SECONDS - (self.time - start), 6)
            if remaining < 0:
                message = f"Chapter {index + 1} exceeded its {CHAPTER_SECONDS}-second slot"
                raise ValueError(message)
            self.wait(remaining)

    def cue(self, message: str, size: int = 28) -> None:
        """Replace the explanation instead of accumulating text blocks.

        Args:
            message: One short explanation or calculation.
            size: Font size for the explanation.
        """
        self.remove(self.annotation)
        self.annotation = label(message, size).move_to([0, -2.95, 0])
        self.play(mn.FadeIn(self.annotation), run_time=0.3)
        self.check_frame()

    def explain(self, *messages: str) -> None:
        """Give each purpose caption six seconds in the existing caption area."""
        for message in messages:
            self.cue(message)
            self.wait(6)

    def keep_only(self, *items: mn.VGroup) -> None:
        """Fade completed work while retaining the named numerical objects.

        Args:
            items: Matrices or vectors that remain visible.
        """
        self.add(self.input_sequence, self.heading, *items)
        keep = (self.input_sequence, self.heading, self.annotation, *items)
        fading = [item for item in self.mobjects if item not in keep]
        if fading:
            self.play(*(mn.FadeOut(item) for item in fading), run_time=0.5)

    def finish(self, result: mn.VGroup, message: str) -> None:
        """Leave one large result for the chapter ending and next chapter.

        Args:
            result: The completed numerical object.
            message: The explanation accompanying that result.
        """
        self.keep_only(result)
        result[0][0].set_color(TEAL)
        self.play(result.animate.move_to([0, 0, 0]).set_opacity(1), run_time=0.8)
        self.carry = result
        self.cue(message)

    def reveal_row(self, source: mn.VGroup, target: mn.VGroup, *, identical: bool = False) -> None:
        """Copy identical numbers or reveal a computed result without glyph morphing.

        Args:
            source: The source elements that explain the result.
            target: The result row or element, initially hidden.
            identical: Whether the source and target contain the same numbers.
        """
        visible = target.copy().set_opacity(1).set_color(TEAL)
        if identical:
            self.play(mn.TransformFromCopy(source, visible), run_time=0.8)
        else:
            self.play(mn.FadeIn(visible, shift=0.12 * mn.UP), run_time=0.8)
        self.remove(visible)
        target.set_opacity(1).set_color(TEAL)

    def product_row(
        self, left: mn.VGroup, right: mn.VGroup, result: mn.VGroup, calculation: str
    ) -> None:
        """Explain one result row through successive row-column products.

        Args:
            left: Left operand; its first row is used.
            right: Right operand.
            result: A one-row result panel.
            calculation: Worked arithmetic for its first element.
        """
        result.move_to([0, -1.85, 0])
        result[0][0].set_opacity(0)
        self.add(result)
        left[0][0].set_opacity(0.18)
        left_row = left[0][0][0]
        left_row.set_opacity(1).set_color(GOLD)
        left_box = mn.SurroundingRectangle(left_row, color=GOLD, buff=0.1)
        self.play(mn.ShowCreation(left_box), run_time=0.4)
        for index, element in enumerate(result[0][0][0]):
            right[0][0].set_opacity(0.18).set_color(INK)
            column = mn.VGroup(*(row[index] for row in right[0][0]))
            column.set_opacity(1).set_color(GOLD)
            box = mn.SurroundingRectangle(column, color=GOLD, buff=0.1)
            self.play(mn.ShowCreation(box), run_time=0.4)
            if index == 0:
                self.cue(calculation, 25)
                self.wait(1)
            else:
                self.cue("Repeat for the next element.")
            self.reveal_row(mn.VGroup(left_row, column), element)
            self.play(mn.FadeOut(box), run_time=0.2)
        self.add(left, right, result)
        self.play(mn.FadeOut(left_box), run_time=0.2)

    def assemble(self, source: mn.VGroup, target: mn.VGroup, source_row: int) -> None:
        """Place the worked row into a matrix and reveal the remaining results.

        Args:
            source: The worked one-row panel.
            target: The complete matrix with explicit row labels.
            source_row: The corresponding row index in the complete matrix.
        """
        self.keep_only(source)
        self.play(source.animate.move_to([0, -1.9, 0]), run_time=0.4)
        target.move_to([0, 0.4, 0])
        target[0][0].set_opacity(0)
        self.add(target)
        self.reveal_row(source[0][0][0], target[0][0][source_row], identical=True)
        self.play(mn.FadeOut(source), run_time=0.4)
        for index, row in enumerate(target[0][0]):
            if index != source_row:
                self.play(row.animate.set_opacity(1).set_color(TEAL), run_time=0.5)

    def check_frame(self) -> None:
        """Reject content outside the frame at captions and chapter endings."""
        for item in self.mobjects:
            if (
                item.get_left()[0] < -FRAME_HALF_WIDTH
                or item.get_right()[0] > FRAME_HALF_WIDTH
                or item.get_top()[1] > FRAME_HALF_HEIGHT
                or item.get_bottom()[1] < -FRAME_HALF_HEIGHT
            ):
                message = f"Object outside the video frame: {item}"
                raise ValueError(message)

    def token_embeddings(self) -> None:
        """Select sat from the complete input sequence."""
        embeddings = panel("Token embeddings E", self.data["token_embeddings"], INPUT_SEQUENCE)
        self.play(mn.FadeIn(embeddings))
        self.explain(
            "Next-token prediction needs information from the input sequence.",
            "sat's token embedding alone cannot represent this input sequence.",
            "Token labels identify tokens. Token embeddings supply numbers for computation.",
            "Training learns token embeddings. These numbers are chosen for the example.",
        )
        embeddings[0][0].set_opacity(0.2)
        embeddings[0][0][-1].set_opacity(1).set_color(GOLD)
        selected = panel("sat's token embedding", [self.data["token_embeddings"][-1]])
        selected.move_to([0, -1.7, 0])
        selected[0][0].set_opacity(0)
        self.add(selected)
        self.reveal_row(embeddings[0][0][-1], selected[0][0][0], identical=True)
        self.finish(selected, "Use this token embedding for all four residual streams.")

    def residual_streams(self) -> None:
        """Copy the carried token embedding into four labeled residual streams."""
        self.explain(
            "Four residual streams let sat retain different vectors across layers.",
            "Copying gives every residual stream the same starting information.",
            "Later, different attention contributions can make the residual streams different.",
        )
        embedding = self.carry
        self.play(embedding.animate.move_to([-3.5, 0.2, 0]))
        state = panel("Input hidden state X0", self.data["input_hidden_state"], STREAM_LABELS)
        state.move_to([3.1, 0.2, 0])
        state[0][0].set_opacity(0)
        self.add(state)
        self.cue("All four residual streams belong to sat.")
        for row in state[0][0]:
            self.reveal_row(embedding[0][0][0], row, identical=True)
        self.finish(state, "Keep X0 for the residual mapping.")

    def input_mapping(self) -> None:
        """Combine the carried input hidden state with the input mapping."""
        self.explain(
            "The attention sublayer takes one vector per token.",
            "The input mapping lets all four residual streams supply that vector.",
            "X0 stays available, so this combination does not replace the saved hidden state.",
        )
        state = self.carry
        state[0][0].set_color(INK)
        self.play(state.animate.move_to([3.25, 0.65, 0]))
        mapping = panel("Input mapping a", self.data["input_mapping"])
        mapping.move_to([-3.35, 0.65, 0])
        self.play(mn.FadeIn(mapping), mn.FadeIn(label("@", 38).move_to([0, 0.45, 0])))
        result = panel("u_sat", self.data["input_mapping_output"])
        self.product_row(mapping, state, result, "0.2(7) + 0.4(7) + 0.6(7) + 0.8(7) = 14")
        self.finish(result, "Four residual streams produce one vector.")

    def rmsnorm(self) -> None:
        """Apply RMSNorm after completing the input mapping."""
        self.explain(
            "Larger inputs can produce larger attention scores just because of their size.",
            "RMSNorm gives the projections a more consistent input scale.",
            "Dividing all elements by the same number keeps their relative pattern.",
        )
        combined = self.carry
        combined[0][0].set_color(INK)
        self.cue("RMSNorm rescales this vector.")
        denominator = label("√((14² + (-16)² + 18²) / 3 + 0.000001) ≈ 16.083", 28)
        denominator.move_to([0, -1.6, 0])
        self.play(mn.FadeIn(denominator))
        self.wait(2)
        self.play(mn.FadeOut(denominator))
        normalized = panel("RMSNorm output z_sat", [self.data["rmsnorm_output"][-1]])
        normalized.move_to([3.3, 0.3, 0])
        self.play(combined.animate.move_to([-3.3, 0.3, 0]))
        division = label("÷ 16.083 ≈", 28).move_to([0, 0.1, 0])
        self.play(mn.FadeIn(division), mn.FadeIn(normalized))
        self.wait(1)
        self.cue("Apply RMSNorm to each token in the input sequence.")
        result = panel("RMSNorm output Z", self.data["rmsnorm_output"], INPUT_SEQUENCE)
        self.assemble(normalized, result, 2)
        self.finish(result, "Rows follow the input sequence: The, cat, sat.")

    def value_matrix(self) -> None:
        """Work sat's row before assembling the complete value matrix."""
        self.explain(
            "Attention needs information to combine from each input position.",
            "The value projection lets training shape that information for later layers.",
            "Here, Wv is chosen for the example, rather than learned through training.",
        )
        normalized = self.carry
        normalized[0][0].set_opacity(0.18)
        normalized[0][0][-1].set_opacity(1).set_color(GOLD)
        selected = panel("sat's row of Z", [self.data["rmsnorm_output"][-1]])
        selected.move_to([-3.25, 0.65, 0])
        selected[0][0].set_opacity(0)
        self.add(selected)
        self.reveal_row(normalized[0][0][-1], selected[0][0][0], identical=True)
        self.keep_only(selected)
        projection = panel("Value projection Wv", self.data["value_projection"])
        projection.move_to([3.25, 0.65, 0])
        self.play(mn.FadeIn(projection), mn.FadeIn(label("@", 38).move_to([0, 0.45, 0])))
        value = panel("sat's value vector", [self.data["value_matrix"][-1]])
        self.product_row(
            selected, projection, value, "0.870(0.4) + (-0.995)(0) + 1.119(0.1) ≈ 0.460"
        )
        self.cue("Apply the same value projection to the other input tokens.")
        result = panel("Value matrix V", self.data["value_matrix"], INPUT_SEQUENCE)
        self.assemble(value, result, 2)
        self.finish(result, "Keep V for the attention output.")

    def attention_scores(self) -> None:
        """Transpose the labeled key matrix and compute one score at a time."""
        self.explain(
            "Query-key comparisons let attention weight each input position differently.",
            "Longer vectors tend to give larger dot products.\n"
            "The square-root divisor corrects their typical size.",
            "Otherwise, vector length alone can push softmax\n"
            "toward almost all weight on one position.",
        )
        self.cue("Keep V. The identity key projection gives K = Z.")
        self.keep_only()
        keys = panel("Key matrix K = Z", self.data["rmsnorm_output"], INPUT_SEQUENCE)
        self.play(mn.FadeIn(keys))
        query = panel("Query vector q", self.data["query_vector"])
        query.move_to([-3.25, 0.65, 0])
        query[0][0].set_opacity(0)
        self.add(query)
        self.play(keys.animate.move_to([3.25, 0.65, 0]))
        self.reveal_row(keys[0][0][-1], query[0][0][0], identical=True)
        transposed = panel("Kᵀ", self.data["transposed_key_matrix"], (), INPUT_SEQUENCE)
        transposed.move_to([3.25, 0.65, 0])
        self.cue("Transpose K: token rows become token columns.")
        query[0][0].set_opacity(0.18)
        offset = transposed[0][0][0][0].get_center() - keys[0][0][0][0].get_center()
        self.play(
            keys[0][0].animate.shift(offset),
            mn.FadeOut(keys[1]),
            mn.FadeOut(keys[2]),
            mn.Transform(keys[0][1], transposed[0][1]),
            run_time=0.5,
        )
        keys[0][0].set_opacity(0.18)
        for row, column in ((0, 1), (0, 2), (1, 2)):
            first = keys[0][0][row][column]
            second = keys[0][0][column][row]
            first.set_opacity(1).set_color(GOLD)
            second.set_opacity(1).set_color(GOLD)
            self.play(
                mn.Transform(
                    first,
                    first.copy().move_to(transposed[0][0][column][row]),
                    path_arc=math.pi / 2,
                ),
                mn.Transform(
                    second,
                    second.copy().move_to(transposed[0][0][row][column]),
                    path_arc=math.pi / 2,
                ),
                run_time=0.6,
            )
            first.set_opacity(0.18).set_color(INK)
            second.set_opacity(0.18).set_color(INK)
        self.remove(keys)
        transposed[1].set_opacity(0)
        transposed[3].set_opacity(0)
        self.add(transposed)
        self.play(
            transposed[1].animate.set_opacity(1),
            transposed[3].animate.set_opacity(1),
            run_time=0.4,
        )
        result = panel("Attention scores", self.data["attention_scores"], (), INPUT_SEQUENCE)
        self.product_row(
            query,
            transposed,
            result,
            "(0.870(0.463) + (-0.995)(0.926) + 1.119(1.389)) / √3 ≈ 0.598",
        )
        self.finish(result, "These attention scores are the input to softmax.")

    def attention_weights(self) -> None:
        """Apply softmax without retaining the preceding dot-product calculation."""
        self.explain(
            "Attention needs weights for combining the value vectors.",
            "Softmax gives larger scores more weight, with a total weight of one.",
            "These weights apply to input positions.\nThey are not next-token probabilities.",
        )
        scores = self.carry
        scores[0][0].set_color(INK)
        self.play(scores.animate.move_to([-3.1, 0.25, 0]))
        result = panel("Attention weights", self.data["attention_weights"], (), INPUT_SEQUENCE)
        result.move_to([3.1, 0.25, 0])
        arrow = mn.Arrow(scores.get_right(), result.get_left(), buff=0.25)
        operation = label("softmax", 27).next_to(arrow, mn.UP, buff=0.25)
        self.cue("Exponentiate each attention score.")
        self.play(mn.GrowArrow(arrow), mn.FadeIn(operation))
        self.wait(2)
        self.cue("Divide each result by the sum of those results.")
        self.play(scores.animate.set_opacity(0.25), mn.FadeIn(result), run_time=1.2)
        self.finish(result, "The attention weights sum to one.")

    def attention_output(self) -> None:
        """Combine the attention weights with matching value vectors."""
        self.explain(
            "sat needs information from The and cat as well as itself.",
            "The weighted sum brings their value vectors into one attention output.",
            "Unlike sat's token embedding alone, y depends on the input sequence.",
        )
        weights = self.carry
        weights[0][0].set_color(INK)
        self.play(weights.animate.move_to([-3.2, 0.65, 0]))
        values = panel("Value matrix V", self.data["value_matrix"], INPUT_SEQUENCE)
        values.move_to([3.25, 0.65, 0])
        self.cue("Use the value matrix V from step 5.")
        self.play(mn.FadeIn(values), mn.FadeIn(label("@", 38).move_to([0, 0.45, 0])))
        result = panel("Attention output y", self.data["attention_output"])
        self.product_row(
            weights, values, result, "0.219(0.324) + 0.099(-0.197) + 0.682(0.460) ≈ 0.365"
        )
        self.finish(result, "The attention output projection is an identity matrix.")

    def residual_mapping(self) -> None:
        """Check C separately, then use the original input hidden state."""
        self.explain(
            "The residual path carries existing information.\n"
            "Mixing lets residual streams share it."
        )
        self.cue("Keep y. Return to input hidden state X0.")
        self.keep_only()
        mapping = panel("Residual mapping C", self.data["residual_mapping"])
        self.play(mn.FadeIn(mapping))
        row_box = mn.SurroundingRectangle(mapping[0][0][0], color=GOLD, buff=0.1)
        self.play(mn.ShowCreation(row_box), run_time=0.4)
        self.explain(
            "Each row sums to one. With nonnegative elements,\n"
            "its result stays within the input range."
        )
        column = mn.VGroup(*(row[0] for row in mapping[0][0]))
        column_box = mn.SurroundingRectangle(column, color=GOLD, buff=0.1)
        self.play(mn.ReplacementTransform(row_box, column_box), run_time=0.4)
        self.explain("Each column sums to one.\nThis preserves the mean across residual streams.")
        self.play(mn.FadeOut(column_box), mapping.animate.move_to([-3.35, 0.65, 0]))
        state = panel("Input hidden state X0", self.data["input_hidden_state"], STREAM_LABELS)
        state.move_to([3.25, 0.65, 0])
        self.play(mn.FadeIn(state), mn.FadeIn(label("@", 38).move_to([0, 0.45, 0])))
        first = panel("B: residual stream 1", [self.data["residual_contribution"][0]])
        self.product_row(mapping, state, first, "0.5(7) + 0.25(7) + 0.125(7) + 0.125(7) = 7")
        self.cue("The input residual streams are identical.")
        result = panel("Residual contribution B", self.data["residual_contribution"], STREAM_LABELS)
        self.assemble(first, result, 0)
        self.finish(
            result,
            "Repeated residual mappings keep these bounds.\n"
            "Attention can still change the hidden state.",
        )

    def output_mapping(self) -> None:
        """Build one attention-contribution row per output-mapping element."""
        self.explain(
            "One attention output must update four residual streams.",
            "Separate scales let each residual stream receive a different amount.",
            "Here, these amounts make the initially identical residual streams become different.",
        )
        self.cue("Keep B. Use attention output y.")
        self.keep_only()
        mapping = panel("Output mapping b", self.data["output_mapping"], STREAM_LABELS)
        mapping.move_to([-5.45, 0.5, 0])
        output = panel("Attention output y", self.data["attention_output"])
        output.move_to([-1.75, 0.5, 0])
        result = panel(
            "Attention contribution A", self.data["attention_contribution"], STREAM_LABELS
        )
        result.move_to([3.8, 0.5, 0])
        result[0][0].set_opacity(0)
        self.add(result)
        self.play(
            mn.FadeIn(mapping),
            mn.FadeIn(output),
            mn.FadeIn(label("@", 38).move_to([-4.45, 0.3, 0])),
            mn.FadeIn(label("=", 38).move_to([0.85, 0.3, 0])),
        )
        for index, row in enumerate(result[0][0]):
            mapping[0][0].set_opacity(0.18)
            element = mapping[0][0][index]
            element.set_opacity(1).set_color(GOLD)
            result[0][0].set_opacity(0.18)
            for hidden_row in result[0][0][index:]:
                hidden_row.set_opacity(0)
            factor = number(self.data["output_mapping"][index][0])
            self.cue(f"Multiply y by {factor} for residual stream {index + 1}.")
            self.reveal_row(output[0][0][0], row)
        result[0][0].set_opacity(1)
        self.finish(result, "Add this attention contribution to B.")

    def output_hidden_state(self) -> None:
        """Add one pair of rows, then present the complete output hidden state."""
        self.explain(
            "Adding B and A retains carried information\nand includes information from attention.",
            "Without B, only the attention contribution remains.\n"
            "Without A, no attention information is added.",
        )
        attention = self.carry
        attention_row = panel("A: residual stream 1", [self.data["attention_contribution"][0]])
        attention_row.move_to([3.15, 0.65, 0])
        attention_row[0][0].set_opacity(0)
        self.add(attention_row)
        self.reveal_row(attention[0][0][0], attention_row[0][0][0], identical=True)
        self.keep_only(attention_row)
        residual_row = panel("B: residual stream 1", [self.data["residual_contribution"][0]])
        residual_row.move_to([-3.15, 0.65, 0])
        self.play(mn.FadeIn(residual_row), mn.FadeIn(label("+", 40).move_to([0, 0.45, 0])))
        first = panel("X1: residual stream 1", [self.data["output_hidden_state"][0]])
        first.move_to([0, -1.85, 0])
        first[0][0].set_opacity(0)
        self.add(first)
        self.cue("Add matching elements.")
        for index, element in enumerate(first[0][0][0]):
            residual_row[0][0].set_opacity(0.18)
            attention_row[0][0].set_opacity(0.18)
            pair = mn.VGroup(residual_row[0][0][0][index], attention_row[0][0][0][index])
            pair.set_opacity(1).set_color(GOLD)
            self.reveal_row(pair, element)
        self.cue("Repeat for the other residual streams.")
        result = panel("Output hidden state X1", self.data["output_hidden_state"], STREAM_LABELS)
        self.assemble(first, result, 0)
        self.finish(result, "X1 supplies all four residual streams to the next sublayer.")
        self.wait(6)
        self.cue(
            "Later layers use this information for next-token prediction.\nThis example ends at X1."
        )
