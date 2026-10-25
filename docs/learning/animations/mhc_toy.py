"""Compute the lesson's attention sublayer with mHC.

Run this file to print the matrices with the same names used in the lesson.
"""

from __future__ import annotations

import json
import math

type Matrix = list[list[float]]

INPUT_SEQUENCE = ("The", "cat", "sat")
TOKEN_EMBEDDINGS = [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]]
INPUT_MAPPING = [[0.2, 0.4, 0.6, 0.8]]
OUTPUT_MAPPING = [[0.25], [0.5], [1.0], [1.5]]
RESIDUAL_MAPPING = [
    [0.5, 0.25, 0.125, 0.125],
    [0.125, 0.5, 0.25, 0.125],
    [0.125, 0.125, 0.5, 0.25],
    [0.25, 0.125, 0.125, 0.5],
]
VALUE_PROJECTION = [[0.4, 0.0, -0.2], [0.0, 0.2, 0.1], [0.1, 0.1, 0.0]]
EPSILON = 1e-6


def matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply the toy matrices, rejecting mismatched inner dimensions.

    Args:
        left: Rows of the left matrix.
        right: Rows of the right matrix.

    Returns:
        The matrix product, with one result row per left row.
    """
    columns = list(zip(*right, strict=True))
    return [
        [sum(a * b for a, b in zip(row, column, strict=True)) for column in columns] for row in left
    ]


def trace() -> dict[str, Matrix]:
    """Compute sat's output hidden state from the fixed input sequence.

    Returns:
        Named matrices shared by the animation and its numerical checks.
    """
    input_hidden_state = [TOKEN_EMBEDDINGS[-1].copy() for _ in OUTPUT_MAPPING]
    input_mapping_output = matmul(INPUT_MAPPING, input_hidden_state)
    sequence_input_mapping_output = [
        matmul(INPUT_MAPPING, [row] * len(OUTPUT_MAPPING))[0] for row in TOKEN_EMBEDDINGS
    ]
    rmsnorm_output = [
        [element / math.sqrt(sum(x * x for x in row) / len(row) + EPSILON) for element in row]
        for row in sequence_input_mapping_output
    ]
    # Query, key, and attention output projections are identities; sat can attend to all positions.
    query_vector = [rmsnorm_output[-1]]
    transposed_key_matrix = [list(column) for column in zip(*rmsnorm_output, strict=True)]
    attention_scores = [
        [
            score / math.sqrt(len(query_vector[0]))
            for score in matmul(query_vector, transposed_key_matrix)[0]
        ]
    ]
    exponentials = [math.exp(score - max(attention_scores[0])) for score in attention_scores[0]]
    attention_weights = [[element / sum(exponentials) for element in exponentials]]
    value_matrix = matmul(rmsnorm_output, VALUE_PROJECTION)
    attention_output = matmul(attention_weights, value_matrix)
    residual_contribution = matmul(RESIDUAL_MAPPING, input_hidden_state)
    attention_contribution = matmul(OUTPUT_MAPPING, attention_output)
    output_hidden_state = [
        [
            residual + attention
            for residual, attention in zip(residual_row, attention_row, strict=True)
        ]
        for residual_row, attention_row in zip(
            residual_contribution, attention_contribution, strict=True
        )
    ]
    return {
        "token_embeddings": TOKEN_EMBEDDINGS,
        "input_hidden_state": input_hidden_state,
        "input_mapping": INPUT_MAPPING,
        "input_mapping_output": input_mapping_output,
        "sequence_input_mapping_output": sequence_input_mapping_output,
        "rmsnorm_output": rmsnorm_output,
        "query_vector": query_vector,
        "transposed_key_matrix": transposed_key_matrix,
        "value_projection": VALUE_PROJECTION,
        "value_matrix": value_matrix,
        "attention_scores": attention_scores,
        "attention_weights": attention_weights,
        "attention_output": attention_output,
        "residual_mapping": RESIDUAL_MAPPING,
        "residual_contribution": residual_contribution,
        "output_mapping": OUTPUT_MAPPING,
        "attention_contribution": attention_contribution,
        "output_hidden_state": output_hidden_state,
    }


if __name__ == "__main__":
    print(json.dumps(trace(), indent=2))
