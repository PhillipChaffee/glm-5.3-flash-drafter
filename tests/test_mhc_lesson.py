import ast
import json
import math
import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANIMATIONS = ROOT / "docs/learning/animations"
LESSON = ROOT / "docs/learning/lessons/0001-follow-one-token-through-mhc.html"


@pytest.fixture(scope="module")
def toy() -> dict[str, list[list[float]]]:
    values: dict[str, list[list[float]]] = runpy.run_path(str(ANIMATIONS / "mhc_toy.py"))["trace"]()
    return values


def test_attention_is_computed_from_the_input_sequence(toy: dict[str, list[list[float]]]) -> None:
    assert toy["input_mapping_output"][0] == pytest.approx([14.0, -16.0, 18.0])
    assert toy["attention_weights"][0] == pytest.approx(
        [0.21937751154245277, 0.09890537905484238, 0.6817171094027048], abs=1e-12
    )
    assert sum(toy["attention_weights"][0]) == pytest.approx(1)
    assert toy["attention_output"][0] == pytest.approx(
        [0.36522867240750445, 0.04298044308764282, -0.1611241146599308], abs=1e-12
    )
    assert all(weight > 0 for weight in toy["attention_weights"][0])


def test_the_same_final_token_is_carried_through_every_stage(
    toy: dict[str, list[list[float]]],
) -> None:
    assert toy["token_embeddings"] == [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]]
    assert toy["input_hidden_state"] == [toy["token_embeddings"][-1]] * 4
    assert toy["sequence_input_mapping_output"][-1] == toy["input_mapping_output"][0]
    assert toy["query_vector"] == [toy["rmsnorm_output"][-1]]
    assert toy["transposed_key_matrix"] == [
        list(column) for column in zip(*toy["rmsnorm_output"], strict=True)
    ]
    assert toy["output_hidden_state"][-1] == pytest.approx(
        [7.547843008611257, -7.935529335368535, 8.758313828010104], abs=1e-12
    )
    assert toy["input_hidden_state"] == [[7.0, -8.0, 9.0]] * 4


def test_residual_mapping_preserves_the_stream_mean(
    toy: dict[str, list[list[float]]],
) -> None:
    mapping = toy["residual_mapping"]
    assert all(element >= 0 for row in mapping for element in row)
    assert all(math.isclose(sum(row), 1) for row in mapping)
    assert all(math.isclose(sum(column), 1) for column in zip(*mapping, strict=True))
    assert mapping[0][1] != mapping[1][0]
    assert toy["residual_contribution"] == toy["input_hidden_state"]
    assert len({tuple(row) for row in toy["output_hidden_state"]}) == 4
    multiply = runpy.run_path(str(ANIMATIONS / "mhc_toy.py"))["matmul"]
    mixed: list[list[float]] = multiply(mapping, toy["output_hidden_state"])
    assert mixed != toy["output_hidden_state"]
    # This asymmetric fixture fails if output/input axes are transposed.
    assert mixed[0] == pytest.approx(
        [7.2054411282292214, -7.975823500763201, 8.909367685503788], abs=1e-12
    )
    assert multiply([[0.25] * 4], mixed)[0] == pytest.approx(
        multiply([[0.25] * 4], toy["output_hidden_state"])[0]
    )
    before = sum(value * value for row in toy["output_hidden_state"] for value in row)
    after = sum(value * value for row in mixed for value in row)
    assert after <= before


def test_output_mapping_scales_every_element(toy: dict[str, list[list[float]]]) -> None:
    for residual, mapping_row, result in zip(
        toy["residual_contribution"], toy["output_mapping"], toy["output_hidden_state"], strict=True
    ):
        difference = [new - previous for previous, new in zip(residual, result, strict=True)]
        assert difference == pytest.approx(
            [mapping_row[0] * element for element in toy["attention_output"][0]]
        )
    assert toy["output_mapping"][-1][0] > 1
    assert sum(toy["input_mapping"][0]) == 2


def test_matrix_product_rejects_mismatched_dimensions() -> None:
    multiply = runpy.run_path(str(ANIMATIONS / "mhc_toy.py"))["matmul"]
    with pytest.raises(ValueError, match="zip"):
        multiply([[1.0, 2.0]], [[3.0]])


def test_printed_trace_runs_without_manim(
    capsys: pytest.CaptureFixture[str], toy: dict[str, list[list[float]]]
) -> None:
    runpy.run_path(str(ANIMATIONS / "mhc_toy.py"), run_name="__main__")
    assert json.loads(capsys.readouterr().out) == toy


def test_html_numbers_match_the_animation_calculation(
    toy: dict[str, list[list[float]]],
) -> None:
    html = LESSON.read_text(encoding="utf-8")
    blocks = re.findall(
        r'<(table|pre)[^>]*data-trace="([^"]+)"[^>]*>(.*?)</\1>', html, flags=re.DOTALL
    )
    assert {key for _, key, _ in blocks} == set(toy)
    for tag, key, content in blocks:
        pattern = r"<td[^>]*>(-?\d+\.\d+)</td>" if tag == "table" else r"-?\d+\.\d+"
        displayed = [float(value) for value in re.findall(pattern, content)]
        calculated = [value for row in toy[key] for value in row]
        assert displayed == pytest.approx(calculated, abs=0.0005), key


def test_chapter_buttons_match_rendered_scene_slots() -> None:
    tree = ast.parse((ANIMATIONS / "mhc_walkthrough.py").read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"CHAPTER_SECONDS", "CHAPTERS"}
    }
    html = LESSON.read_text(encoding="utf-8")
    starts = [int(start) for start in re.findall(r'data-start="(\d+)"', html)]
    assert starts == [
        index * constants["CHAPTER_SECONDS"] for index in range(len(constants["CHAPTERS"]))
    ]
    written_steps = [int(step) for step in re.findall(r'data-step="(\d+)"', html)]
    assert written_steps == list(range(1, len(constants["CHAPTERS"]) + 1))
    pause_duration = re.search(r"playFrom\(start, start \+ (\d+)\)", html)
    assert pause_duration is not None
    assert int(pause_duration[1]) == constants["CHAPTER_SECONDS"]
    assert constants["CHAPTER_SECONDS"] == 40
