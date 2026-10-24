import hashlib
import json
import runpy
import tomllib
from pathlib import Path

import pytest

from glm_flash_drafter.references import verify_references

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    reference = tmp_path / "reference"
    reference.mkdir()
    content = b"pinned upstream source\n"
    (reference / "source.txt").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    (reference / "SHA256SUMS").write_text(f"{digest}  reference/source.txt\n", encoding="utf-8")
    return tmp_path


def test_pinned_sources_are_intact() -> None:
    assert verify_references(ROOT) == []


def test_valid_snapshot(snapshot: Path) -> None:
    assert verify_references(snapshot) == []


def test_reports_missing_reference(snapshot: Path) -> None:
    (snapshot / "reference/source.txt").unlink()
    assert verify_references(snapshot) == ["reference/source.txt: missing"]


def test_reports_modified_reference(snapshot: Path) -> None:
    (snapshot / "reference/source.txt").write_text("changed", encoding="utf-8")
    assert verify_references(snapshot) == ["reference/source.txt: checksum mismatch"]


def test_empty_checksum_list_fails(snapshot: Path) -> None:
    (snapshot / "reference/SHA256SUMS").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum list is empty"):
        verify_references(snapshot)


def test_malformed_checksum_list_fails(snapshot: Path) -> None:
    (snapshot / "reference/SHA256SUMS").write_text("malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not enough values"):
        verify_references(snapshot)


def test_cli_success(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(snapshot)
    runpy.run_module("glm_flash_drafter", run_name="__main__")
    assert capsys.readouterr().out == "Pinned reference checksums verified.\n"


def test_cli_failure(snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (snapshot / "reference/source.txt").unlink()
    monkeypatch.chdir(snapshot)
    with pytest.raises(SystemExit, match=r"reference/source\.txt: missing"):
        runpy.run_module("glm_flash_drafter", run_name="__main__")


def test_model_snapshot_matches_target_contract() -> None:
    config = json.loads((ROOT / "reference/glm-5.3-flash/config.json").read_text())
    manifest = tomllib.loads((ROOT / "UPSTREAMS.toml").read_text())
    assert manifest["model"]["repo_id"] == "zai-org/GLM-5.3-Flash"
    assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
    assert config["model_type"] == "glm5_next"
    assert config["text_config"]["hidden_size"] == 4096
    assert config["text_config"]["vocab_size"] == 154880
