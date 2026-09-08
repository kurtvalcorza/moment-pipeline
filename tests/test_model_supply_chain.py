"""Loader supply-chain rules. No network: every download is mocked or refused first."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path

import pytest

from moment_pipeline import model as model_module
from moment_pipeline.model import (
    ALLOW_PATTERNS,
    KNOWN_FORBIDDEN_WEIGHT_FILE,
    PINNED_CONFIG_SHA256,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    PINNED_WEIGHTS_BYTES,
    PINNED_WEIGHTS_FILENAME,
    PINNED_WEIGHTS_SHA256,
    IntegrityError,
    ModelSourceError,
    assert_pinned_source,
    fetch_verified_snapshot,
    find_forbidden_weight_files,
    load_moment,
    verify_snapshot_dir,
)

CONFIG_BODY = json.dumps({"seq_len": 512, "patch_len": 8, "patch_stride_len": 8}).encode()
WEIGHTS_BODY = b"safetensors-stand-in-payload"
CONFIG_SHA = hashlib.sha256(CONFIG_BODY).hexdigest()
WEIGHTS_SHA = hashlib.sha256(WEIGHTS_BODY).hexdigest()


def build_snapshot(tmp_path: Path, name: str = PINNED_REVISION, weights: bytes = WEIGHTS_BODY):
    root = tmp_path / name
    root.mkdir()
    (root / "config.json").write_bytes(CONFIG_BODY)
    (root / PINNED_WEIGHTS_FILENAME).write_bytes(weights)
    return root


def verify(root: Path, **overrides):
    kwargs = {
        "expected_config_sha256": CONFIG_SHA,
        "expected_weights_sha256": WEIGHTS_SHA,
        "expected_weights_bytes": len(WEIGHTS_BODY),
    }
    kwargs.update(overrides)
    return verify_snapshot_dir(root, **kwargs)


# --- the constants themselves are part of the contract ------------------------------


def test_pinned_constants_match_the_rfc():
    assert PINNED_MODEL_ID == "AutonLab/MOMENT-1-base"
    assert PINNED_REVISION == "9fea447e740eb968a9e8d80c7562ae122bdb5dde"
    assert PINNED_WEIGHTS_SHA256 == (
        "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825"
    )
    assert PINNED_WEIGHTS_BYTES == 453_940_120
    assert PINNED_CONFIG_SHA256 == (
        "f1c66c2bb845229c0ed27a1600dbcc956b85ab21f9e5fd8a1663e6641bed7755"
    )


def test_allow_patterns_cannot_pull_the_pickle_weights():
    assert not any(fnmatch.fnmatch(KNOWN_FORBIDDEN_WEIGHT_FILE, p) for p in ALLOW_PATTERNS)
    assert PINNED_WEIGHTS_FILENAME in ALLOW_PATTERNS


# --- source rejection matrix (no network) -------------------------------------------


def test_pinned_source_accepted():
    assert_pinned_source(PINNED_MODEL_ID, PINNED_REVISION)  # must not raise


@pytest.mark.parametrize(
    "model_id,revision",
    [
        (PINNED_MODEL_ID, "main"),
        (PINNED_MODEL_ID, "latest"),
        (PINNED_MODEL_ID, "v1"),
        (PINNED_MODEL_ID, "0000000000000000000000000000000000000000"),
        (PINNED_MODEL_ID, PINNED_REVISION.upper()),
        ("AutonLab/MOMENT-1-large", PINNED_REVISION),
        ("autonlab/moment-1-base", PINNED_REVISION),
        ("s3://my-bucket/moment", PINNED_REVISION),
        ("https://huggingface.co/AutonLab/MOMENT-1-base", PINNED_REVISION),
        ("./local-copy", PINNED_REVISION),
        ("/opt/models/moment", PINNED_REVISION),
        ("C:\\models\\moment", PINNED_REVISION),
        ("~/moment", PINNED_REVISION),
    ],
)
def test_unapproved_sources_rejected(model_id, revision):
    with pytest.raises(ModelSourceError):
        assert_pinned_source(model_id, revision)


def test_rejection_happens_before_any_download(monkeypatch):
    def explode(**kwargs):  # pragma: no cover - must never run
        raise AssertionError("snapshot_download must not be reached for a rejected source")

    monkeypatch.setattr(model_module, "snapshot_download", explode)
    with pytest.raises(ModelSourceError):
        fetch_verified_snapshot(revision="main")


def test_download_is_restricted_to_the_allow_patterns(monkeypatch, tmp_path):
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(build_snapshot(tmp_path))

    monkeypatch.setattr(model_module, "snapshot_download", fake_download)
    with pytest.raises(IntegrityError) as excinfo:
        fetch_verified_snapshot()
    # It got as far as the real digest check, which the stand-in payload cannot satisfy.
    assert excinfo.value.code == "CONFIG_DIGEST_MISMATCH"
    assert captured["allow_patterns"] == ALLOW_PATTERNS
    assert captured["revision"] == PINNED_REVISION
    assert captured["repo_id"] == PINNED_MODEL_ID


# --- snapshot verification ----------------------------------------------------------


def test_well_formed_snapshot_verifies(tmp_path):
    root = build_snapshot(tmp_path)
    config_sha, config_bytes, weights_sha, weights_bytes, config = verify(root)
    assert config_sha == CONFIG_SHA
    assert weights_sha == WEIGHTS_SHA
    assert config_bytes == len(CONFIG_BODY)
    assert weights_bytes == len(WEIGHTS_BODY)
    assert config["seq_len"] == 512


def test_bin_in_snapshot_refused_before_anything_else(tmp_path):
    root = build_snapshot(tmp_path)
    (root / KNOWN_FORBIDDEN_WEIGHT_FILE).write_bytes(b"pickle")
    with pytest.raises(IntegrityError) as excinfo:
        # Even with every expectation wrong, the .bin is what stops it.
        verify(root, expected_config_sha256="deadbeef", expected_weights_sha256="deadbeef")
    assert excinfo.value.code == "FORBIDDEN_WEIGHT_FILE"
    assert excinfo.value.details["files"] == [KNOWN_FORBIDDEN_WEIGHT_FILE]


def test_nested_bin_is_found_too(tmp_path):
    root = build_snapshot(tmp_path)
    (root / "extra").mkdir()
    (root / "extra" / "adapter.bin").write_bytes(b"pickle")
    assert find_forbidden_weight_files(root) == [str(Path("extra") / "adapter.bin")]


def test_wrong_revision_directory_refused(tmp_path):
    root = build_snapshot(tmp_path, name="0" * 40)
    with pytest.raises(IntegrityError) as excinfo:
        verify(root)
    assert excinfo.value.code == "REVISION_MISMATCH"


def test_missing_weight_file_refused(tmp_path):
    root = build_snapshot(tmp_path)
    (root / PINNED_WEIGHTS_FILENAME).unlink()
    with pytest.raises(IntegrityError) as excinfo:
        verify(root)
    assert excinfo.value.code == "SNAPSHOT_INCOMPLETE"


def test_config_digest_mismatch_refused(tmp_path):
    root = build_snapshot(tmp_path)
    (root / "config.json").write_bytes(CONFIG_BODY + b" ")
    with pytest.raises(IntegrityError) as excinfo:
        verify(root)
    assert excinfo.value.code == "CONFIG_DIGEST_MISMATCH"


def test_weights_size_mismatch_refused(tmp_path):
    root = build_snapshot(tmp_path, weights=WEIGHTS_BODY + b"!")
    with pytest.raises(IntegrityError) as excinfo:
        verify(root)
    assert excinfo.value.code == "WEIGHTS_SIZE_MISMATCH"


def test_weights_digest_mismatch_refused_at_identical_size(tmp_path):
    tampered = bytearray(WEIGHTS_BODY)
    tampered[0] ^= 0xFF
    root = build_snapshot(tmp_path, weights=bytes(tampered))
    with pytest.raises(IntegrityError) as excinfo:
        verify(root)
    assert excinfo.value.code == "WEIGHTS_DIGEST_MISMATCH"
    assert excinfo.value.details["expected"] == WEIGHTS_SHA


def test_config_shape_mismatch_refused(monkeypatch, tmp_path):
    """A digest-clean snapshot whose config declares a different geometry is still refused."""
    root = build_snapshot(tmp_path)
    monkeypatch.setattr(model_module, "snapshot_download", lambda **kw: str(root))
    monkeypatch.setattr(
        model_module,
        "verify_snapshot_dir",
        lambda directory, **kw: (
            CONFIG_SHA,
            len(CONFIG_BODY),
            WEIGHTS_SHA,
            len(WEIGHTS_BODY),
            {"seq_len": 1024, "patch_len": 8, "patch_stride_len": 8},
        ),
    )
    with pytest.raises(IntegrityError) as excinfo:
        fetch_verified_snapshot()
    assert excinfo.value.code == "CONFIG_SHAPE_MISMATCH"
    assert excinfo.value.details["seq_len"] == 1024


def test_config_shape_accepted_when_it_matches(monkeypatch, tmp_path):
    root = build_snapshot(tmp_path)
    monkeypatch.setattr(model_module, "snapshot_download", lambda **kw: str(root))
    monkeypatch.setattr(
        model_module,
        "verify_snapshot_dir",
        lambda directory, **kw: (
            CONFIG_SHA,
            len(CONFIG_BODY),
            WEIGHTS_SHA,
            len(WEIGHTS_BODY),
            {"seq_len": 512, "patch_len": 8, "patch_stride_len": 8, "task_name": "reconstruction"},
        ),
    )
    snapshot = fetch_verified_snapshot()
    assert snapshot.revision == PINNED_REVISION
    assert snapshot.task_name_in_config == "reconstruction"
    assert snapshot.weights_path.name == PINNED_WEIGHTS_FILENAME


# --- task scope ---------------------------------------------------------------------


@pytest.mark.parametrize("task", ["forecasting", "classification", "anomaly", ""])
def test_v1_refuses_non_v1_tasks_without_touching_the_network(monkeypatch, task):
    def explode(**kwargs):  # pragma: no cover - must never run
        raise AssertionError("no download for a refused task")

    monkeypatch.setattr(model_module, "snapshot_download", explode)
    with pytest.raises(ModelSourceError) as excinfo:
        load_moment(task=task)
    assert "v1 exposes only" in str(excinfo.value)
