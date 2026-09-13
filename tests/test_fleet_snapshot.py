"""Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): manifest-driven staging and verification.

Nothing here touches the network or the real weights: the snapshot directory is built from
stand-in bytes, the manifest is written by hand, and downloads go through an injected callable.
"""

# ruff: noqa: E501  -- offline fixtures and assertions are kept on single lines for readability
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from moment_pipeline import model as model_mod
from moment_pipeline.model import (
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    PINNED_WEIGHTS_FILENAME,
    IntegrityError,
    sha256_file,
    stage_missing_files,
    verify_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
OTHER_SHA = "0" * 40
CONFIG_JSON = json.dumps(
    {"seq_len": 512, "patch_len": 8, "patch_stride_len": 8, "task_name": "reconstruction", "d_model": 768}
).encode("utf-8")


def write_manifest(root: Path, files: dict[str, bytes], **overrides) -> dict:
    entries = []
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append({"path": name, "bytes": len(payload), "sha256": sha256_file(path)})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
        **overrides,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def stand_in_snapshot(root: Path) -> dict:
    return write_manifest(
        root, {"README.md": b"# stand-in\n", "config.json": CONFIG_JSON, PINNED_WEIGHTS_FILENAME: b"safetensors-stand-in"}
    )


@pytest.fixture
def pinned_to_stand_in(monkeypatch, tmp_path):
    """Point the PINNED_* digest constants at the stand-in bytes so the manifest path verifies."""
    stand_in_snapshot(tmp_path)
    monkeypatch.setattr(model_mod, "PINNED_CONFIG_SHA256", sha256_file(tmp_path / "config.json"))
    monkeypatch.setattr(model_mod, "PINNED_WEIGHTS_SHA256", sha256_file(tmp_path / PINNED_WEIGHTS_FILENAME))
    monkeypatch.setattr(model_mod, "PINNED_WEIGHTS_BYTES", (tmp_path / PINNED_WEIGHTS_FILENAME).stat().st_size)
    # `verify_snapshot_dir` binds the pinned digests as keyword defaults at definition time, so the
    # module-level patches above do not reach callers that rely on those defaults; route them through.
    original_verify_dir = model_mod.verify_snapshot_dir

    def verify_dir_with_patched_pins(directory, **kwargs):
        kwargs.setdefault("expected_config_sha256", model_mod.PINNED_CONFIG_SHA256)
        kwargs.setdefault("expected_weights_sha256", model_mod.PINNED_WEIGHTS_SHA256)
        kwargs.setdefault("expected_weights_bytes", model_mod.PINNED_WEIGHTS_BYTES)
        return original_verify_dir(directory, **kwargs)

    monkeypatch.setattr(model_mod, "verify_snapshot_dir", verify_dir_with_patched_pins)
    return tmp_path


# --------------------------------------------------------------------------
# Identity constants and the committed manifest
# --------------------------------------------------------------------------


def test_fleet_identity_names_alias_the_package_pins():
    assert (MODEL_ID, MODEL_REVISION) == (PINNED_MODEL_ID, PINNED_REVISION)
    assert MODEL_LICENSE == "MIT"
    assert MODEL_KEY == "moment-1-base"
    assert model_mod.DEFAULT_WEIGHTS_DIR == ROOT / "weights" / MODEL_KEY


def test_committed_manifest_names_the_pinned_identity_and_digests():
    manifest = json.loads((ROOT / "weights" / MODEL_KEY / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    assert by_path[PINNED_WEIGHTS_FILENAME]["sha256"] == model_mod.PINNED_WEIGHTS_SHA256
    assert by_path[PINNED_WEIGHTS_FILENAME]["bytes"] == model_mod.PINNED_WEIGHTS_BYTES
    assert by_path["config.json"]["sha256"] == model_mod.PINNED_CONFIG_SHA256
    assert by_path["config.json"]["bytes"] == model_mod.PINNED_CONFIG_BYTES
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert sorted(by_path) == sorted(model_mod.ALLOW_PATTERNS)


# --------------------------------------------------------------------------
# stage_missing_files
# --------------------------------------------------------------------------


def test_stage_fetches_only_the_absent_entries_through_the_injected_downloader(tmp_path):
    stand_in_snapshot(tmp_path)
    (tmp_path / PINNED_WEIGHTS_FILENAME).unlink()
    (tmp_path / "README.md").unlink()
    calls: list[tuple[str, Path]] = []

    def downloader(relative_path: str, root: Path) -> None:
        calls.append((relative_path, root))
        (root / relative_path).write_bytes(b"safetensors-stand-in" if relative_path == PINNED_WEIGHTS_FILENAME else b"# stand-in\n")

    fetched = stage_missing_files(tmp_path, allow_download=True, downloader=downloader)
    assert sorted(fetched) == sorted(["README.md", PINNED_WEIGHTS_FILENAME])
    assert [c[0] for c in calls] == fetched and all(c[1] == tmp_path for c in calls)
    assert stage_missing_files(tmp_path, allow_download=False, downloader=downloader) == []
    assert len(calls) == 2


def test_stage_refuses_to_download_by_default(tmp_path):
    stand_in_snapshot(tmp_path)
    (tmp_path / PINNED_WEIGHTS_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)


@pytest.mark.parametrize("override", [{"modelId": "someone/else"}, {"revision": OTHER_SHA}])
def test_stage_refuses_a_manifest_with_the_wrong_identity(tmp_path, override):
    write_manifest(tmp_path, {"config.json": b"{}", PINNED_WEIGHTS_FILENAME: b"w"}, **override)
    with pytest.raises(IntegrityError) as exc:
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)
    assert exc.value.code == "MANIFEST_IDENTITY_MISMATCH"


def test_stage_refuses_a_directory_without_a_manifest(tmp_path):
    with pytest.raises(IntegrityError) as exc:
        stage_missing_files(tmp_path)
    assert exc.value.code == "MANIFEST_MISSING"


def test_the_default_downloader_fetches_at_the_pinned_revision(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_hf_hub_download(repo_id, filename, *, revision, local_dir):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision, "local_dir": local_dir})

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    model_mod._hub_download(PINNED_WEIGHTS_FILENAME, tmp_path)
    assert calls == [{"repo_id": MODEL_ID, "filename": PINNED_WEIGHTS_FILENAME, "revision": MODEL_REVISION, "local_dir": str(tmp_path)}]


# --------------------------------------------------------------------------
# verify_snapshot on a manifest-described directory
# --------------------------------------------------------------------------


def test_a_manifest_snapshot_verifies_and_reports_the_manifest(pinned_to_stand_in):
    result = verify_snapshot(pinned_to_stand_in)
    assert result["revision"] == MODEL_REVISION
    assert result["modelKey"] == MODEL_KEY
    assert [entry["path"] for entry in result["files"]] == ["README.md", "config.json", PINNED_WEIGHTS_FILENAME]
    assert result["path"] == str(pinned_to_stand_in)


def test_a_tampered_manifest_digest_is_refused(pinned_to_stand_in):
    manifest = json.loads((pinned_to_stand_in / MANIFEST_NAME).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == PINNED_WEIGHTS_FILENAME:
            entry["sha256"] = "f" * 64
    (pinned_to_stand_in / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IntegrityError) as exc:
        verify_snapshot(pinned_to_stand_in)
    assert exc.value.code == "MANIFEST_DIGEST_MISMATCH"


def test_a_tampered_file_of_the_same_size_is_refused_against_the_manifest(pinned_to_stand_in):
    (pinned_to_stand_in / PINNED_WEIGHTS_FILENAME).write_bytes(b"Safetensors-stand-in")
    with pytest.raises(IntegrityError) as exc:
        verify_snapshot(pinned_to_stand_in)
    assert exc.value.code == "MANIFEST_DIGEST_MISMATCH"


def test_a_manifest_that_disagrees_with_the_pinned_digest_constants_is_refused(pinned_to_stand_in, monkeypatch):
    """The package's own PINNED_* digests are asserted equal to the manifest, never replaced."""
    monkeypatch.setattr(model_mod, "PINNED_WEIGHTS_SHA256", "e" * 64)
    with pytest.raises(IntegrityError) as exc:
        verify_snapshot(pinned_to_stand_in)
    assert exc.value.code == "WEIGHTS_DIGEST_MISMATCH"


def test_a_manifest_snapshot_with_the_wrong_identity_is_refused(tmp_path):
    write_manifest(tmp_path, {"config.json": b"{}", PINNED_WEIGHTS_FILENAME: b"w"}, revision=OTHER_SHA)
    with pytest.raises(IntegrityError) as exc:
        verify_snapshot(tmp_path)
    assert exc.value.code == "MANIFEST_IDENTITY_MISMATCH"


def test_a_missing_manifest_entry_is_refused(pinned_to_stand_in):
    (pinned_to_stand_in / "README.md").unlink()
    with pytest.raises(IntegrityError) as exc:
        verify_snapshot(pinned_to_stand_in)
    assert exc.value.code == "SNAPSHOT_INCOMPLETE"


def test_pickle_weights_next_to_a_manifest_are_refused(pinned_to_stand_in):
    (pinned_to_stand_in / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(IntegrityError) as exc:
        verify_snapshot(pinned_to_stand_in)
    assert exc.value.code == "FORBIDDEN_WEIGHT_FILE"


def test_verify_snapshot_dir_accepts_a_manifest_directory_named_by_key(pinned_to_stand_in):
    """The directory is `<tmp>/`, not `<sha>/`: the manifest carries the revision instead."""
    config_sha, _config_bytes, weights_sha, weights_bytes, config = model_mod.verify_snapshot_dir(
        pinned_to_stand_in,
        expected_config_sha256=model_mod.PINNED_CONFIG_SHA256,
        expected_weights_sha256=model_mod.PINNED_WEIGHTS_SHA256,
        expected_weights_bytes=model_mod.PINNED_WEIGHTS_BYTES,
    )
    assert (config_sha, weights_sha, weights_bytes) == (model_mod.PINNED_CONFIG_SHA256, model_mod.PINNED_WEIGHTS_SHA256, model_mod.PINNED_WEIGHTS_BYTES)
    assert config["seq_len"] == 512
    snapshot = model_mod.verified_snapshot_from_dir(pinned_to_stand_in)
    assert snapshot.revision == MODEL_REVISION
    assert snapshot.path == pinned_to_stand_in


# --------------------------------------------------------------------------
# load_moment(weights_dir=...) stages, verifies, then loads from that directory
# --------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self):
        self.config = types.SimpleNamespace(d_model=768)
        self.head = types.SimpleNamespace(__class__=type("PretrainHead", (), {}))

    def init(self):
        pass

    def eval(self):
        return self

    def to(self, **kwargs):
        return self


def test_load_moment_from_weights_dir_never_calls_snapshot_download(monkeypatch, pinned_to_stand_in):
    loaded_from: list[str] = []

    class FakeMomentPipeline:
        @staticmethod
        def from_pretrained(path, model_kwargs=None):
            loaded_from.append(path)
            return _FakePipeline()

    monkeypatch.setitem(sys.modules, "momentfm", types.SimpleNamespace(MOMENTPipeline=FakeMomentPipeline))

    def explode(*args, **kwargs):
        raise AssertionError("snapshot_download must not run on the weights_dir path")

    monkeypatch.setattr(model_mod, "snapshot_download", explode)
    monkeypatch.setattr(model_mod, "prove_pinned_weights_are_live", lambda pipeline, weights_path, task: {"proof": "stand-in"})
    loaded = model_mod.load_moment(task="reconstruction", device="cpu", weights_dir=pinned_to_stand_in)
    assert loaded_from == [str(pinned_to_stand_in)]
    assert loaded.identity.revision == MODEL_REVISION
    assert loaded.snapshot.path == pinned_to_stand_in
    assert loaded.identity.task == "reconstruction"


def test_load_moment_from_weights_dir_refuses_to_download_by_default(monkeypatch, pinned_to_stand_in):
    (pinned_to_stand_in / PINNED_WEIGHTS_FILENAME).unlink()
    monkeypatch.setitem(sys.modules, "momentfm", types.SimpleNamespace(MOMENTPipeline=object))
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        model_mod.load_moment(task="reconstruction", device="cpu", weights_dir=pinned_to_stand_in)


def test_load_moment_refuses_weights_dir_together_with_a_snapshot(monkeypatch, pinned_to_stand_in):
    monkeypatch.setitem(sys.modules, "momentfm", types.SimpleNamespace(MOMENTPipeline=object))
    snapshot = model_mod.verified_snapshot_from_dir(pinned_to_stand_in)
    with pytest.raises(model_mod.ModelSourceError, match="not both"):
        model_mod.load_moment(task="reconstruction", device="cpu", weights_dir=pinned_to_stand_in, snapshot=snapshot)
