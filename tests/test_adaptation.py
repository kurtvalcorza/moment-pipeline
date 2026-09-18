"""Offline tests for the labelled-window dataset contract, the pinned HAPT archive reader, the seeded
user-level split, the canonical windowing bridge, classification metrics with the majority and k-NN
baselines, BYOD loaders, adapt() argument guards, and the full probe / unfreeze / artifact path against a
tiny stand-in encoder. No momentfm import and no weights."""

# ruff: noqa: E501  -- assertion lines are kept on one line

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from moment_pipeline import (
    ARTIFACT_FORMAT,
    CHANNELS,
    ENCODER_BLOCKS,
    POLICY_FROZEN,
    SAMPLE_SPLIT,
    WINDOW_LENGTH,
    ClassifierAdapter,
    adapt,
    build_sample_dataset,
    check_split_disjoint,
    class_names,
    classification_metrics,
    classify,
    dataset_digest,
    evaluate,
    features,
    fetch_sample_dataset,
    knn_baseline,
    knn_predict,
    load_artifact,
    load_byod_dataset,
    majority_baseline,
    read_corpus,
    records_to_long_frame,
    save_artifact,
    split_dataset,
    user_summary,
    validate_dataset,
    window_digest,
    windows_for,
    write_dataset_csv,
)
from moment_pipeline import adaptation as ad
from moment_pipeline import samples as sm
from moment_pipeline.model import LoadedMoment, ModelIdentity
from moment_pipeline.samples import fetch_corpus

# --- synthetic windows and a stand-in encoder ------------------------------------------------------


def _window(label: str, seed: int, channels: int = 6) -> np.ndarray:
    """Six-channel 512-sample windows whose class is a frequency: a stand-in encoder can separate them."""
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW_LENGTH) / 50.0
    freq = {"a": 0.5, "b": 1.5, "c": 3.0}[label]
    base = np.sin(2 * np.pi * freq * t)
    return np.stack(
        [base * (1 + 0.1 * c) + rng.normal(0, 0.05, WINDOW_LENGTH) for c in range(channels)]
    ).astype(np.float32)


def _records(n=12):
    labels = ["a", "b", "c"]
    return [
        {
            "id": f"r{i:02d}",
            "x": _window(labels[i % 3], i),
            "label": labels[i % 3],
            "user": f"user{i // 3:02d}",
        }
        for i in range(n)
    ]


class _StandIn(torch.nn.Module):
    """A tiny encoder with MOMENT's parameter naming (`encoder.block.<k>.*`) and `embed` signature."""

    def __init__(self, d_model: int = 16):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.block = torch.nn.ModuleList(
            [torch.nn.Linear(WINDOW_LENGTH, d_model) for _ in range(ENCODER_BLOCKS)]
        )
        self.head = torch.nn.Identity()

    def embed(self, x_enc, input_mask, reduction="mean"):
        out = self.encoder.block[-1](x_enc.mean(dim=1)) + self.encoder.block[-2](x_enc.std(dim=1))

        class _Result:
            embeddings = out

        return _Result()


def _model(d_model: int = 16, task: str = "embedding") -> LoadedMoment:
    identity = ModelIdentity(
        name="AutonLab/MOMENT-1-base",
        revision="9fea447e740eb968a9e8d80c7562ae122bdb5dde",
        config_sha256="c" * 64,
        weights_sha256="w" * 64,
        weights_bytes=1,
        license="MIT",
        license_basis="test",
        weight_file_loaded="model.safetensors",
        seq_len=WINDOW_LENGTH,
        patch_len=8,
        patch_stride=8,
        d_model_effective=d_model,
        task=task,
        device="cpu",
        dtype="float32",
    )
    torch.manual_seed(0)
    return LoadedMoment(pipeline=_StandIn(d_model), identity=identity, snapshot=None, proof={})


def _archive(users: int = 3, seed: int = 0) -> bytes:
    """A miniature HAPT archive: labels.txt plus acc/gyro files for `users` volunteers, one experiment each."""
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        rows = []
        for user in range(1, users + 1):
            exp = user
            n = 7 * 800
            acc = rng.normal(size=(n, 3)).astype(np.float32)
            gyro = rng.normal(size=(n, 3)).astype(np.float32)
            for activity in range(1, 7):
                start, end = (activity - 1) * 800 + 10, activity * 800 - 10
                rows.append(f"{exp} {user} {activity} {start} {end}")
            rows.append(f"{exp} {user} 7 {6 * 800 + 20} {6 * 800 + 100}")  # a transition: skipped
            zf.writestr(
                f"HAPT/RawData/acc_exp{exp:02d}_user{user:02d}.txt",
                "\n".join(" ".join(f"{v:.6f}" for v in r) for r in acc),
            )
            zf.writestr(
                f"HAPT/RawData/gyro_exp{exp:02d}_user{user:02d}.txt",
                "\n".join(" ".join(f"{v:.6f}" for v in r) for r in gyro),
            )
        zf.writestr("HAPT/RawData/labels.txt", "\n".join(rows))
    return buffer.getvalue()


def _pin(monkeypatch, payload: bytes):
    monkeypatch.setattr(sm, "CORPUS_BYTES", len(payload))
    monkeypatch.setattr(sm, "CORPUS_SHA256", hashlib.sha256(payload).hexdigest())


# --- pinned archive and reader ---------------------------------------------------------------------


def test_pinned_corpus_constants_are_complete():
    assert sm.CORPUS_URL.startswith("https://archive.ics.uci.edu/static/public/341/")
    assert len(sm.CORPUS_SHA256) == 64 and sm.CORPUS_BYTES == 79_596_192
    assert set(sm.ACTIVITIES.values()) == {
        "walking",
        "walking_upstairs",
        "walking_downstairs",
        "sitting",
        "standing",
        "laying",
    }
    assert (
        CHANNELS == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
        and WINDOW_LENGTH == 512
    )
    assert sum(SAMPLE_SPLIT.values()) == sm.N_USERS and set(SAMPLE_SPLIT) == {
        "train",
        "validation",
        "test",
    }


def test_fetch_corpus_verifies_the_archive_and_caches(tmp_path, monkeypatch):
    payload = _archive()
    _pin(monkeypatch, payload)
    calls = []

    def fetcher(url):
        calls.append(url)
        return payload

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == payload
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == payload and calls == [sm.CORPUS_URL]
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_cuts_the_centre_window_of_the_first_long_segment(monkeypatch):
    records = read_corpus(_archive(users=2))
    assert (
        len(records) == 12 and records[0]["id"] == "u01-walking" and records[0]["user"] == "user01"
    )
    assert records[0]["x"].shape == (6, WINDOW_LENGTH) and records[0]["x"].dtype == np.float32
    assert (
        records[0]["segment"] == [10, 790] and records[0]["first_sample"] == (10 + 790) // 2 - 256
    )
    assert {r["label"] for r in records} == set(sm.ACTIVITIES.values())
    with pytest.raises(zipfile.BadZipFile):
        read_corpus(_archive(users=1)[:100] + b"x")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("RawData/labels.txt", "1 1 1 10 20 30")  # six columns
    with pytest.raises(ValueError, match="labels.txt"):
        read_corpus(buffer.getvalue())


def test_sample_split_is_by_user_seeded_and_disjoint(monkeypatch):
    records = read_corpus(_archive(users=4))
    sizes = {"train": 2, "validation": 1, "test": 1}
    splits = build_sample_dataset(records, seed=42, sizes=sizes)
    assert check_split_disjoint(splits) == {"train": 12, "validation": 6, "test": 6}
    summary = user_summary(splits)
    assert summary["train"]["users"] == 2 and summary["test"]["label_counts"]["walking"] == 1
    assert splits["train"][0]["id"] == "train-000" and splits["train"][0]["source_id"].startswith(
        "u"
    )
    assert (
        build_sample_dataset(records, seed=42, sizes=sizes)["test"][0]["source_id"]
        == splits["test"][0]["source_id"]
    )
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(records, sizes={"train": 4, "validation": 1, "test": 1})
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": splits["train"], "test": [splits["train"][0]]})
    other = next(r for r in splits["train"][1:] if r["user"] == splits["train"][0]["user"])
    with pytest.raises(ValueError, match="windows in both"):
        check_split_disjoint({"train": [splits["train"][0]], "test": [other]})


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch):
    payload = _archive(users=4)
    _pin(monkeypatch, payload)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda url: payload,
        sizes={"train": 2, "validation": 1, "test": 1},
    )
    manifests = {name: validate_dataset(part, min_records=1) for name, part in splits.items()}
    assert (
        manifests["train"]["n_records"] == 12
        and manifests["train"]["users"] == 2
        and manifests["train"]["n_channels"] == 6
    )
    assert len({m["digest"] for m in manifests.values()}) == 3


# --- dataset contract and the canonical bridge -------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path):
    records = _records()
    report = validate_dataset(records)
    assert (
        report["n_records"] == 12 and report["classes"] == ["a", "b", "c"] and report["users"] == 4
    )
    assert (
        report["n_channels"] == 6
        and report["window_length"] == WINDOW_LENGTH
        and report["digest"] == dataset_digest(records)
    )
    np.save(tmp_path / "w.npy", records[0]["x"])
    from_path = validate_dataset([{"id": "p", "x": tmp_path / "w.npy", "label": "a"}, *records[1:]])
    assert from_path["records"][0]["x"].shape == (6, WINDOW_LENGTH)
    one_channel = validate_dataset([{**r, "x": r["x"][0]} for r in records])
    assert one_channel["n_channels"] == 1
    bad = [
        ({**records[0], "id": "bad id"}, "id must match"),
        ({**records[0], "x": "nope.npy"}, "x file not found"),
        ({**records[0], "x": records[0]["x"][:, :-1]}, "channels"),
        ({**records[0], "x": records[0]["x"] * np.nan}, "finite"),
        ({**records[0], "x": records[0]["x"] * 1e6}, "finite"),
        ({**records[0], "label": ""}, "label must be"),
        ({"id": "x", "x": records[0]["x"]}, "missing 'label'"),
    ]
    for record, message in bad:
        with pytest.raises(ValueError, match=message):
            validate_dataset([record, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], records[0], *records[2:]])
    with pytest.raises(ValueError, match="same channel count"):
        validate_dataset([{**records[0], "x": records[0]["x"][:3]}, *records[1:]])
    with pytest.raises(ValueError, match="distinct labels"):
        validate_dataset([{**r, "label": "a"} for r in records])
    with pytest.raises(ValueError, match="8..1024"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_windows_for_goes_through_the_canonical_path_in_id_order():
    records = _records(9)[::-1]  # reversed input order
    windows, ordered = windows_for(records)
    assert windows.x_enc.shape == (9, 6, WINDOW_LENGTH) and windows.channels == CHANNELS
    assert (
        [r["id"] for r in ordered] == list(windows.series_ids) == sorted(r["id"] for r in records)
    )
    assert windows.padded_fraction == 0.0 and windows.masked_point_fraction == 0.0
    assert np.allclose(windows.x_enc[0], ordered[0]["x"], atol=1e-6)
    frame = records_to_long_frame(records[:1])
    assert (
        list(frame.columns) == ["series_id", "timestamp", "channel", "value"]
        and len(frame) == 6 * WINDOW_LENGTH
    )


def test_split_dataset_groups_by_user_deduplicates_and_is_seeded():
    records = _records(30)
    duplicated = [*records, {**records[0], "id": "dup", "user": "other"}]
    splits = split_dataset(duplicated, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert sum(len(part) for part in splits.values()) == 30
    check_split_disjoint(splits)
    assert {
        r["id"]
        for r in split_dataset(duplicated, val_fraction=0.2, test_fraction=0.2, seed=1)["test"]
    } == {r["id"] for r in splits["test"]}
    assert len(splits["test"]) % 3 == 0  # whole users of three windows
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)


# --- metrics and baselines --------------------------------------------------------------------------


def test_classification_metrics_and_majority_baseline():
    metrics = classification_metrics(["a", "a", "b", "c"], ["a", "b", "b", "c"], ["a", "b", "c"])
    assert metrics["accuracy"] == 0.75 and metrics["confusion"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert (
        metrics["per_class"]["a"]["recall"] == 0.5 and metrics["per_class"]["b"]["precision"] == 0.5
    )
    assert abs(metrics["macro_f1"] - (2 / 3 + 2 / 3 + 1) / 3) < 1e-9
    floor = majority_baseline(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert floor["accuracy"] == pytest.approx(1 / 3) and "majority" in floor["baseline"]
    with pytest.raises(ValueError, match="outside"):
        classification_metrics(["z"], ["a"], ["a"])
    with pytest.raises(ValueError, match="gold labels"):
        classification_metrics(["a"], [], ["a"])


def test_knn_predict_votes_by_cosine():
    train = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    assert knn_predict(
        train, ["x", "x", "y", "y"], np.array([[1, 0.05], [0.05, 1]], dtype=np.float32), k=3
    ) == ["x", "y"]
    with pytest.raises(ValueError, match="k must be"):
        knn_predict(train, ["x"] * 4, train, k=5)


# --- BYOD ------------------------------------------------------------------------------------------


def test_byod_directory_and_zip_round_trip_and_rejections(tmp_path):
    records = _records(9)
    folder = tmp_path / "byod"
    folder.mkdir()
    with open(folder / "records.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "file", "label", "group"])
        writer.writeheader()
        for record in records:
            np.save(folder / f"{record['id']}.npy", record["x"])
            writer.writerow(
                {
                    "id": record["id"],
                    "file": f"{record['id']}.npy",
                    "label": record["label"],
                    "group": record["user"],
                }
            )
    loaded = load_byod_dataset(folder)
    assert [r["id"] for r in loaded] == [r["id"] for r in records] and loaded[0][
        "group"
    ] == "user00"
    assert (
        window_digest(loaded[0]["x"]) == window_digest(records[0]["x"])
        and validate_dataset(loaded)["n_records"] == 9
    )
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"inner/{path.name}")
    from_zip = load_byod_dataset(archive)
    assert window_digest(from_zip[1]["x"]) == window_digest(records[1]["x"])
    assert not list(tmp_path.glob("inner"))
    exported = write_dataset_csv(records, tmp_path / "out" / "train.csv")
    rows = list(csv.DictReader(exported.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["group"] == "user00" and rows[0]["file"] == "r00.npy"
    with pytest.raises(ValueError, match="records.csv"):
        load_byod_dataset(tmp_path / "missing")


# --- the adaptation path against the stand-in encoder ------------------------------------------------


def test_features_knn_probe_unfreeze_and_artifact_round_trip(tmp_path):
    model = _model()
    records = _records(24)
    train, test = records[:18], records[18:]
    vectors, ordered = features(train, model)
    assert vectors.shape == (18, 16) and np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5
    )
    baseline = knn_baseline(model, train, test, k=3)
    assert baseline["n"] == 6 and baseline["accuracy"] >= 2 / 3
    adapter = adapt(
        model, train, test, probe_steps=200, trainable_blocks=2, epochs=2, lr=1e-3, batch_size=6
    )
    assert adapter.classes == ["a", "b", "c"] and adapter.history[0]["stage"].startswith(
        "linear probe"
    )
    assert len(adapter.history) == 3 and adapter.config["best_epoch"] in (0, 1, 2)
    assert adapter.config["n_trainable_head"] == 16 * 3 + 3 and adapter.config[
        "n_trainable_blocks"
    ] == 2 * (WINDOW_LENGTH * 16 + 16)
    assert (
        adapter.policy == adapter.history[adapter.config["best_epoch"]]["stage"]
        or adapter.policy == POLICY_FROZEN
    )
    metrics = evaluate(model, adapter, test)
    assert (
        metrics["n"] == 6
        and metrics["accuracy"] >= 2 / 3
        and metrics["adapted"] is True
        and metrics["verdict"] == "measured-small-sample"
    )
    assert metrics["log_loss"] == pytest.approx(
        adapter.history[adapter.config["best_epoch"]]["val"]["log_loss"], abs=1e-4
    )
    labels = classify(model, adapter, test[:3])
    assert (
        labels["classes"] == ["a", "b", "c"]
        and len(labels["probabilities"][0]) == 3
        and labels["ids"] == sorted(r["id"] for r in test[:3])
    )
    artifact = save_artifact(model, adapter, tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == ARTIFACT_FORMAT and "head.weight" in manifest["tensors"]
    assert (len(manifest["tensors"]) > 2) == (adapter.policy != POLICY_FROZEN)
    fresh = _model()
    reloaded = load_artifact(fresh, artifact)
    assert (
        isinstance(reloaded, ClassifierAdapter)
        and reloaded.classes == adapter.classes
        and reloaded.policy == adapter.policy
    )
    assert classify(fresh, reloaded, test[:3])["probabilities"] == labels["probabilities"]


def test_adapt_and_artifacts_guard_their_arguments(tmp_path):
    model = _model()
    for kwargs, message in [
        ({"probe_steps": 0}, "probe_steps"),
        ({"probe_lr": 2.0}, "probe_lr"),
        ({"epochs": -1}, "epochs"),
        ({"lr": 1.0}, "lr"),
        ({"batch_size": 0}, "batch_size"),
        ({"trainable_blocks": ENCODER_BLOCKS + 1}, "trainable_blocks"),
    ]:
        with pytest.raises(ValueError, match=message):
            adapt(model, _records(), **kwargs)
    with pytest.raises(ValueError, match="task='embedding'"):
        adapt(_model(task="reconstruction"), _records())
    with pytest.raises(ValueError, match="distinct labels"):
        adapt(model, [{**r, "label": "a"} for r in _records()])



def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    model = _model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {
            "id": model.identity.name,
            "revision": model.identity.revision,
            "weight_sha256": model.identity.weights_sha256,
        },
        "files": [{"path": ad.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["encoder.block.11.weight", "head.bias", "head.weight"],
        "adapter": {"classes": ["a", "b"], "policy": POLICY_FROZEN},
    }
    (tmp_path / ad.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        load_artifact(model, tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / ad.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        load_artifact(model, tmp_path)
    (tmp_path / ad.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        load_artifact(model, tmp_path)
    (tmp_path / ad.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        load_artifact(model, tmp_path)
    assert class_names(_records()) == ["a", "b", "c"]
