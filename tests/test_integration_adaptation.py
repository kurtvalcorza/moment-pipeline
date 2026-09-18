"""Model-backed checks of the classification adaptation contract against the real pinned
MOMENT-1-base weights (CPU): frozen features and the k-NN baseline on synthetic frequency-coded
windows, a linear probe plus a one-epoch unfreeze of the last block, and the artifact round trip
with head and block tensors.

Marked `integration` and excluded from the default CI job like `test_integration_model.py`; the
fixture prefers the fleet snapshot under `weights/moment-1-base/` when it is staged and falls back
to the verified Hub cache otherwise."""

from __future__ import annotations

import json

import numpy as np
import pytest

from moment_pipeline import (
    ENCODER_BLOCKS,
    POLICY_FROZEN,
    WINDOW_LENGTH,
    adapt,
    classify,
    evaluate,
    features,
    knn_baseline,
    load_artifact,
    save_artifact,
)
from moment_pipeline.model import (
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    PINNED_WEIGHTS_FILENAME,
    fetch_verified_snapshot,
    load_moment,
)

pytestmark = pytest.mark.integration


def _window(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW_LENGTH) / 50.0
    freq = {"slow": 0.4, "mid": 1.6, "fast": 4.0}[label]
    base = np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28))
    return np.stack(
        [base * (1 + 0.2 * c) + rng.normal(0, 0.1, WINDOW_LENGTH) for c in range(3)]
    ).astype(np.float32)


LABELS = ["slow", "mid", "fast"]
RECORDS = [
    {
        "id": f"s{i:02d}",
        "x": _window(LABELS[i % 3], i),
        "label": LABELS[i % 3],
        "user": f"u{i // 3:02d}",
    }
    for i in range(18)
]


def _load():
    root = DEFAULT_WEIGHTS_DIR
    if (root / MANIFEST_NAME).is_file() and (root / PINNED_WEIGHTS_FILENAME).is_file():
        return load_moment(task="embedding", device="cpu", weights_dir=root)
    return load_moment(task="embedding", device="cpu", snapshot=fetch_verified_snapshot())


@pytest.fixture(scope="module")
def model():
    return _load()


def test_frozen_features_and_knn_separate_frequencies(model):
    vectors, ordered = features(RECORDS[:9], model)
    assert vectors.shape == (9, 768) and np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4
    )
    assert [r["id"] for r in ordered] == sorted(r["id"] for r in RECORDS[:9])
    baseline = knn_baseline(model, RECORDS[:12], RECORDS[12:], k=3)
    assert baseline["n"] == 6 and baseline["accuracy"] >= 2 / 3


def test_probe_then_one_epoch_unfreeze_and_artifact_round_trip(model, tmp_path):
    adapter = adapt(
        model,
        RECORDS[:12],
        RECORDS[12:],
        probe_steps=100,
        trainable_blocks=1,
        epochs=1,
        batch_size=6,
    )
    assert adapter.classes == ["fast", "mid", "slow"] and adapter.history[0]["stage"].startswith(
        "linear probe"
    )
    assert (
        adapter.config["n_trainable_head"] == 768 * 3 + 3
        and adapter.config["n_trainable_blocks"] == 7_079_424
    )
    assert adapter.config["n_total"] == 109_635_456 and len(adapter.history) == 2
    assert adapter.config["best_epoch"] in (0, 1) and adapter.config["trainable_blocks"] == 1
    assert (adapter.policy == POLICY_FROZEN) == (adapter.config["best_epoch"] == 0)
    metrics = evaluate(model, adapter, RECORDS[12:])
    assert metrics["n"] == 6 and metrics["adapted"] is True and metrics["policy"] == adapter.policy
    labels = classify(model, adapter, RECORDS[:3])
    assert labels["classes"] == adapter.classes and len(labels["probabilities"][0]) == 3
    artifact = save_artifact(model, adapter, tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert (
        "head.weight" in manifest["tensors"] and manifest["adapter"]["classes"] == adapter.classes
    )
    assert any(
        t.startswith(f"encoder.block.{ENCODER_BLOCKS - 1}.") for t in manifest["tensors"]
    ) == (adapter.policy != POLICY_FROZEN)
    fresh = _load()
    reloaded = load_artifact(fresh, artifact)
    assert classify(fresh, reloaded, RECORDS[:3])["probabilities"] == labels["probabilities"]
    assert (
        reloaded.config["best_epoch"] == adapter.config["best_epoch"]
        and reloaded.classes == adapter.classes
    )
    assert features(RECORDS[:2], fresh)[0].tolist() == features(RECORDS[:2], model)[0].tolist()
