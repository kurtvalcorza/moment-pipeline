"""Integration: the real pinned weights on CPU.

Marked `integration` and excluded from the default CI job. Everything here needs the
~454 MB `model.safetensors` and network access on a cold cache; none of it needs a GPU.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conftest import make_long_frame, set_value
from moment_pipeline.canonical import to_windows
from moment_pipeline.embedding import TaskMismatchError, embed
from moment_pipeline.imputation import reconstruct
from moment_pipeline.model import (
    KNOWN_FORBIDDEN_WEIGHT_FILE,
    PINNED_CONFIG_SHA256,
    PINNED_REVISION,
    PINNED_WEIGHTS_BYTES,
    PINNED_WEIGHTS_SHA256,
    fetch_verified_snapshot,
    find_forbidden_weight_files,
    load_moment,
)
from moment_pipeline.provenance import build_provenance

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def snapshot():
    return fetch_verified_snapshot()


@pytest.fixture(scope="module")
def embedding_model(snapshot):
    return load_moment(task="embedding", device="cpu", snapshot=snapshot)


@pytest.fixture(scope="module")
def reconstruction_model(snapshot):
    return load_moment(task="reconstruction", device="cpu", snapshot=snapshot)


# --- supply chain against the real artifact -----------------------------------------


def test_pinned_revision_and_digests_resolve(snapshot):
    assert snapshot.path.name == PINNED_REVISION
    assert snapshot.revision == PINNED_REVISION
    assert snapshot.config_sha256 == PINNED_CONFIG_SHA256
    assert snapshot.config_bytes == 949
    assert snapshot.weights_sha256 == PINNED_WEIGHTS_SHA256
    assert snapshot.weights_bytes == PINNED_WEIGHTS_BYTES
    assert (snapshot.seq_len, snapshot.patch_len, snapshot.patch_stride) == (512, 8, 8)
    assert snapshot.task_name_in_config == "reconstruction"


def test_snapshot_never_contains_the_pickle_weights(snapshot):
    assert find_forbidden_weight_files(snapshot.path) == []
    assert not (snapshot.path / KNOWN_FORBIDDEN_WEIGHT_FILE).exists()
    assert sorted(p.name for p in snapshot.path.iterdir()) == [
        "README.md",
        "config.json",
        "model.safetensors",
    ]


# --- which weight file was actually loaded (RFC invariant 7) ------------------------


def test_reconstruction_load_is_proven_against_the_safetensors_file(reconstruction_model):
    proof = reconstruction_model.proof
    assert proof["weight_file"] == "model.safetensors"
    assert proof["n_tensors_in_file"] == 116
    assert len(proof["encoder_tensors_checked"]) >= 3
    assert proof["head_tensors_checked"] == ["head.linear.bias", "head.linear.weight"]
    assert proof["head_type"] == "PretrainHead"


def test_embedding_load_is_proven_and_head_free(embedding_model):
    assert embedding_model.proof["head_type"] == "Identity"
    assert embedding_model.proof["head_tensors_checked"] == []
    assert type(embedding_model.pipeline.head) is torch.nn.Identity
    assert [k for k in embedding_model.pipeline.state_dict() if k.startswith("head.")] == []


def test_d_model_is_derived_not_declared(snapshot, embedding_model):
    import json

    declared = json.loads((snapshot.path / "config.json").read_text())["d_model"]
    assert declared is None  # config.json leaves it null; the backbone decides
    assert embedding_model.identity.d_model_effective == 768


def test_two_loads_give_identical_head_weights(snapshot):
    a = load_moment(task="reconstruction", device="cpu", snapshot=snapshot)
    b = load_moment(task="reconstruction", device="cpu", snapshot=snapshot)
    sa, sb = a.pipeline.state_dict(), b.pipeline.state_dict()
    head_keys = [k for k in sa if k.startswith("head.")]
    assert head_keys
    for key in head_keys:
        assert torch.equal(sa[key], sb[key]), key
    assert torch.equal(sa["encoder.final_layer_norm.weight"], sb["encoder.final_layer_norm.weight"])


# --- embeddings ---------------------------------------------------------------------


def test_embedding_dimension_and_determinism(embedding_model):
    windows = to_windows(make_long_frame(n_points=512, series=("A", "B"), seed=5))
    first = embed(windows, embedding_model)
    second = embed(windows, embedding_model)
    assert first.embeddings.shape == (2, 768)
    assert first.d_model == 768
    assert np.allclose(first.embeddings, second.embeddings, atol=1e-6)
    assert np.isfinite(first.embeddings).all()
    assert first.reduction == "mean"
    frame = first.to_frame()
    assert list(frame.columns[:2]) == ["series_id", "window_id"]
    assert frame.shape == (2, 770)


def test_embedding_refuses_a_reconstruction_instance(reconstruction_model):
    windows = to_windows(make_long_frame(n_points=512))
    with pytest.raises(TaskMismatchError):
        embed(windows, reconstruction_model)


def test_embeddings_are_finite_with_nan_bearing_input(embedding_model):
    frame = make_long_frame(n_points=512, seed=9)
    for index in range(13, 45):
        frame = set_value(frame, "A", "c1", index, np.nan)
    windows = to_windows(frame)
    result = embed(windows, embedding_model)
    assert np.isfinite(result.embeddings).all()


# --- reconstruction -----------------------------------------------------------------


def _masked_windows(seed: int = 11, nan_slice: slice | None = None):
    frame = make_long_frame(n_points=512, series=("A", "B"), seed=seed)
    if nan_slice is not None:
        for series_id in ("A", "B"):
            for index in range(nan_slice.start, nan_slice.stop):
                frame = set_value(frame, series_id, "c1", index, np.nan)
    return to_windows(frame)


def test_reconstruction_smoke_with_four_masked_patches(reconstruction_model):
    windows = _masked_windows()
    mask = np.ones((2, 512), np.float32)
    mask[:, 8:40] = 0.0  # patches 1-4, exactly aligned
    result = reconstruct(windows, reconstruction_model, mask=mask)
    assert result.reconstruction.shape == (2, 1, 512)
    assert np.isfinite(result.reconstruction).all()
    assert result.masked_patch_count == 8  # 4 patches x 2 windows
    assert result.masked_patch_fraction == pytest.approx(4 / 64)
    assert result.masked_point_fraction == pytest.approx(32 / 512)
    assert result.model_mask[:, 8:40].sum() == 0.0
    assert result.model_mask.sum() == 2 * 480


def test_nan_regression_masked_positions_still_yield_finite_output(reconstruction_model):
    """RFC M-2: NaN-bearing BYOD input must produce finite reconstruction."""
    windows = _masked_windows(nan_slice=slice(8, 40))
    assert np.isfinite(windows.x_enc).all()
    result = reconstruct(windows, reconstruction_model)
    assert np.isfinite(result.reconstruction).all()
    assert result.masked_point_fraction == pytest.approx(32 / 512)
    assert result.masked_patch_fraction == pytest.approx(4 / 64)


def test_nan_off_patch_boundary_expands_to_whole_patches(reconstruction_model):
    windows = _masked_windows(nan_slice=slice(13, 14))
    result = reconstruct(windows, reconstruction_model)
    assert np.isfinite(result.reconstruction).all()
    assert result.masked_point_fraction == pytest.approx(1 / 512)
    assert result.masked_patch_fraction == pytest.approx(1 / 64)
    assert result.model_mask[0, 8:16].sum() == 0.0


def test_reconstruct_refuses_an_embedding_instance(embedding_model):
    windows = _masked_windows()
    with pytest.raises(TaskMismatchError):
        reconstruct(windows, embedding_model)


def test_negative_control_raw_nan_reaches_the_output(reconstruction_model):
    """Documents *why* the pre-fill exists.

    `MOMENT.reconstruct` has no `nan_to_num` (unlike `MOMENT.embed`), and
    `PatchEmbedding.forward` computes `mask * linear(x) + (1 - mask) * mask_embedding`,
    so 0 * NaN = NaN survives the masking. Feeding raw NaN straight to the pinned entry
    point therefore produces NaN — which is exactly what the canonical converter prevents.
    """
    generator = torch.Generator().manual_seed(0)
    x_enc = torch.randn(2, 1, 512, generator=generator)
    x_enc[:, :, 8:40] = float("nan")
    input_mask = torch.ones(2, 512)
    mask = torch.ones(2, 512)
    mask[:, 8:40] = 0.0
    with torch.no_grad():
        out = reconstruction_model.pipeline.reconstruct(
            x_enc=x_enc, input_mask=input_mask, mask=mask
        )
    assert torch.isnan(out.reconstruction).any()


def test_reconstruct_refuses_non_finite_windows(reconstruction_model):
    windows = _masked_windows()
    poisoned = np.array(windows.x_enc, copy=True)
    poisoned[0, 0, 3] = np.nan
    broken = dataclasses.replace(windows, x_enc=poisoned)
    with pytest.raises(ValueError, match="non-finite"):
        reconstruct(broken, reconstruction_model)


# --- provenance over a real run -----------------------------------------------------


def test_provenance_of_a_real_embedding_run(embedding_model):
    windows = to_windows(make_long_frame(n_points=512, series=("A", "B")))
    result = embed(windows, embedding_model)
    record = build_provenance(embedding_model, windows, result)
    assert record["model"]["weights_sha256"] == PINNED_WEIGHTS_SHA256
    assert record["model"]["config_sha256"] == PINNED_CONFIG_SHA256
    assert record["load_proof"]["weight_file"] == "model.safetensors"
    assert record["inference"]["embedding_dim"] == 768
    assert record["inference"]["latency_seconds"] > 0.0
    assert record["runtime"]["momentfm"] == "0.1.5"
