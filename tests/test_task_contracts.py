"""Task-layer contracts that need no weights: mask accounting, guards, warm-up policy.

The model is replaced with a counting stub, so every assertion here is about this
repository's own arithmetic and control flow rather than MOMENT's numerics.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conftest import make_long_frame, set_value
from moment_pipeline.canonical import (
    hidden_position_fraction,
    missing_point_fraction,
    to_patch_view,
    to_windows,
)
from moment_pipeline.embedding import EmbeddingResult, embed
from moment_pipeline.imputation import DegenerateMaskError, mask_accounting, reconstruct
from moment_pipeline.model import LoadedMoment, ModelIdentity
from moment_pipeline.provenance import build_provenance
from moment_pipeline.validation import ValidationError

IDENTITY = ModelIdentity(
    name="AutonLab/MOMENT-1-base",
    revision="9fea447e740eb968a9e8d80c7562ae122bdb5dde",
    config_sha256="f" * 64,
    weights_sha256="1" * 64,
    weights_bytes=453_940_120,
    license="MIT",
    license_basis="model card metadata",
    weight_file_loaded="model.safetensors",
    seq_len=512,
    patch_len=8,
    patch_stride=8,
    d_model_effective=768,
    task="embedding",
    device="cpu",
    dtype="float32",
)


class CountingPipeline:
    """Counts forward passes and returns deterministic, correctly shaped output."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.reconstruct_calls = 0

    def embed(self, *, x_enc, input_mask, reduction):
        self.embed_calls += 1
        batch = x_enc.shape[0]
        return dataclasses.make_dataclass("Out", ["embeddings"])(torch.zeros(batch, 768))

    def reconstruct(self, *, x_enc, input_mask, mask):
        self.reconstruct_calls += 1
        return dataclasses.make_dataclass("Out", ["reconstruction"])(torch.zeros_like(x_enc))


def stub_model(task: str = "embedding") -> LoadedMoment:
    return LoadedMoment(
        pipeline=CountingPipeline(),
        identity=dataclasses.replace(IDENTITY, task=task),
        snapshot=None,
        proof={"weight_file": "model.safetensors", "encoder_tensors_checked": []},
    )


def multichannel_windows_with_one_missing_point():
    """2 channels x 512 points, a single missing value at index 13 of `c1` only.

    The fixture the two `masked_point_fraction` definitions disagreed on: 1/1024 counted
    as cells, 2/1024 when a channel-collapsed mask was broadcast back over the channels.
    """
    frame = make_long_frame(n_points=512, channels=("c1", "c2"), seed=3)
    return to_windows(set_value(frame, "A", "c1", 13, np.nan))


# --- R-5: one name, one definition, one denominator ---------------------------------


def test_both_masked_point_fraction_call_sites_agree_on_a_multichannel_fixture():
    windows = multichannel_windows_with_one_missing_point()
    assert windows.masked_point_count == 1
    assert windows.masked_point_fraction == pytest.approx(1 / 1024)

    visible = windows.model_point_mask
    patch_mask = to_patch_view(visible, windows.patch_length)
    accounting = mask_accounting(windows, visible, patch_mask)

    # The identically named quantity must be identical, not merely close.
    assert accounting["masked_point_fraction"] == windows.masked_point_fraction
    assert accounting["masked_point_count"] == windows.masked_point_count

    # The channel-collapsed quantity is a different number and now carries a different
    # name and a different denominator: non-padded (window, position) pairs, no channels.
    assert accounting["model_masked_point_count"] == 1
    assert accounting["model_masked_point_fraction"] == pytest.approx(1 / 512)
    assert accounting["masked_patch_fraction"] == pytest.approx(1 / 64)
    assert accounting["masked_patch_count"] == 1


def test_a_caller_mask_moves_only_the_model_side_fraction():
    """Source missingness is a property of the data; hiding is a property of the call."""
    windows = to_windows(make_long_frame(n_points=512, seed=4))
    hide = np.ones((1, 512), np.float32)
    hide[:, 8:40] = 0.0
    patch_mask = to_patch_view(hide, windows.patch_length)
    accounting = mask_accounting(windows, hide, patch_mask)
    assert accounting["masked_point_fraction"] == 0.0  # nothing was missing
    assert accounting["model_masked_point_fraction"] == pytest.approx(32 / 512)


def test_the_two_shared_definitions_use_the_denominators_they_document():
    point_mask = np.ones((1, 2, 512), np.float32)
    point_mask[0, 0, 300] = 0.0  # inside the kept region even when 256 steps are padded
    input_mask = np.ones((1, 512), np.float32)
    assert missing_point_fraction(point_mask, input_mask) == pytest.approx(1 / 1024)
    assert hidden_position_fraction(point_mask.min(axis=1), input_mask) == pytest.approx(1 / 512)

    # Padding is excluded from both denominators.
    input_mask[0, :256] = 0.0
    assert missing_point_fraction(point_mask, input_mask) == pytest.approx(1 / 512)
    assert hidden_position_fraction(point_mask.min(axis=1), input_mask) == pytest.approx(1 / 256)


# --- R-4: a window with nothing visible is refused, and the message names the cause --


def test_reconstruct_refuses_a_window_with_no_visible_patch():
    """Pre-fix this reached the model, RevIN returned NaN, and the M-2 finiteness check
    blamed the pre-fill -- for an input whose every value is finite."""
    windows = to_windows(make_long_frame(n_points=512, seed=6))
    hide = np.zeros((1, 512), np.float32)
    with pytest.raises(DegenerateMaskError) as excinfo:
        reconstruct(windows, stub_model("reconstruction"), mask=hide)
    message = str(excinfo.value)
    assert "DEGENERATE_MASK" in message
    assert "not a pre-fill fault" in message
    assert np.isfinite(windows.x_enc).all()


def test_a_fully_missing_series_channel_never_reaches_a_task_function():
    frame = make_long_frame(n_points=512, series=("A", "B"), channels=("c1", "c2"))
    selector = (frame["series_id"] == "B") & (frame["channel"] == "c2")
    frame.loc[selector, "value"] = np.nan
    with pytest.raises(ValidationError) as excinfo:
        to_windows(frame)
    assert excinfo.value.code == "FULLY_MISSING_SERIES_CHANNEL"


# --- R-9: the warm-up is opt-out-able -----------------------------------------------


def test_embed_runs_the_model_once_when_warmup_is_off():
    windows = to_windows(make_long_frame(n_points=512, seed=7))
    model = stub_model("embedding")
    embed(windows, model, warmup=False)
    assert model.pipeline.embed_calls == 1


def test_embed_still_warms_up_by_default():
    windows = to_windows(make_long_frame(n_points=512, seed=7))
    model = stub_model("embedding")
    result = embed(windows, model)
    assert model.pipeline.embed_calls == 2
    assert result.latency_seconds >= 0.0


def test_reconstruct_runs_the_model_once_when_warmup_is_off():
    windows = to_windows(make_long_frame(n_points=512, seed=8))
    model = stub_model("reconstruction")
    reconstruct(windows, model, warmup=False)
    assert model.pipeline.reconstruct_calls == 1


def test_reconstruct_still_warms_up_by_default():
    windows = to_windows(make_long_frame(n_points=512, seed=8))
    model = stub_model("reconstruction")
    reconstruct(windows, model)
    assert model.pipeline.reconstruct_calls == 2


# --- R-2: an exported embedding is self-describing about missingness -----------------


def test_embedding_result_records_the_missingness_it_could_not_pass_on():
    windows = multichannel_windows_with_one_missing_point()
    result = embed(windows, stub_model("embedding"), warmup=False)
    assert isinstance(result, EmbeddingResult)
    assert result.masked_point_count == 1
    assert result.masked_point_fraction == pytest.approx(1 / 1024)
    assert result.masked_patch_fraction == pytest.approx(1 / 64)
    assert result.missingness_visible_to_model is False
    assert "no per-point observedness mask" in result.missingness_policy


def test_embedding_provenance_carries_the_missingness_fractions_and_the_disclaimer():
    windows = multichannel_windows_with_one_missing_point()
    model = stub_model("embedding")
    record = build_provenance(model, windows, embed(windows, model, warmup=False))
    inference = record["inference"]
    assert inference["masked_point_fraction"] == pytest.approx(1 / 1024)
    assert inference["masked_point_count"] == 1
    assert inference["masked_patch_fraction"] == pytest.approx(1 / 64)
    assert inference["missingness_visible_to_model"] is False
    assert "MOMENT.embed accepts no per-point observedness mask" in (
        inference["missingness_policy"]
    )


def test_a_clean_window_reports_zero_missingness():
    """Negative control: the fields must discriminate, not merely exist."""
    windows = to_windows(make_long_frame(n_points=512, channels=("c1", "c2"), seed=3))
    result = embed(windows, stub_model("embedding"), warmup=False)
    assert result.masked_point_fraction == 0.0
    assert result.masked_point_count == 0
    assert result.masked_patch_fraction == 0.0
