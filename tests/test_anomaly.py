"""Task 3 contracts that need no weights.

The model is a stub whose reconstruction is a known function of the input, so every
assertion here is about this repository's scoring arithmetic — the scored domain, the
loss, the aggregation, the refusals — and not about MOMENT's numerics. The real-weight
half lives in `test_integration_model.py`.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_long_frame, set_value
from moment_pipeline import anomaly as anomaly_module
from moment_pipeline.anomaly import (
    ANOMALY_LOSSES,
    CHANNEL_AGGREGATIONS,
    AnomalyConfigError,
    AnomalyResult,
    aggregate_channels,
    residual,
    score_anomalies,
    score_from_reconstruction,
)
from moment_pipeline.canonical import to_windows
from moment_pipeline.imputation import reconstruct
from moment_pipeline.model import LoadedMoment
from moment_pipeline.provenance import build_provenance
from test_task_contracts import IDENTITY


class OffsetPipeline:
    """`reconstruct` returns `x_enc + offset[channel]` — a residual with a known value.

    Counting forward passes as well, so a test can prove a refusal happened *before* the
    model ran rather than after a 454 MB round trip.
    """

    def __init__(self, offsets: tuple[float, ...] = (1.0,)) -> None:
        self.offsets = offsets
        self.reconstruct_calls = 0

    def reconstruct(self, *, x_enc, input_mask, mask):
        self.reconstruct_calls += 1
        offsets = torch.tensor(self.offsets, dtype=x_enc.dtype).reshape(1, -1, 1)
        out = x_enc + offsets
        return dataclasses.make_dataclass("Out", ["reconstruction"])(out)


def stub_model(offsets: tuple[float, ...] = (1.0,)) -> LoadedMoment:
    return LoadedMoment(
        pipeline=OffsetPipeline(offsets),
        identity=dataclasses.replace(IDENTITY, task="reconstruction"),
        snapshot=None,
        proof={"weight_file": "model.safetensors", "encoder_tensors_checked": []},
    )


def clean_windows(channels: tuple[str, ...] = ("c1",)):
    return to_windows(make_long_frame(n_points=512, channels=channels, seed=5))


def windows_with_one_missing_point(channels: tuple[str, ...] = ("c1",), index: int = 13):
    frame = make_long_frame(n_points=512, channels=channels, seed=5)
    return to_windows(set_value(frame, "A", channels[0], index, np.nan))


# --- the primitive: shape, loss, and that nothing is thresholded ---------------------


def test_raw_score_shape_is_batch_channel_timestep_before_aggregation():
    """RFC Task 3: the primitive score is per `(batch, channel, timestep)`."""
    windows = clean_windows(channels=("c1", "c2"))
    result = score_anomalies(windows, stub_model(offsets=(1.0, 2.0)), warmup=False)

    assert result.anomaly_score.shape == (windows.n_windows, 2, 512)
    assert result.anomaly_score.shape == windows.x_enc.shape
    assert result.channel_aggregation == "none"


def test_mae_and_mse_are_the_documented_elementwise_losses():
    x = np.array([[[0.0, 1.0, -2.0]]], dtype=np.float32)
    reconstruction = np.array([[[1.0, -1.0, 0.0]]], dtype=np.float32)

    mae = residual(x, reconstruction, "mae")
    mse = residual(x, reconstruction, "mse")

    np.testing.assert_allclose(mae, [[[1.0, 2.0, 2.0]]])
    np.testing.assert_allclose(mse, np.square(mae), rtol=1e-6)


def test_the_two_losses_are_not_the_same_number_so_recording_the_loss_matters():
    """A provenance field nobody can distinguish from its alternative is decoration."""
    windows = clean_windows()
    model = stub_model(offsets=(3.0,))

    mae = score_anomalies(windows, model, loss="mae", warmup=False)
    mse = score_anomalies(windows, model, loss="mse", warmup=False)

    assert not np.allclose(np.nanmax(mae.anomaly_score), np.nanmax(mse.anomaly_score))
    assert mae.loss == "mae" and mse.loss == "mse"


def test_no_binary_threshold_is_applied_anywhere_in_the_result():
    """RFC: raw scores are the v1 output; there is no universal threshold.

    Three ways a threshold could sneak in — a keyword argument, a boolean verdict field,
    or a score quantized to a verdict — and none of them is present.
    """
    windows = clean_windows()
    result = score_anomalies(windows, stub_model(), warmup=False)

    parameters = inspect.signature(score_anomalies).parameters
    assert not [name for name in parameters if "threshold" in name.lower()]

    verdict_fields = [
        f.name
        for f in dataclasses.fields(AnomalyResult)
        if f.name in {"is_anomaly", "anomaly_label", "anomalies", "flagged", "threshold"}
    ]
    assert verdict_fields == []

    scored = result.anomaly_score[~np.isnan(result.anomaly_score)]
    assert np.unique(scored).size > 2, "a score with <=2 distinct values is a verdict"
    assert result.threshold_policy.startswith("none applied")


def test_the_score_is_the_residual_of_the_reconstruction_this_module_ran():
    """The score is not an independent quantity: it is |x - x_hat| for the stub's x_hat."""
    windows = clean_windows()
    model = stub_model(offsets=(2.5,))
    result = score_anomalies(windows, model, warmup=False)

    np.testing.assert_allclose(result.anomaly_score, np.full_like(result.anomaly_score, 2.5),
                               rtol=1e-5)


# --- the scored domain --------------------------------------------------------------


def test_a_fabricated_position_is_never_scored():
    """The pre-fill sentinel must not be scored: |x_hat - 0.0| describes the sentinel."""
    windows = windows_with_one_missing_point()
    result = score_anomalies(windows, stub_model(), warmup=False)

    scored = result.scored_mask == 1
    assert scored.any()
    assert (windows.point_mask[scored] == 1).all(), "a pre-filled cell entered the score"
    assert np.isnan(result.anomaly_score[windows.point_mask == 0]).all()


def test_one_missing_point_removes_its_whole_patch_from_the_scored_domain():
    """Patch quantization (RFC M-3) reaches the scored domain, not just the mask.

    One missing point at index 13 sits in patch 1 (points 8..15). The model was shown
    that patch as unobserved, so the residual at all 8 of its points is an imputation
    residual, not a self-reconstruction residual, and none of them is scored.
    """
    windows = windows_with_one_missing_point(index=13)
    result = score_anomalies(windows, stub_model(), warmup=False)

    unscored = np.flatnonzero(result.scored_mask[0, 0] == 0)
    np.testing.assert_array_equal(unscored, np.arange(8, 16))
    assert result.unscored_prefilled_count == 1
    assert result.unscored_hidden_by_patch_count == 7
    assert result.scored_point_count == 512 - 8


def test_the_three_domain_counts_partition_the_non_padded_cells():
    """scored + prefilled + hidden-by-patch == every non-padded (window, channel, point)."""
    windows = windows_with_one_missing_point(channels=("c1", "c2"))
    result = score_anomalies(windows, stub_model(offsets=(1.0, 1.0)), warmup=False)

    total = int(windows.input_mask.sum() * windows.n_channels)
    counted = (
        result.scored_point_count
        + result.unscored_prefilled_count
        + result.unscored_hidden_by_patch_count
    )
    assert counted == total
    assert result.scored_point_fraction == pytest.approx(result.scored_point_count / total)


def test_padding_is_never_scored():
    windows = to_windows(make_long_frame(n_points=400, seed=2))
    result = score_anomalies(windows, stub_model(), warmup=False)

    padded = np.broadcast_to(windows.input_mask[:, None, :], result.anomaly_score.shape) == 0
    assert padded.any()
    assert np.isnan(result.anomaly_score[padded]).all()
    assert (result.scored_mask[padded] == 0).all()


def test_unscored_positions_are_nan_by_construction_not_by_propagation():
    """The RFC's finiteness contract holds where the score is defined.

    NaN in `anomaly_score` marks "not defined here". The underlying residual is finite
    everywhere, so a NaN can never have come from the model or the pre-fill.
    """
    windows = windows_with_one_missing_point()
    result = score_anomalies(windows, stub_model(), warmup=False)

    assert np.isfinite(result.reconstruction_error).all()
    assert np.isfinite(result.anomaly_score[result.scored_mask == 1]).all()
    assert np.isnan(result.anomaly_score[result.scored_mask == 0]).all()


# --- aggregation --------------------------------------------------------------------


def test_aggregation_collapses_the_channel_axis_and_changes_nothing_else():
    """RFC required test: aggregation touches only the documented dimension."""
    windows = clean_windows(channels=("c1", "c2"))
    model = stub_model(offsets=(1.0, 3.0))
    none = score_anomalies(windows, model, warmup=False)

    for how in ("mean", "max"):
        aggregated = score_anomalies(windows, model, channel_aggregation=how, warmup=False)
        assert aggregated.anomaly_score.shape == (windows.n_windows, 512)
        assert aggregated.channel_aggregation == how
        # the reconstruction and the raw residual keep their channel axis untouched
        np.testing.assert_allclose(aggregated.reconstruction, none.reconstruction)
        np.testing.assert_allclose(aggregated.reconstruction_error, none.reconstruction_error)
        expected = (np.nanmean if how == "mean" else np.nanmax)(none.anomaly_score, axis=1)
        np.testing.assert_allclose(aggregated.anomaly_score, expected, rtol=1e-6)

    # and the two aggregations are distinguishable, so recording which one ran matters
    mean = score_anomalies(windows, model, channel_aggregation="mean", warmup=False)
    maximum = score_anomalies(windows, model, channel_aggregation="max", warmup=False)
    assert not np.allclose(np.nanmax(mean.anomaly_score), np.nanmax(maximum.anomaly_score))


def test_none_is_the_default_aggregation_and_returns_the_score_untouched():
    score = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    np.testing.assert_array_equal(aggregate_channels(score, "none"), score)
    assert anomaly_module.DEFAULT_AGGREGATION == "none"


def test_aggregation_is_nan_aware_and_stays_nan_only_where_no_channel_scored():
    score = np.array([[[1.0, np.nan], [np.nan, np.nan]]], dtype=np.float32)

    np.testing.assert_array_equal(aggregate_channels(score, "mean"), [[1.0, np.nan]])
    np.testing.assert_array_equal(aggregate_channels(score, "max"), [[1.0, np.nan]])


# --- refusals -----------------------------------------------------------------------


@pytest.mark.parametrize("loss", ["rmse", "l1", "", "MAE"])
def test_an_unsupported_loss_is_refused_before_the_model_runs(loss):
    windows = clean_windows()
    model = stub_model()

    with pytest.raises(AnomalyConfigError) as excinfo:
        score_anomalies(windows, model, loss=loss, warmup=False)

    assert str(list(ANOMALY_LOSSES)) in str(excinfo.value)
    assert model.pipeline.reconstruct_calls == 0, "refused only after a forward pass"


@pytest.mark.parametrize("how", ["median", "sum", "concat"])
def test_an_unsupported_aggregation_is_refused_before_the_model_runs(how):
    windows = clean_windows()
    model = stub_model()

    with pytest.raises(AnomalyConfigError) as excinfo:
        score_anomalies(windows, model, channel_aggregation=how, warmup=False)

    assert str(list(CHANNEL_AGGREGATIONS)) in str(excinfo.value)
    assert model.pipeline.reconstruct_calls == 0


def test_scoring_reuses_the_pinned_reconstruction_entry_point_once():
    """No second forward pass, and no second entry point: one `reconstruct` call."""
    windows = clean_windows()
    model = stub_model()

    score_anomalies(windows, model, warmup=False)

    assert model.pipeline.reconstruct_calls == 1


def test_score_from_reconstruction_reuses_a_reconstruction_without_rerunning_the_model():
    windows = clean_windows()
    model = stub_model(offsets=(2.0,))
    reconstructed = reconstruct(windows, model, warmup=False)
    calls_after_reconstruct = model.pipeline.reconstruct_calls

    result = score_from_reconstruction(windows, reconstructed)

    assert model.pipeline.reconstruct_calls == calls_after_reconstruct
    np.testing.assert_allclose(result.anomaly_score, 2.0, rtol=1e-5)


# --- export -------------------------------------------------------------------------


def test_to_frame_carries_the_rfc_columns_one_row_per_non_padded_position():
    windows = clean_windows(channels=("c1", "c2"))
    result = score_anomalies(windows, stub_model(offsets=(1.0, 2.0)), warmup=False)

    frame = result.to_frame()

    for column in (
        "series_id",
        "timestamp",
        "channel",
        "reconstruction",
        "reconstruction_error",
        "anomaly_score",
    ):
        assert column in frame.columns
    assert len(frame) == 512 * 2
    assert set(frame["channel"]) == {"c1", "c2"}
    assert not frame["timestamp"].isna().any()


def test_to_frame_drops_padding_and_keeps_unscored_real_positions_visible():
    windows = to_windows(set_value(make_long_frame(n_points=400, seed=2), "A", "c1", 13, np.nan))
    result = score_anomalies(windows, stub_model(), warmup=False)

    frame = result.to_frame()

    assert len(frame) == 400, "padded positions must not reach the export"
    assert not frame["timestamp"].isna().any()
    unscored = frame.loc[~frame["scored"]]
    assert len(unscored) == 8, "the hidden patch stays visible as unscored rows"
    assert unscored["anomaly_score"].isna().all()


def test_an_aggregated_frame_omits_the_columns_that_do_not_survive_the_collapse():
    """Emitting a per-channel `reconstruction` next to a channel-collapsed score would
    invent a number. The aggregation is score-only, and the frame says so."""
    windows = clean_windows(channels=("c1", "c2"))
    result = score_anomalies(
        windows, stub_model(offsets=(1.0, 3.0)), channel_aggregation="max", warmup=False
    )

    frame = result.to_frame()

    assert len(frame) == 512
    assert set(frame["channel"]) == {"max"}
    assert "reconstruction" not in frame.columns
    assert "reconstruction_error" not in frame.columns
    assert frame["anomaly_score"].max() == pytest.approx(3.0, rel=1e-5)


def test_provenance_records_the_loss_the_aggregation_and_the_absence_of_a_threshold():
    windows = clean_windows(channels=("c1", "c2"))
    model = stub_model(offsets=(1.0, 2.0))
    result = score_anomalies(windows, model, loss="mse", channel_aggregation="mean",
                             warmup=False)

    provenance = build_provenance(model, windows, result)
    inference = provenance["inference"]

    assert inference["anomaly_loss"] == "mse"
    assert inference["channel_aggregation"] == "mean"
    assert inference["threshold_policy"].startswith("none applied")
    assert "unmasked self-reconstruction residual" in inference["score_policy"]
    assert inference["scored_point_count"] == 512 * 2
    assert inference["unscored_prefilled_count"] == 0
    assert "threshold" not in {
        key for key in inference if key.endswith(("_value", "_cutoff"))
    }


def test_a_timestampless_windowset_still_exports():
    """`WindowSet.timestamps` defaults to (); the export must not require it."""
    windows = dataclasses.replace(clean_windows(), timestamps=())
    result = score_anomalies(windows, stub_model(), warmup=False)

    frame = result.to_frame()

    assert len(frame) == 512
    assert pd.api.types.is_integer_dtype(frame["timestamp"])
