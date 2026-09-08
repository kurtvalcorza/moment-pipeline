"""Canonical converter: ordering, alignment, padding, truncation, pre-fill, patch masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_long_frame, set_value
from moment_pipeline.canonical import expand_patch_view, to_patch_view, to_windows
from moment_pipeline.config import MomentConfig, ResourceLimits
from moment_pipeline.validation import ValidationError


def test_shapes_and_ids(clean_frame):
    windows = to_windows(clean_frame)
    assert windows.x_enc.shape == (1, 1, 512)
    assert windows.input_mask.shape == (1, 512)
    assert windows.point_mask.shape == (1, 1, 512)
    assert windows.window_ids == ("A::w0",)
    assert windows.x_enc.dtype == np.float32


def test_channel_axis_follows_sorted_names_not_row_order():
    frame = make_long_frame(n_points=512, channels=("zeta", "alpha"))
    frame.loc[frame["channel"] == "alpha", "value"] = 1.0
    frame.loc[frame["channel"] == "zeta", "value"] = 2.0
    windows = to_windows(frame)
    assert windows.channels == ("alpha", "zeta")
    assert np.allclose(windows.x_enc[0, 0], 1.0)
    assert np.allclose(windows.x_enc[0, 1], 2.0)


def test_row_order_does_not_change_the_tensor(clean_frame):
    shuffled = clean_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = to_windows(clean_frame)
    b = to_windows(shuffled)
    assert np.array_equal(a.x_enc, b.x_enc)
    assert np.array_equal(a.point_mask, b.point_mask)
    assert a.series_ids == b.series_ids


def test_long_series_is_truncated_to_the_last_512_points():
    frame = make_long_frame(n_points=600)
    windows = to_windows(frame)
    assert windows.truncated == (True,)
    assert windows.padded == (False,)
    assert windows.n_source_timestamps == (600,)
    expected = frame["value"].to_numpy()[-512:]
    assert np.allclose(windows.x_enc[0, 0], expected.astype(np.float32))
    assert windows.input_mask.sum() == 512


def test_short_series_is_left_padded_and_disclosed():
    frame = make_long_frame(n_points=100)
    windows = to_windows(frame)
    assert windows.padded == (True,)
    assert windows.truncated == (False,)
    assert windows.input_mask[0, :412].sum() == 0.0
    assert windows.input_mask[0, 412:].sum() == 100.0
    assert np.allclose(windows.x_enc[0, 0, :412], 0.0)
    assert np.allclose(windows.x_enc[0, 0, 412:], frame["value"].to_numpy().astype(np.float32))
    assert windows.point_mask[0, 0, :412].sum() == 0.0
    # Padding is disclosed separately and never inflates the missingness metric.
    assert windows.masked_point_fraction == 0.0
    assert windows.padded_fraction == pytest.approx(412 / 512)


def test_prefill_oracle_nan_becomes_finite_and_only_the_mask_remembers(clean_frame):
    frame = set_value(clean_frame, "A", "c1", 13, np.nan)
    frame = set_value(frame, "A", "c1", 400, np.nan)
    windows = to_windows(frame)
    assert np.isfinite(windows.x_enc).all()
    missing = np.flatnonzero(windows.point_mask[0, 0] == 0.0)
    assert missing.tolist() == [13, 400]
    assert windows.x_enc[0, 0, 13] == 0.0
    assert windows.x_enc[0, 0, 400] == 0.0
    observed = np.ones(512, bool)
    observed[[13, 400]] = False
    original = clean_frame["value"].to_numpy().astype(np.float32)
    # The sentinel does not leak into observed positions.
    assert np.allclose(windows.x_enc[0, 0, observed], original[observed])


def test_prefill_value_is_configurable_and_still_finite(clean_frame):
    frame = set_value(clean_frame, "A", "c1", 13, np.nan)
    windows = to_windows(frame, MomentConfig(prefill_value=-1.5))
    assert windows.x_enc[0, 0, 13] == pytest.approx(-1.5)
    assert windows.point_mask[0, 0, 13] == 0.0


def test_patch_expansion_oracle_one_missing_point_costs_a_whole_patch(clean_frame):
    frame = set_value(clean_frame, "A", "c1", 13, np.nan)
    windows = to_windows(frame)
    assert windows.masked_point_fraction == pytest.approx(1 / 512)
    assert windows.masked_patch_fraction == pytest.approx(1 / 64)
    patch_mask = windows.patch_mask
    assert patch_mask[0, 1] == 0.0
    assert patch_mask[0, np.arange(64) != 1].sum() == 63.0
    # The mask handed to momentfm is already quantized: the whole patch, not one point.
    quantized = windows.patch_quantized_mask()
    assert quantized[0, 8:16].sum() == 0.0
    assert quantized[0].sum() == 504.0


def test_patch_boundaries_are_respected(clean_frame):
    frame = set_value(clean_frame, "A", "c1", 7, np.nan)
    frame = set_value(frame, "A", "c1", 8, np.nan)
    windows = to_windows(frame)
    assert windows.masked_point_fraction == pytest.approx(2 / 512)
    assert windows.masked_patch_fraction == pytest.approx(2 / 64)
    assert windows.patch_mask[0, 0] == 0.0
    assert windows.patch_mask[0, 1] == 0.0


def test_channel_collapse_is_conservative():
    frame = make_long_frame(n_points=512, channels=("c1", "c2"))
    frame = set_value(frame, "A", "c2", 13, np.nan)
    windows = to_windows(frame)
    assert windows.point_mask[0, 0, 13] == 1.0
    assert windows.point_mask[0, 1, 13] == 0.0
    # A point missing in ANY channel is unobserved for the model, which has no channel axis.
    assert windows.model_point_mask[0, 13] == 0.0
    assert windows.masked_point_fraction == pytest.approx(1 / 1024)
    assert windows.masked_patch_fraction == pytest.approx(1 / 64)


def test_all_missing_window_rejected():
    """Observed history exists, but none of it survives into the last-512 window.

    This is the only route to `ALL_MISSING_WINDOW` now that a (series, channel) with no
    observed value anywhere is refused earlier, by `FULLY_MISSING_SERIES_CHANNEL` (R-4).
    """
    frame = make_long_frame(n_points=600, series=("A", "B"))
    selector = frame["series_id"] == "B"
    positions = np.flatnonzero(selector.to_numpy())[88:]  # keep the first 88 observed
    frame.loc[frame.index[positions], "value"] = np.nan
    with pytest.raises(ValidationError) as excinfo:
        to_windows(frame)
    assert excinfo.value.code == "ALL_MISSING_WINDOW"
    assert excinfo.value.details["series_id"] == "B"


def test_single_channel_series_with_no_observed_value_is_rejected_in_validation():
    """Pre-fix this reached `to_windows` and raised `ALL_MISSING_WINDOW`; the earlier,
    more specific code names the actual shape of the problem (R-4)."""
    frame = make_long_frame(n_points=512, series=("A", "B"))
    frame.loc[frame["series_id"] == "B", "value"] = np.nan
    with pytest.raises(ValidationError) as excinfo:
        to_windows(frame)
    assert excinfo.value.code == "FULLY_MISSING_SERIES_CHANNEL"
    assert excinfo.value.details["pairs"] == [["B", "c1"]]


def test_max_windows_limit():
    frame = make_long_frame(n_points=8, series=("A", "B", "C"))
    config = MomentConfig(limits=ResourceLimits(max_windows=2))
    with pytest.raises(ValidationError) as excinfo:
        to_windows(frame, config)
    assert excinfo.value.details["limit"] == "max_windows"


def test_irregular_timestamps_are_carried_through_without_interpolation(clean_frame):
    frame = clean_frame.copy()
    frame = frame.drop(index=[200, 201]).reset_index(drop=True)
    windows = to_windows(frame)
    assert windows.validation.has_irregular_frequency is True
    # 510 real points, left-padded to 512; nothing was invented to fill the gap.
    assert windows.n_source_timestamps == (510,)
    assert windows.input_mask[0].sum() == 510.0
    assert windows.point_mask[0, 0].sum() == 510.0


def test_to_patch_view_matches_upstream_masking():
    """The patch quantization is copied from momentfm; prove it against the real thing."""
    torch = pytest.importorskip("torch")
    masking = pytest.importorskip("momentfm.utils.masking")
    rng = np.random.default_rng(3)
    mask = (rng.random((4, 512)) > 0.02).astype(np.float32)
    ours = to_patch_view(mask, 8)
    theirs = masking.Masking.convert_seq_to_patch_view(torch.from_numpy(mask), 8)
    assert np.array_equal(ours, theirs.numpy().astype(np.float32))


def test_expand_patch_view_round_trip():
    patch = np.ones((2, 64), np.float32)
    patch[0, 3] = 0.0
    expanded = expand_patch_view(patch, 8)
    assert expanded.shape == (2, 512)
    assert expanded[0, 24:32].sum() == 0.0
    assert np.array_equal(to_patch_view(expanded, 8), patch)


def test_timestamps_are_right_aligned_with_nat_padding():
    frame = make_long_frame(n_points=10)
    windows = to_windows(frame)
    stamps = windows.timestamps[0]
    assert pd.isna(stamps[:502]).all()
    assert not pd.isna(stamps[502:]).any()
    assert stamps[-1] == frame["timestamp"].max().to_datetime64()
