"""Common validation contract — one passing and one failing fixture per rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_long_frame
from moment_pipeline.config import MomentConfig, ResourceLimits
from moment_pipeline.validation import ValidationError, validate_long_frame


def _codes(excinfo) -> str:
    return excinfo.value.code


# --- rule 1: required columns -------------------------------------------------------


def test_clean_frame_passes_every_rule(clean_frame):
    report, frame = validate_long_frame(clean_frame)
    assert report.n_rows == 512
    assert report.series_ids == ("A",)
    assert report.channels == ("c1",)
    assert report.has_irregular_frequency is False
    assert frame["value"].notna().all()


def test_missing_column_rejected(clean_frame):
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(clean_frame.drop(columns=["channel"]))
    assert _codes(excinfo) == "MISSING_COLUMNS"
    assert excinfo.value.details["missing"] == ["channel"]


def test_empty_frame_rejected(clean_frame):
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(clean_frame.iloc[0:0])
    assert _codes(excinfo) == "EMPTY_INPUT"


# --- rule 2: non-null ids -----------------------------------------------------------


@pytest.mark.parametrize("column", ["series_id", "channel"])
@pytest.mark.parametrize("bad", [None, "   "])
def test_null_or_blank_id_rejected(clean_frame, column, bad):
    frame = clean_frame.copy()
    frame.loc[3, column] = bad
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame)
    assert _codes(excinfo) == "NULL_ID"
    assert excinfo.value.details["column"] == column


# --- rule 3: timestamps parse -------------------------------------------------------


def test_null_timestamp_rejected(clean_frame):
    frame = clean_frame.copy()
    frame["timestamp"] = frame["timestamp"].astype(object)
    frame.loc[5, "timestamp"] = None
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame)
    assert _codes(excinfo) == "NULL_TIMESTAMP"


def test_unparseable_timestamp_rejected(clean_frame):
    frame = clean_frame.copy()
    frame["timestamp"] = frame["timestamp"].astype(str)
    frame.loc[7, "timestamp"] = "not-a-timestamp"
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame)
    assert _codes(excinfo) == "UNPARSEABLE_TIMESTAMP"
    assert excinfo.value.details["row_positions"] == [7]


def test_string_timestamps_are_accepted(clean_frame):
    frame = clean_frame.copy()
    frame["timestamp"] = frame["timestamp"].astype(str)
    report, _ = validate_long_frame(frame)
    assert report.n_rows == 512


# --- rule 4: duplicates -------------------------------------------------------------


def test_duplicate_key_rejected(clean_frame):
    frame = pd.concat([clean_frame, clean_frame.iloc[[10]]], ignore_index=True)
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame)
    assert _codes(excinfo) == "DUPLICATE_ROWS"
    assert excinfo.value.details["examples"]


def test_same_timestamp_in_two_channels_is_not_a_duplicate():
    frame = make_long_frame(channels=("c1", "c2"))
    report, _ = validate_long_frame(frame)
    assert report.channels == ("c1", "c2")


# --- rule 5: deterministic channel ordering -----------------------------------------


def test_channel_order_is_sorted_not_row_order():
    frame = make_long_frame(channels=("zeta", "alpha"))
    report, _ = validate_long_frame(frame)
    assert report.channels == ("alpha", "zeta")
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert validate_long_frame(shuffled)[0].channels == ("alpha", "zeta")


# --- rule 6: numeric policy ---------------------------------------------------------


def test_non_numeric_value_rejected(clean_frame):
    frame = clean_frame.copy()
    frame["value"] = frame["value"].astype(object)
    frame.loc[11, "value"] = "abc"
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame)
    assert _codes(excinfo) == "NON_NUMERIC_VALUE"
    assert excinfo.value.details["row_positions"] == [11]


def test_numeric_string_is_accepted(clean_frame):
    frame = clean_frame.copy()
    frame["value"] = frame["value"].astype(str)
    report, normalized = validate_long_frame(frame)
    assert report.n_rows == 512
    assert normalized["value"].dtype.kind == "f"


def test_infinite_value_rejected(clean_frame):
    frame = clean_frame.copy()
    frame.loc[13, "value"] = np.inf
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame)
    assert _codes(excinfo) == "NON_FINITE_VALUE"


def test_nan_value_is_missing_not_an_error(clean_frame):
    frame = clean_frame.copy()
    frame.loc[13, "value"] = np.nan
    report, normalized = validate_long_frame(frame)
    assert report.n_rows == 512
    assert int(normalized["value"].isna().sum()) == 1


# --- rule 7: irregular frequency is surfaced, not rejected --------------------------


def test_irregular_frequency_is_surfaced(clean_frame):
    frame = clean_frame.copy()
    frame.loc[100, "timestamp"] = frame.loc[100, "timestamp"] + pd.Timedelta("37min")
    report, _ = validate_long_frame(frame)
    assert report.has_irregular_frequency is True
    assert report.irregular_series == ("A",)
    assert len(report.frequencies[0].distinct_deltas) > 1


def test_irregular_frequency_rejected_only_under_strict_flag(clean_frame):
    frame = clean_frame.copy()
    frame.loc[100, "timestamp"] = frame.loc[100, "timestamp"] + pd.Timedelta("37min")
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame, MomentConfig(strict_frequency=True))
    assert _codes(excinfo) == "IRREGULAR_FREQUENCY"


def test_regular_frame_passes_strict_frequency(clean_frame):
    report, _ = validate_long_frame(clean_frame, MomentConfig(strict_frequency=True))
    assert report.has_irregular_frequency is False


# --- rule 9: empty channels ---------------------------------------------------------


def test_globally_empty_channel_rejected():
    frame = make_long_frame(channels=("c1", "c2"))
    frame.loc[frame["channel"] == "c2", "value"] = np.nan
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame)
    assert _codes(excinfo) == "EMPTY_CHANNEL"
    assert excinfo.value.details["channels"] == ["c2"]


def test_channel_missing_for_one_series_only_is_surfaced():
    frame = make_long_frame(series=("A", "B"), channels=("c1", "c2"))
    selector = (frame["series_id"] == "B") & (frame["channel"] == "c2")
    frame.loc[selector, "value"] = np.nan
    report, _ = validate_long_frame(frame)
    assert report.fully_missing_series_channels == (("B", "c2"),)


# --- rule 10: resource limits -------------------------------------------------------


def test_max_rows_limit(clean_frame):
    config = MomentConfig(limits=ResourceLimits(max_rows=10))
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(clean_frame, config)
    assert excinfo.value.details["limit"] == "max_rows"


def test_max_series_limit():
    frame = make_long_frame(n_points=16, series=("A", "B", "C"))
    config = MomentConfig(limits=ResourceLimits(max_series=2))
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame, config)
    assert excinfo.value.details["limit"] == "max_series"


def test_max_channels_limit():
    frame = make_long_frame(n_points=16, channels=("a", "b", "c"))
    config = MomentConfig(limits=ResourceLimits(max_channels=2))
    with pytest.raises(ValidationError) as excinfo:
        validate_long_frame(frame, config)
    assert excinfo.value.details["limit"] == "max_channels"


def test_limits_allow_input_at_the_boundary():
    frame = make_long_frame(n_points=16, series=("A", "B"))
    config = MomentConfig(limits=ResourceLimits(max_series=2, max_rows=32))
    report, _ = validate_long_frame(frame, config)
    assert report.series_ids == ("A", "B")
