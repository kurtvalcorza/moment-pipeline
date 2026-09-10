from __future__ import annotations

import pandas as pd
import pytest

import moment_pipeline.csvio as csvio
from moment_pipeline import ValidationError, read_long_csv_bytes


def _valid_payload() -> bytes:
    return (
        b"series_id,timestamp,channel,value\n"
        b"A,2026-01-01T00:00:00,c1,1.0\n"
    )


def test_valid_csv_bytes_parse_with_exact_long_schema() -> None:
    frame = read_long_csv_bytes(_valid_payload())
    assert list(frame.columns) == ["series_id", "timestamp", "channel", "value"]
    assert len(frame) == 1


def test_duplicate_headers_are_rejected_before_pandas_can_mangle(monkeypatch) -> None:
    payload = (
        b"series_id,timestamp,channel,value,value\n"
        b"A,2026-01-01T00:00:00,c1,1.0,2.0\n"
    )

    def pandas_must_not_run(*args, **kwargs):
        raise AssertionError("pandas.read_csv ran before raw-header validation")

    monkeypatch.setattr(csvio.pd, "read_csv", pandas_must_not_run)
    with pytest.raises(ValidationError) as excinfo:
        read_long_csv_bytes(payload)
    assert excinfo.value.code == "DUPLICATE_COLUMNS"
    assert excinfo.value.details["columns"] == ["value"]


def test_case_or_whitespace_ambiguous_headers_are_rejected() -> None:
    payload = (
        b"series_id,timestamp,channel,value, Value \n"
        b"A,2026-01-01T00:00:00,c1,1.0,2.0\n"
    )
    with pytest.raises(ValidationError) as excinfo:
        read_long_csv_bytes(payload)
    assert excinfo.value.code == "DUPLICATE_COLUMNS"


def test_missing_required_header_is_named_before_dataframe_validation() -> None:
    payload = b"series_id,timestamp,value\nA,2026-01-01T00:00:00,1.0\n"
    with pytest.raises(ValidationError) as excinfo:
        read_long_csv_bytes(payload)
    assert excinfo.value.code == "MISSING_COLUMNS"
    assert excinfo.value.details["missing"] == ["channel"]


def test_utf8_bom_is_accepted() -> None:
    frame = read_long_csv_bytes(b"\xef\xbb\xbf" + _valid_payload())
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["series_id", "timestamp", "channel", "value"]
