"""Common validation contract for long-format BYOD input.

Implements rules 1-12 of the RFC "Common validation contract". Every rule raises a
`ValidationError` carrying a stable machine-readable `code`, except rule 7
(frequency irregularity), which the RFC says must be *surfaced* rather than
rejected — it is reported as a flag and only escalated to an error when
`MomentConfig.strict_frequency` is set.

This module depends only on numpy/pandas: the unit suite around it runs with no
network and no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import MomentConfig

REQUIRED_COLUMNS: tuple[str, ...] = ("series_id", "timestamp", "channel", "value")


class ValidationError(Exception):
    """A violated input contract rule.

    Attributes:
        code: stable identifier, e.g. ``DUPLICATE_ROWS``. Safe to branch on.
        message: human-readable explanation.
        details: structured context (offending ids/positions). Never contains raw
            user payload values beyond the minimum needed to locate the problem.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details or {}
        super().__init__(f"[{code}] {message}")


@dataclass(frozen=True)
class SeriesFrequency:
    """Per-series timestamp regularity (RFC rule 7 — surfaced, not silently fixed)."""

    series_id: str
    n_timestamps: int
    regular: bool
    distinct_deltas: tuple[str, ...]


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of a successful validation pass."""

    n_rows: int
    series_ids: tuple[str, ...]
    channels: tuple[str, ...]
    frequencies: tuple[SeriesFrequency, ...]
    fully_missing_series_channels: tuple[tuple[str, str], ...]

    @property
    def irregular_series(self) -> tuple[str, ...]:
        return tuple(f.series_id for f in self.frequencies if not f.regular)

    @property
    def has_irregular_frequency(self) -> bool:
        return bool(self.irregular_series)


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(
            "MISSING_COLUMNS",
            f"long-format input requires columns {list(REQUIRED_COLUMNS)}; missing {missing}",
            {"missing": missing, "present": list(map(str, df.columns))},
        )


def _check_ids(df: pd.DataFrame) -> None:
    for column in ("series_id", "channel"):
        null_positions = np.flatnonzero(df[column].isna().to_numpy())
        if null_positions.size:
            raise ValidationError(
                "NULL_ID",
                f"{column} must be non-null; {null_positions.size} null value(s) found",
                {"column": column, "row_positions": null_positions[:20].tolist()},
            )
        empty = np.flatnonzero(
            df[column].astype("string").str.strip().fillna("").eq("").to_numpy()
        )
        if empty.size:
            raise ValidationError(
                "NULL_ID",
                f"{column} must be non-empty; {empty.size} blank value(s) found",
                {"column": column, "row_positions": empty[:20].tolist()},
            )


def _parse_timestamps(df: pd.DataFrame) -> pd.Series:
    raw = df["timestamp"]
    null_positions = np.flatnonzero(raw.isna().to_numpy())
    if null_positions.size:
        raise ValidationError(
            "NULL_TIMESTAMP",
            f"timestamp must be non-null; {null_positions.size} null value(s) found",
            {"row_positions": null_positions[:20].tolist()},
        )
    if pd.api.types.is_datetime64_any_dtype(raw):
        parsed = raw.astype("datetime64[ns]")
    else:
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    bad = np.flatnonzero(parsed.isna().to_numpy())
    if bad.size:
        raise ValidationError(
            "UNPARSEABLE_TIMESTAMP",
            f"{bad.size} timestamp value(s) could not be parsed",
            {"row_positions": bad[:20].tolist()},
        )
    return parsed


def _coerce_values(df: pd.DataFrame) -> pd.Series:
    """Strict numeric policy: nothing is silently turned into a missing value.

    A value is acceptable if it is already missing (NaN/None -> missing, carried by
    the mask) or parses as a finite float. Anything that would *become* NaN through
    coercion, and any +/-inf, is an error.
    """
    raw = df["value"]
    originally_missing = raw.isna().to_numpy()
    coerced = pd.to_numeric(raw, errors="coerce")
    newly_missing = np.flatnonzero(coerced.isna().to_numpy() & ~originally_missing)
    if newly_missing.size:
        raise ValidationError(
            "NON_NUMERIC_VALUE",
            f"{newly_missing.size} value(s) are not numeric under the strict policy",
            {"row_positions": newly_missing[:20].tolist()},
        )
    numeric = coerced.astype("float64")
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    non_finite = np.flatnonzero(~np.isnan(values) & ~np.isfinite(values))
    if non_finite.size:
        raise ValidationError(
            "NON_FINITE_VALUE",
            f"{non_finite.size} value(s) are +/-inf; use a missing value instead",
            {"row_positions": non_finite[:20].tolist()},
        )
    return numeric


def _check_duplicates(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(subset=["series_id", "channel", "timestamp"], keep=False)
    if bool(duplicated.any()):
        offenders = (
            frame.loc[duplicated, ["series_id", "channel", "timestamp"]]
            .astype(str)
            .drop_duplicates()
            .head(20)
            .to_dict("records")
        )
        raise ValidationError(
            "DUPLICATE_ROWS",
            f"{int(duplicated.sum())} row(s) duplicate a (series_id, channel, timestamp) key",
            {"examples": offenders},
        )


def _check_limits(frame: pd.DataFrame, config: MomentConfig) -> None:
    limits = config.limits
    n_series = frame["series_id"].nunique()
    n_channels = frame["channel"].nunique()
    if n_series > limits.max_series:
        raise ValidationError(
            "LIMIT_EXCEEDED",
            f"{n_series} series exceeds max_series={limits.max_series}",
            {"limit": "max_series", "actual": int(n_series), "allowed": limits.max_series},
        )
    if n_channels > limits.max_channels:
        raise ValidationError(
            "LIMIT_EXCEEDED",
            f"{n_channels} channels exceeds max_channels={limits.max_channels}",
            {"limit": "max_channels", "actual": int(n_channels), "allowed": limits.max_channels},
        )


def normalize_long_frame(df: pd.DataFrame, config: MomentConfig | None = None) -> pd.DataFrame:
    """Validate and return a normalized copy: string ids, datetime64 stamps, float values."""
    config = config or MomentConfig()
    _require_columns(df)
    if len(df) == 0:
        raise ValidationError("EMPTY_INPUT", "input frame has no rows", {"n_rows": 0})
    if len(df) > config.limits.max_rows:
        raise ValidationError(
            "LIMIT_EXCEEDED",
            f"{len(df)} rows exceeds max_rows={config.limits.max_rows}",
            {"limit": "max_rows", "actual": len(df), "allowed": config.limits.max_rows},
        )
    df = df.reset_index(drop=True)
    _check_ids(df)
    frame = pd.DataFrame(
        {
            "series_id": df["series_id"].astype("string").astype("object").map(str),
            "channel": df["channel"].astype("string").astype("object").map(str),
            "timestamp": _parse_timestamps(df),
            "value": _coerce_values(df),
        }
    )
    _check_duplicates(frame)
    _check_limits(frame, config)
    return frame


def validate_long_frame(
    df: pd.DataFrame, config: MomentConfig | None = None
) -> tuple[ValidationReport, pd.DataFrame]:
    """Apply the full common validation contract.

    Returns the report and the normalized frame so callers do not re-parse. Raises
    `ValidationError` on the first violated rule.
    """
    config = config or MomentConfig()
    frame = normalize_long_frame(df, config)

    channels = tuple(sorted(frame["channel"].unique().tolist()))
    series_ids = tuple(sorted(frame["series_id"].unique().tolist()))

    observed_per_channel = frame.loc[frame["value"].notna(), "channel"].unique().tolist()
    empty_channels = [c for c in channels if c not in set(observed_per_channel)]
    if empty_channels:
        raise ValidationError(
            "EMPTY_CHANNEL",
            f"channel(s) {empty_channels} contain no observed values anywhere in the input",
            {"channels": empty_channels},
        )

    frequencies: list[SeriesFrequency] = []
    for series_id in series_ids:
        selected = frame.loc[frame["series_id"] == series_id, "timestamp"]
        stamps = np.sort(selected.unique().astype("datetime64[ns]"))
        deltas = np.diff(stamps)
        distinct = np.unique(deltas)
        frequencies.append(
            SeriesFrequency(
                series_id=series_id,
                n_timestamps=int(stamps.size),
                regular=bool(distinct.size <= 1),
                distinct_deltas=tuple(str(d) for d in distinct[:10]),
            )
        )

    fully_missing: list[tuple[str, str]] = []
    observed = frame.loc[frame["value"].notna(), ["series_id", "channel"]]
    observed_pairs = set(map(tuple, observed.drop_duplicates().to_numpy().tolist()))
    for series_id in series_ids:
        for channel in channels:
            if (series_id, channel) not in observed_pairs:
                fully_missing.append((series_id, channel))

    report = ValidationReport(
        n_rows=len(frame),
        series_ids=series_ids,
        channels=channels,
        frequencies=tuple(frequencies),
        fully_missing_series_channels=tuple(fully_missing),
    )

    if config.strict_frequency and report.has_irregular_frequency:
        raise ValidationError(
            "IRREGULAR_FREQUENCY",
            "strict_frequency=True and these series have non-uniform timestamp spacing: "
            f"{list(report.irregular_series)}",
            {"series_ids": list(report.irregular_series)},
        )
    return report, frame
