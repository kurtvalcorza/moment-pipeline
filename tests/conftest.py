from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_long_frame(
    n_points: int = 512,
    series: tuple[str, ...] = ("A",),
    channels: tuple[str, ...] = ("c1",),
    start: str = "2026-01-01",
    freq: str = "h",
    seed: int = 0,
) -> pd.DataFrame:
    """A clean, regular long-format frame: one row per (series, channel, timestamp)."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start=start, periods=n_points, freq=freq)
    rows = []
    for series_id in series:
        for channel in channels:
            rows.append(
                pd.DataFrame(
                    {
                        "series_id": series_id,
                        "timestamp": stamps,
                        "channel": channel,
                        "value": rng.normal(size=n_points),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def set_value(df: pd.DataFrame, series_id: str, channel: str, index: int, value) -> pd.DataFrame:
    """Replace the value at positional `index` within one (series, channel) track."""
    out = df.copy()
    selector = (out["series_id"] == series_id) & (out["channel"] == channel)
    positions = np.flatnonzero(selector.to_numpy())
    out.loc[out.index[positions[index]], "value"] = value
    return out


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    return make_long_frame()
