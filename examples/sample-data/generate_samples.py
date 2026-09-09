"""Generate deterministic synthetic samples for MOMENT v1 tutorials."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def build_samples() -> dict[str, pd.DataFrame]:
    n = 256
    timestamps = pd.date_range("2026-01-01", periods=n, freq="15min")
    step = np.arange(n, dtype=float)

    rows: list[tuple[str, pd.Timestamp, str, float]] = []
    for channel, base, amplitude, period, phase, trend in (
        ("vibration", 1.5, 0.35, 32.0, 0.0, 0.0008),
        ("temperature", 28.0, 2.5, 96.0, 9.0, 0.0015),
    ):
        values = (
            base
            + trend * step
            + amplitude * np.sin(2.0 * np.pi * (step + phase) / period)
            + 0.08 * np.cos(2.0 * np.pi * step / 16.0)
        )
        rows.extend(("A", stamp, channel, float(value)) for stamp, value in zip(timestamps, values))

    clean = pd.DataFrame(rows, columns=["series_id", "timestamp", "channel", "value"])
    anomaly = clean.copy()
    labels = pd.DataFrame(
        {"series_id": "A", "timestamp": timestamps, "is_injected_anomaly": False}
    )

    injected = {184: 2.8, 201: -3.2, 233: 3.6}
    for index, delta in injected.items():
        labels.loc[index, "is_injected_anomaly"] = True
        selector = (anomaly["channel"] == "vibration") & (
            anomaly["timestamp"] == timestamps[index]
        )
        anomaly.loc[selector, "value"] = anomaly.loc[selector, "value"] + delta

    return {
        "moment_clean.csv": clean,
        "moment_anomaly.csv": anomaly,
        "moment_anomaly_labels.csv": labels,
    }


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        float_format="%.6f",
    ).encode("utf-8")


def generate(root: Path = ROOT) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, frame in build_samples().items():
        payload = csv_bytes(frame)
        (root / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    manifest = "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items()))
    (root / "SHA256SUMS").write_text(manifest, encoding="utf-8", newline="\n")
    return digests


if __name__ == "__main__":
    generate()
