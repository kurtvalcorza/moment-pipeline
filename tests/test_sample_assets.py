from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "examples" / "sample-data" / "generate_samples.py"
MANIFEST = ROOT / "examples" / "sample-data" / "SHA256SUMS"


def _load_generator():
    spec = importlib.util.spec_from_file_location("moment_sample_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    return entries


def test_generated_samples_match_the_committed_sha256_manifest() -> None:
    generator = _load_generator()
    expected = _manifest()
    samples = generator.build_samples()

    assert set(samples) == set(expected)
    actual = {
        name: hashlib.sha256(generator.csv_bytes(frame)).hexdigest()
        for name, frame in samples.items()
    }
    assert actual == expected


def test_anomaly_labels_match_the_injected_sample_timestamps() -> None:
    generator = _load_generator()
    samples = generator.build_samples()
    anomaly = samples["moment_anomaly.csv"]
    labels = samples["moment_anomaly_labels.csv"]

    injected = labels.loc[labels["is_injected_anomaly"], "timestamp"]
    assert len(injected) == 3
    vibration = anomaly[anomaly["channel"] == "vibration"]
    assert set(injected) <= set(vibration["timestamp"])
