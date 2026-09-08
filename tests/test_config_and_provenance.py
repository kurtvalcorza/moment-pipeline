"""Config guards and the exported provenance shape (no model load required)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from moment_pipeline.canonical import to_windows
from moment_pipeline.config import ConfigError, MomentConfig, ResourceLimits
from moment_pipeline.embedding import EmbeddingResult
from moment_pipeline.imputation import ReconstructionResult
from moment_pipeline.model import (
    MOMENTFM_SOURCE_COMMIT,
    PINNED_REVISION,
    LoadedMoment,
    ModelIdentity,
)
from moment_pipeline.provenance import build_provenance, measure_latency

IDENTITY = ModelIdentity(
    name="AutonLab/MOMENT-1-base",
    revision=PINNED_REVISION,
    config_sha256="f1c66c2bb845229c0ed27a1600dbcc956b85ab21f9e5fd8a1663e6641bed7755",
    weights_sha256="1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825",
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


def fake_model(task: str = "embedding") -> LoadedMoment:
    identity = dataclasses.replace(IDENTITY, task=task)
    return LoadedMoment(
        pipeline=None,
        identity=identity,
        snapshot=None,
        proof={"weight_file": "model.safetensors", "encoder_tensors_checked": ["encoder.a"]},
    )


# --- config -------------------------------------------------------------------------


def test_defaults_are_the_pinned_geometry():
    config = MomentConfig()
    assert (config.sequence_length, config.patch_length, config.patch_stride) == (512, 8, 8)
    assert config.n_patches == 64
    assert config.prefill_value == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "forecasting"},
        {"task": "classification"},
        {"sequence_length": 1024},
        {"patch_length": 16},
        {"patch_stride": 4},
        {"batch_size": 0},
        {"device": "tpu"},
        {"dtype": "int8"},
        {"prefill_value": float("nan")},
    ],
)
def test_unsupported_config_rejected(kwargs):
    with pytest.raises(ConfigError):
        MomentConfig(**kwargs)


def test_resource_limits_must_be_positive():
    with pytest.raises(ConfigError):
        ResourceLimits(max_series=0)


def test_resolved_device_is_never_a_hard_cuda_requirement():
    assert MomentConfig(device="cpu").resolved_device() == "cpu"
    assert MomentConfig(device="auto").resolved_device() in {"cpu", "cuda"}


# --- provenance ---------------------------------------------------------------------


def test_embedding_provenance_records_reduction_and_supply_chain(clean_frame):
    windows = to_windows(clean_frame)
    result = EmbeddingResult(
        embeddings=np.zeros((1, 768), np.float32),
        series_ids=windows.series_ids,
        window_ids=windows.window_ids,
        d_model=768,
        latency_seconds=0.25,
    )
    record = build_provenance(fake_model("embedding"), windows, result)

    assert record["model"]["revision"] == PINNED_REVISION
    assert record["model"]["weight_file_loaded"] == "model.safetensors"
    assert record["model"]["d_model_effective"] == 768
    assert record["runtime"]["momentfm_source"]["commit"] == MOMENTFM_SOURCE_COMMIT
    assert record["runtime"]["torch"] is not None
    assert record["runtime"]["device"] == "cpu"
    assert record["inference"]["task"] == "embedding"
    assert record["inference"]["reduction"] == "mean"
    assert "averaged" in record["inference"]["channel_policy"]
    assert record["inference"]["n_windows"] == 1
    assert record["inference"]["latency_seconds"] == pytest.approx(0.25)


def test_reconstruction_provenance_records_both_fractions(clean_frame):
    windows = to_windows(clean_frame)
    result = ReconstructionResult(
        reconstruction=np.zeros((1, 1, 512), np.float32),
        point_mask=windows.point_mask,
        model_mask=windows.patch_quantized_mask(),
        patch_mask=windows.patch_mask,
        masked_point_fraction=1 / 512,
        masked_patch_fraction=1 / 64,
        masked_point_count=1,
        masked_patch_count=1,
        series_ids=windows.series_ids,
        window_ids=windows.window_ids,
        channels=windows.channels,
        latency_seconds=0.5,
    )
    record = build_provenance(fake_model("reconstruction"), windows, result)
    assert record["inference"]["masked_point_fraction"] == pytest.approx(1 / 512)
    assert record["inference"]["masked_patch_fraction"] == pytest.approx(1 / 64)
    assert "mask=None is never passed" in record["inference"]["mask_policy"]


def test_measure_latency_discards_the_warm_up():
    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    result, seconds = measure_latency(fn, warmup=2)
    assert result == 3
    assert len(calls) == 3
    assert seconds >= 0.0
