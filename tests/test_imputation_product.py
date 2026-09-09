from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conftest import make_long_frame, set_value
from moment_pipeline import impute, masked_point_metrics, to_windows
from moment_pipeline.model import LoadedMoment, ModelIdentity

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
    task="reconstruction",
    device="cpu",
    dtype="float32",
)


class ZeroReconstructionPipeline:
    def reconstruct(self, *, x_enc, input_mask, mask):
        return dataclasses.make_dataclass("Out", [("reconstruction", torch.Tensor)])(
            torch.zeros_like(x_enc)
        )


def stub_model() -> LoadedMoment:
    return LoadedMoment(
        pipeline=ZeroReconstructionPipeline(),
        identity=IDENTITY,
        snapshot=None,
        proof={"weight_file": "model.safetensors", "encoder_tensors_checked": []},
    )


def test_default_imputed_series_never_overwrites_an_observed_source_value() -> None:
    frame = make_long_frame(n_points=512, seed=2)
    windows = to_windows(set_value(frame, "A", "c1", 13, np.nan))
    result = impute(windows, stub_model(), warmup=False)

    observed = windows.point_mask == 1
    missing = windows.point_mask == 0

    assert np.array_equal(result.imputed[observed], windows.x_enc[observed])
    assert np.array_equal(result.imputed[missing], result.reconstruction[missing])

    # The missing point hides its whole 8-step patch from MOMENT, but observed
    # neighbours in that patch still remain unchanged in the default imputed product.
    assert result.model_mask[0, 8:16].sum() == 0
    for position in list(range(8, 13)) + [14, 15]:
        assert result.imputed[0, 0, position] == windows.x_enc[0, 0, position]


def test_artificial_mask_replaces_only_requested_points_not_patch_expansion_neighbours() -> None:
    windows = to_windows(make_long_frame(n_points=512, seed=3))
    requested = np.ones((1, 512), dtype=np.float32)
    requested[:, 16:24] = 0.0  # exactly one MOMENT patch

    result = impute(windows, stub_model(), mask=requested, warmup=False)

    assert np.array_equal(result.imputed[0, 0, 16:24], np.zeros(8, dtype=np.float32))
    assert result.imputed[0, 0, 15] == windows.x_enc[0, 0, 15]
    assert result.imputed[0, 0, 24] == windows.x_enc[0, 0, 24]


def test_masked_point_metrics_score_only_requested_source_observed_ground_truth() -> None:
    windows = to_windows(make_long_frame(n_points=512, seed=4))
    requested = np.ones((1, 512), dtype=np.float32)
    requested[:, 24:32] = 0.0
    result = impute(windows, stub_model(), mask=requested, warmup=False)

    metrics = masked_point_metrics(result)
    truth = windows.x_enc[0, 0, 24:32]

    assert metrics.n == 8
    assert metrics.mae == pytest.approx(float(np.mean(np.abs(truth))))
    assert metrics.rmse == pytest.approx(float(np.sqrt(np.mean(np.square(truth)))))


def test_source_missing_points_are_not_counted_as_imputation_ground_truth() -> None:
    frame = make_long_frame(n_points=512, seed=5)
    frame = set_value(frame, "A", "c1", 27, np.nan)
    windows = to_windows(frame)
    requested = np.ones((1, 512), dtype=np.float32)
    requested[:, 24:32] = 0.0
    result = impute(windows, stub_model(), mask=requested, warmup=False)

    metrics = masked_point_metrics(result)
    assert metrics.n == 7


def test_imputation_frame_preserves_ground_truth_for_artificial_mask_audit() -> None:
    windows = to_windows(make_long_frame(n_points=512, seed=6))
    requested = np.ones((1, 512), dtype=np.float32)
    requested[:, 32:40] = 0.0
    result = impute(windows, stub_model(), mask=requested, warmup=False)

    frame = result.to_frame()
    hidden = frame[frame["requested_hidden"]]

    assert len(frame) == 512
    assert len(hidden) == 8
    assert hidden["source_observed"].all()
    assert hidden["original_value"].notna().all()
    assert np.array_equal(hidden["imputed_value"].to_numpy(), np.zeros(8))
    assert set(
        [
            "series_id",
            "window_id",
            "timestamp",
            "channel",
            "original_value",
            "imputed_value",
            "reconstruction",
            "source_observed",
            "source_missing",
            "requested_hidden",
            "model_hidden",
        ]
    ).issubset(frame.columns)


def test_masked_point_metrics_refuses_when_no_ground_truth_was_deliberately_hidden() -> None:
    windows = to_windows(make_long_frame(n_points=512, seed=7))
    result = impute(windows, stub_model(), warmup=False)
    with pytest.raises(ValueError, match="no deliberately hidden"):
        masked_point_metrics(result)
