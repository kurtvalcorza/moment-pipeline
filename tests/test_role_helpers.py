"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21).

No weights and no network: the model is the stub from test_anomaly.py and the task functions
run against it, so every number here is this repository's arithmetic.
"""

# ruff: noqa: E501  -- offline fixtures and assertions are kept on single lines for readability
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_long_frame, set_value
from moment_pipeline import (
    INPUT_SCHEMA,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    EmbeddingResult,
    MomentConfig,
    ValidationError,
    evaluation_report,
    impute,
    score_anomalies,
    to_windows,
    top_k_recall,
    validate_inputs,
)
from moment_pipeline.config import PATCH_LENGTH, SEQUENCE_LENGTH
from test_anomaly import stub_model

# --------------------------------------------------------------------------
# validate_inputs
# --------------------------------------------------------------------------


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    frame = make_long_frame(n_points=600, series=("A", "B"), channels=("c1", "c2"))
    manifest = validate_inputs(frame, MomentConfig(task="reconstruction"), names=["first", "second"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["sequence_length"] == SEQUENCE_LENGTH
    assert manifest["schema"]["patch_length"] == PATCH_LENGTH
    assert manifest["schema"]["max_windows"] == MomentConfig().limits.max_windows
    assert [entry["id"] for entry in manifest["inputs"]] == ["first", "second"]
    assert [entry["series_id"] for entry in manifest["inputs"]] == ["A", "B"]
    assert all(entry["truncated"] and not entry["padded"] for entry in manifest["inputs"])
    assert all(entry["valid_positions"] == SEQUENCE_LENGTH for entry in manifest["inputs"])
    assert manifest["task"] == "reconstruction"
    assert manifest["n_rows"] == 600 * 2 * 2
    assert (manifest["n_series"], manifest["n_channels"], manifest["channels"]) == (2, 2, ["c1", "c2"])
    assert manifest["window_shape"] == [2, 2, SEQUENCE_LENGTH]
    assert manifest["masked_point_fraction"] == 0.0
    assert (manifest["model_id"], manifest["model_revision"]) == (PINNED_MODEL_ID, PINNED_REVISION)


def test_validate_inputs_default_ids_are_window_ids_and_padding_is_reported() -> None:
    manifest = validate_inputs(make_long_frame(n_points=100))
    assert len(manifest["inputs"]) == 1
    assert manifest["inputs"][0]["id"] == str(to_windows(make_long_frame(n_points=100)).window_ids[0])
    assert manifest["inputs"][0]["padded"] is True
    assert manifest["inputs"][0]["valid_positions"] == 100


def test_validate_inputs_rejects_exactly_like_to_windows() -> None:
    frame = make_long_frame(n_points=64)
    broken = frame.drop(columns=["channel"])
    with pytest.raises(ValidationError) as via_helper:
        validate_inputs(broken)
    with pytest.raises(ValidationError) as via_task:
        to_windows(broken)
    assert via_helper.value.code == via_task.value.code

    empty_channel = frame.copy()
    empty_channel["value"] = np.nan
    with pytest.raises(ValidationError) as via_helper:
        validate_inputs(empty_channel)
    with pytest.raises(ValidationError) as via_task:
        to_windows(empty_channel)
    assert via_helper.value.code == via_task.value.code == "EMPTY_CHANNEL"


def test_validate_inputs_names_must_match_the_windows() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_inputs(make_long_frame(series=("A", "B")), names=["only-one"])
    assert exc.value.code == "NAMES_LENGTH_MISMATCH"


# --------------------------------------------------------------------------
# evaluation_report
# --------------------------------------------------------------------------


def _embedding_result() -> EmbeddingResult:
    return EmbeddingResult(
        embeddings=np.zeros((2, 768), dtype=np.float32),
        series_ids=("A", "B"),
        window_ids=("A", "B"),
        d_model=768,
        latency_seconds=0.0,
    )


def test_evaluation_report_embeddings_are_always_not_measurable() -> None:
    model = stub_model()
    report = evaluation_report(_embedding_result(), model=model)
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == [] and report["baselines"] == []
    assert report["task"] == "embedding"
    assert report["embedding_shape"] == [2, 768]
    assert "labelled" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (model.identity.name, model.identity.revision)


def test_evaluation_report_imputation_sample_sanity_with_an_artificial_mask() -> None:
    windows = to_windows(make_long_frame(n_points=512, seed=5))
    visible = np.ones_like(windows.input_mask, dtype=np.float32)
    visible[0, 504:512] = 0.0
    model = stub_model((1.0,))
    result = impute(windows, model, mask=visible, warmup=False)
    report = evaluation_report(result, model=model, sample_kind="synthetic", baseline={"mae": 0.25, "rmse": 0.3})
    assert report["verdict"] == "sample-sanity"
    assert report["task"] == "imputation"
    assert report["n_scored"] == 8
    by_metric = {m["metric"]: m["value"] for m in report["metrics"]}
    assert set(m["id"] for m in report["metrics"]) == {"masked_point_metrics"}
    assert by_metric["mae"] == pytest.approx(1.0)
    assert by_metric["rmse"] == pytest.approx(1.0)
    assert report["baselines"][0]["id"] == "linear_interpolation"
    assert {m["metric"]: m["value"] for m in report["baselines"][0]["metrics"]} == {"mae": 0.25, "rmse": 0.3}


def test_evaluation_report_imputation_not_measurable_without_a_deliberate_mask() -> None:
    windows = to_windows(make_long_frame(n_points=512, seed=5))
    result = impute(windows, stub_model(), warmup=False)
    report = evaluation_report(result)
    assert report["verdict"] == "not-measurable"
    assert "artificial mask" in report["needs"]
    assert report["model_id"] is None


def test_evaluation_report_anomaly_sample_sanity_with_labels_and_not_measurable_without() -> None:
    frame = make_long_frame(n_points=512, seed=5)
    spiked = set_value(frame, "A", "c1", 100, 50.0)
    spiked = set_value(spiked, "A", "c1", 300, -50.0)
    windows = to_windows(spiked)
    model = stub_model((0.0,))  # reconstruction == input + 0 -> residual 0 everywhere...
    result = score_anomalies(windows, model, loss="mae", channel_aggregation="none", warmup=False)
    stamps = pd.date_range("2026-01-01", periods=512, freq="h")
    labels = pd.DataFrame({"series_id": ["A", "A"], "timestamp": [stamps[100], stamps[300]], "is_injected_anomaly": [True, True]})
    report = evaluation_report(result, labels, model=model, channel="c1")
    assert report["verdict"] == "sample-sanity"
    assert report["task"] == "anomaly-scoring"
    (metric,) = report["metrics"]
    assert (metric["id"], metric["k"], metric["channel"]) == ("top_k_recall", 2, "c1")
    assert 0.0 <= metric["value"] <= 1.0
    assert metric["value"] == top_k_recall(result.to_frame(), labels, channel="c1")["value"]
    unlabelled = evaluation_report(result, model=model)
    assert unlabelled["verdict"] == "not-measurable"
    assert "top_k_recall" in unlabelled["needs"]
    assert unlabelled["threshold_policy"].startswith("none applied")


def test_top_k_recall_ranks_only_scored_positions_and_refuses_unlabelled_frames() -> None:
    windows = to_windows(make_long_frame(n_points=512, seed=5))
    result = score_anomalies(windows, stub_model((1.0,)), warmup=False)
    scores = result.to_frame()
    labels = pd.DataFrame({"series_id": ["A"], "timestamp": [pd.Timestamp("2030-01-01")], "is_injected_anomaly": [True]})
    with pytest.raises(ValueError, match="mark no anomaly"):
        top_k_recall(scores, labels, channel="c1")
    first = scores[scores["scored"]].iloc[0]
    hit = pd.DataFrame({"series_id": [first["series_id"]], "timestamp": [first["timestamp"]], "is_injected_anomaly": [True]})
    out = top_k_recall(scores, hit, channel="c1")
    assert out["k"] == 1 and out["value"] in (0.0, 1.0)
    assert list(out["ranked"]["rank"]) == list(range(1, len(out["ranked"]) + 1))
    assert out["injected_ranks"][0]["rank"] >= 1


def test_evaluation_report_refuses_unknown_result_types() -> None:
    with pytest.raises(TypeError):
        evaluation_report(object())  # type: ignore[arg-type]
