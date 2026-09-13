"""Role-stage helpers shared by the three tutorials (DIMER NOTEBOOK_SPEC 1.1 DAT24 / EVAL21).

`validate_inputs` is the public validation stage: it routes the long frame through exactly
the calls the task functions rely on (`validate_long_frame` then `to_windows`), so it raises
exactly what canonicalisation would raise, and reports what was proven as an **input
manifest**. `evaluation_report` is the public evaluation stage: it adds no metric of its own —
imputation evidence comes from `masked_point_metrics`, anomaly evidence from `top_k_recall`,
and embeddings are representations with no correctness metric — and always produces a report,
saying what would make the task measurable when nothing is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .anomaly import AnomalyResult, top_k_recall
from .canonical import WindowSet, to_windows
from .config import PATCH_LENGTH, SEQUENCE_LENGTH, MomentConfig
from .embedding import EmbeddingResult
from .imputation import ReconstructionResult, masked_point_metrics
from .model import PINNED_MODEL_ID, PINNED_REVISION, LoadedMoment, ModelIdentity
from .validation import REQUIRED_COLUMNS, ValidationError, validate_long_frame

__all__ = ["INPUT_SCHEMA", "validate_inputs", "evaluation_report"]

_DEFAULT_LIMITS = MomentConfig().limits

#: The input contract and every named ceiling, in one readable structure.
INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "long-format pandas DataFrame with columns "
        + ", ".join(REQUIRED_COLUMNS)
        + "; one row per (series_id, channel, timestamp)"
    ),
    "timestamps": (
        "parseable and strictly increasing per (series_id, channel); irregular spacing is "
        "reported per series and refused when strict_frequency is set"
    ),
    "values": "numeric; NaN marks a source-missing point, which never reaches the model",
    "sequence_length": SEQUENCE_LENGTH,
    "patch_length": PATCH_LENGTH,
    "max_rows": _DEFAULT_LIMITS.max_rows,
    "max_series": _DEFAULT_LIMITS.max_series,
    "max_channels": _DEFAULT_LIMITS.max_channels,
    "max_windows": _DEFAULT_LIMITS.max_windows,
    "canonicalisation": (
        "one window per series: the final sequence_length timestamps are kept (longer series "
        "are truncated, shorter ones left-padded and the padding excluded by the input mask)"
    ),
}


def validate_inputs(
    df: pd.DataFrame,
    config: MomentConfig | None = None,
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-window observations, verdict).

    Rejection is reported by raising exactly as the task path would — `validate_long_frame`
    followed by `to_windows`, the same two calls every tutorial and task function makes — so a
    caller that wants the finding recorded catches `ValidationError` and stores `str(exc)`
    under `findings`.
    """
    config = config or MomentConfig()
    report, frame = validate_long_frame(df, config)
    windows: WindowSet = to_windows(frame, config, report=report, frame=frame)
    if names is not None and len(names) != windows.n_windows:
        raise ValidationError(
            "NAMES_LENGTH_MISMATCH",
            f"names must have one entry per window: got {len(names)} for {windows.n_windows}",
            {"n_names": len(names), "n_windows": windows.n_windows},
        )
    inputs: list[dict[str, Any]] = []
    for index, window_id in enumerate(windows.window_ids):
        valid = int(windows.input_mask[index].sum())
        inputs.append(
            {
                "id": names[index] if names else str(window_id),
                "series_id": str(windows.series_ids[index]),
                "valid_positions": valid,
                "padded": bool(windows.padded[index]),
                "truncated": bool(windows.truncated[index]),
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": inputs,
        "task": config.task,
        "n_rows": int(report.n_rows),
        "n_series": len(report.series_ids),
        "n_channels": len(report.channels),
        "channels": list(windows.channels),
        "irregular_series": list(report.irregular_series),
        "window_shape": list(windows.x_enc.shape),
        "masked_point_fraction": float(windows.masked_point_fraction),
        "masked_patch_fraction": float(windows.masked_patch_fraction),
        "verdict": "accepted",
        "findings": [],
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_REVISION,
    }


def _identity_block(model: LoadedMoment | ModelIdentity | None) -> dict[str, Any]:
    identity = getattr(model, "identity", model)
    return {
        "model_id": getattr(identity, "name", None),
        "model_revision": getattr(identity, "revision", None),
    }


def _not_measurable(base: dict[str, Any], reason: str, needs: str) -> dict[str, Any]:
    return {
        **base,
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": reason,
        "needs": needs,
    }


def evaluation_report(
    result: EmbeddingResult | ReconstructionResult | AnomalyResult,
    labels: pd.DataFrame | None = None,
    *,
    model: LoadedMoment | ModelIdentity | None = None,
    sample_kind: str = "synthetic",
    channel: str | None = None,
    baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    - `EmbeddingResult`: embeddings are representations, not predictions — the verdict is
      always `not-measurable` and the report says what downstream task labels would score them.
    - `ReconstructionResult`: when the result carries a deliberate artificial mask,
      `masked_point_metrics` (MAE/RMSE on the deliberately hidden, source-observed points) is
      reported with the verdict `sample-sanity`; an optional tutorial `baseline`
      (`{"mae": ..., "rmse": ...}`, e.g. linear interpolation) is carried under `baselines`.
      Without a deliberate mask nothing has known withheld truth and the verdict is
      `not-measurable`.
    - `AnomalyResult`: with `labels` (`series_id`, `timestamp`, `is_injected_anomaly`) the
      `top_k_recall` of the raw-score ranking on `channel` is reported with the verdict
      `sample-sanity`; without labels the verdict is `not-measurable`.
    Everything the metric helpers would raise is raised here unchanged.
    """
    base: dict[str, Any] = {"sample_kind": sample_kind, **_identity_block(model)}
    if isinstance(result, EmbeddingResult):
        return _not_measurable(
            {
                **base,
                "task": "embedding",
                "score_semantics": (
                    f"{result.d_model}-dimensional {result.reduction}-pooled window "
                    "representations; not predictions, no correctness metric exists"
                ),
                "n_windows": len(result.window_ids),
                "embedding_shape": list(result.embeddings.shape),
            },
            "embeddings are representations, not predictions",
            "a downstream labelled task (e.g. window classes or retrieval pairs) scored by a "
            "probe or nearest-neighbour evaluation on held-out labels",
        )
    if isinstance(result, ReconstructionResult):
        report_base = {
            **base,
            "task": "imputation",
            "score_semantics": (
                "reconstructed values at deliberately hidden, source-observed points; "
                "source-missing points have no truth"
            ),
            "n_windows": len(result.window_ids),
            "masked_point_fraction": float(result.masked_point_fraction),
            "model_masked_point_fraction": float(result.model_masked_point_fraction),
            "masked_patch_fraction": float(result.masked_patch_fraction),
        }
        requested = result.requested_visible_mask
        if requested is None or not np.any(requested == 0):
            return _not_measurable(
                report_base,
                "no deliberately hidden source-observed points exist in this result",
                "an artificial mask over source-observed positions (pass mask= to impute()) so "
                "masked_point_metrics can score the reconstruction against withheld truth",
            )
        metrics = masked_point_metrics(result)
        estimation = "single deterministic artificial patch holdout on one sample; no dispersion"
        baselines = []
        if baseline is not None:
            baselines.append(
                {
                    "id": "linear_interpolation",
                    "metrics": [
                        {"id": "linear_interpolation", "metric": key, "value": float(value)}
                        for key, value in baseline.items()
                    ],
                }
            )
        return {
            **report_base,
            "n_scored": metrics.n,
            "metrics": [
                {
                    "id": "masked_point_metrics",
                    "metric": "mae",
                    "value": metrics.mae,
                    "estimation": estimation,
                },
                {
                    "id": "masked_point_metrics",
                    "metric": "rmse",
                    "value": metrics.rmse,
                    "estimation": estimation,
                },
            ],
            "baselines": baselines,
            "verdict": "sample-sanity",
            "reason": (
                f"{metrics.n} deliberately hidden points from one artificial holdout; "
                "not a benchmark"
            ),
            "needs": (
                "many artificial holdouts over held-out series of the deployment domain, compared "
                "against domain baselines, for any generalisable imputation claim"
            ),
        }
    if isinstance(result, AnomalyResult):
        report_base = {
            **base,
            "task": "anomaly-scoring",
            "score_semantics": (
                f"raw {result.loss} reconstruction residual per scored position; higher means "
                "more anomaly evidence; no threshold is shipped"
            ),
            "n_windows": len(result.window_ids),
            "scored_point_fraction": float(result.scored_point_fraction),
            "threshold_policy": result.threshold_policy,
        }
        if labels is None:
            return _not_measurable(
                report_base,
                "no anomaly labels were supplied for the scored positions",
                "labelled anomalies (series_id, timestamp, is_injected_anomaly) for the scored "
                "positions so top_k_recall can be computed on the raw-score ranking, or an "
                "operator-calibrated threshold from the deployment domain",
            )
        recall = top_k_recall(result.to_frame(), labels, channel=channel)
        return {
            **report_base,
            "metrics": [
                {
                    "id": "top_k_recall",
                    "k": recall["k"],
                    "channel": recall["channel"],
                    "value": recall["value"],
                    "estimation": "single labelled sample; no dispersion estimate",
                }
            ],
            "baselines": [],
            "verdict": "sample-sanity",
            "reason": f"{recall['k']} labelled positions on one sample; not a benchmark",
            "needs": (
                "labelled anomalies from the deployment domain scored over many series, plus an "
                "operator-chosen threshold, for any generalisable detection claim"
            ),
        }
    raise TypeError(f"unsupported result type {type(result).__name__}")
