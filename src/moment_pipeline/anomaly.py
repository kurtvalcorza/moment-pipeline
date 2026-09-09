"""Task 3 — reconstruction-based anomaly **scoring**.

Phase 4. The primitive is the per-element reconstruction residual under a named loss,
shape `(batch, channel, timestep)`, and that is the whole product: a score, not a verdict.

What the RFC fixes, and what this module therefore fixes:

* **Raw scores are mandatory output.** `AnomalyResult.anomaly_score` is the residual
  itself. Nothing here converts it into a boolean.
* **There is no universal binary threshold in v1**, so this module ships none — not as a
  default, not as a keyword argument, not as a constant. Calibration belongs to a caller
  who owns a reference segment, and `MODEL_CARD.md` says why a shipped default would be
  indefensible on an unmasked self-reconstruction residual.
* **Channel aggregation is DIMER-owned, explicit and recorded.** `"none"` is the default
  and the safest mode; `"mean"` and `"max"` collapse the channel axis and nothing else.
* **The score is an unmasked self-reconstruction residual** (RFC M-6): the model
  reconstructs a window it can see, and the residual is scored. It is not a forecast
  residual, and a drift the model reconstructs faithfully scores low by construction.

The scored domain is the sharp edge, and it is narrower than "every position":

1. **Padding is never scored.** Left-padding is not data.
2. **Pre-filled positions are never scored.** `x_enc` there holds `prefill_value`, a
   fabricated number; `|reconstruction - 0.0|` says something about the sentinel, not
   about the series.
3. **Positions the model could not see are never scored.** The mask is patch-quantized,
   so one missing point hides its whole 8-step patch (RFC M-3). The residual at the seven
   observed neighbours is a genuine number — but it is an *imputation* residual, produced
   from a patch the encoder was shown as unobserved, and it is systematically larger than
   a self-reconstruction residual. Mixing the two into one column would put a spike at
   every gap edge and invite the reader to call it an anomaly.

So `scored_mask` is `input_mask AND model_mask`, and every value in `anomaly_score` is the
same quantity. Unscored positions are `NaN` — a deliberate "not defined here" marker, not
a propagated NaN — and are counted by reason on the result. `reconstruction_error` keeps
the raw finite residual at every position for debugging; it is not the export column.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import WindowSet
from .imputation import ReconstructionResult, reconstruct
from .model import LoadedMoment

#: Elementwise losses. Names match the upstream vocabulary of
#: `momentfm.utils.utils.get_anomaly_criterion` ("mae" -> L1Loss, "mse" -> MSELoss), both
#: with `reduction="none"`, which is what makes the score per-element.
ANOMALY_LOSSES = ("mae", "mse")
DEFAULT_LOSS = "mae"

#: Channel aggregation. "none" is the default: it is the only mode that loses nothing.
CHANNEL_AGGREGATIONS = ("none", "mean", "max")
DEFAULT_AGGREGATION = "none"

SCORE_POLICY = (
    "unmasked self-reconstruction residual: the model reconstructs a window it can see in "
    "full and the residual is scored. NOT a forecast residual -- a drift the model "
    "reconstructs faithfully scores low by construction. Scores are raw and uncalibrated; "
    "this pipeline ships no binary threshold, and the residual scale depends on the "
    "series, so a threshold fitted on one series does not transfer to another"
)

SCORED_DOMAIN_POLICY = (
    "scored only where the position is non-padded AND was visible to the model. Padding "
    "is not data; pre-filled positions would score the sentinel rather than the series; "
    "and a position inside a patch the mask hid yields an imputation residual, which is "
    "systematically larger than a self-reconstruction residual and would read as a spike "
    "at every gap edge. Unscored positions are NaN by construction, not by propagation"
)


class AnomalyConfigError(ValueError):
    """An unsupported loss or channel aggregation was requested."""


@dataclass(frozen=True)
class AnomalyResult:
    """Raw reconstruction residuals plus the accounting needed to read them.

    `anomaly_score` is the export column: `NaN` wherever the score is not defined (see
    `SCORED_DOMAIN_POLICY`). `reconstruction_error` is the same residual computed
    everywhere and left finite, for callers debugging the pre-fill or the mask. With
    `channel_aggregation="none"` the two share the shape `(batch, channel, timestep)` and
    differ only in those NaNs.
    """

    reconstruction: np.ndarray
    reconstruction_error: np.ndarray
    anomaly_score: np.ndarray
    scored_mask: np.ndarray
    point_mask: np.ndarray
    model_mask: np.ndarray
    series_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    channels: tuple[str, ...]
    loss: str
    channel_aggregation: str
    latency_seconds: float
    #: Cells the score is defined at, and the two reasons the rest were dropped. Counted
    #: over non-padded (window, channel, position) cells, so the three add up to
    #: `input_mask.sum() * n_channels`.
    scored_point_count: int
    unscored_prefilled_count: int
    unscored_hidden_by_patch_count: int
    scored_point_fraction: float
    masked_point_fraction: float
    masked_patch_fraction: float
    score_policy: str = SCORE_POLICY
    scored_domain_policy: str = SCORED_DOMAIN_POLICY
    #: There is no threshold in v1. The field states the absence so an export cannot be
    #: read as "thresholded with the default".
    threshold_policy: str = "none applied; raw scores only (RFC Task 3)"
    timestamps: tuple[np.ndarray, ...] = field(repr=False, default=())

    @property
    def is_aggregated(self) -> bool:
        return self.channel_aggregation != "none"

    def to_frame(self) -> pd.DataFrame:
        """`series_id, timestamp, channel, reconstruction, reconstruction_error,
        anomaly_score` — the RFC Task 3 output, one row per non-padded position.

        Padded positions are dropped: they carry no timestamp (`NaT`) and belong to no
        series. Unscored-but-real positions are kept with `anomaly_score = NaN` and
        `scored = False`, because silently dropping them would hide exactly the gaps a
        reader needs to see.

        Under a channel aggregation the frame carries `channel = <aggregation name>` and
        **omits `reconstruction` and `reconstruction_error`**: those are per-channel
        quantities and collapsing them would invent a number. Only the score aggregates,
        which is the whole point of the aggregation being score-only.
        """
        frames: list[pd.DataFrame] = []
        for w, window_id in enumerate(self.window_ids):
            stamps = self.timestamps[w] if self.timestamps else None
            keep = (
                np.flatnonzero(~pd.isna(stamps))
                if stamps is not None
                else np.arange(self.reconstruction.shape[2])
            )
            stamp_column = stamps[keep] if stamps is not None else keep
            common = {"series_id": self.series_ids[w], "window_id": window_id}
            if self.is_aggregated:
                frames.append(
                    pd.DataFrame(
                        {
                            **common,
                            "timestamp": stamp_column,
                            "channel": self.channel_aggregation,
                            "anomaly_score": self.anomaly_score[w, keep],
                            "scored": self.scored_mask[w, keep].astype(bool),
                        }
                    )
                )
                continue
            for c, channel in enumerate(self.channels):
                frames.append(
                    pd.DataFrame(
                        {
                            **common,
                            "timestamp": stamp_column,
                            "channel": channel,
                            "reconstruction": self.reconstruction[w, c, keep],
                            "reconstruction_error": self.reconstruction_error[w, c, keep],
                            "anomaly_score": self.anomaly_score[w, c, keep],
                            "scored": self.scored_mask[w, c, keep].astype(bool),
                        }
                    )
                )
        return pd.concat(frames, ignore_index=True)


def residual(x_enc: np.ndarray, reconstruction: np.ndarray, loss: str = DEFAULT_LOSS) -> np.ndarray:
    """Elementwise reconstruction residual under `loss`. Pure; no model, no weights.

    `"mae"` is `|x - x_hat|` and `"mse"` is `(x - x_hat)**2`, matching
    `get_anomaly_criterion`'s `L1Loss(reduction="none")` / `MSELoss(reduction="none")`.
    """
    if loss not in ANOMALY_LOSSES:
        raise AnomalyConfigError(
            f"loss={loss!r} is not supported; choose one of {list(ANOMALY_LOSSES)}"
        )
    difference = reconstruction.astype(np.float64) - x_enc.astype(np.float64)
    error = np.abs(difference) if loss == "mae" else np.square(difference)
    return error.astype(np.float32)


def aggregate_channels(score: np.ndarray, how: str = DEFAULT_AGGREGATION) -> np.ndarray:
    """Collapse the channel axis of a `(batch, channel, timestep)` score, and nothing else.

    NaN-aware: a position aggregates over the channels whose score is defined, and stays
    NaN only where no channel scored. `"none"` returns the input untouched.
    """
    if how not in CHANNEL_AGGREGATIONS:
        raise AnomalyConfigError(
            f"channel_aggregation={how!r} is not supported; choose one of "
            f"{list(CHANNEL_AGGREGATIONS)}"
        )
    if how == "none":
        return score
    defined = ~np.isnan(score)
    all_nan = ~defined.any(axis=1)
    # `nanmean`/`nanmax` warn on an all-NaN slice and this is a routine case -- a position
    # unscored in every channel -- so the reduction is done on a filled copy with a
    # neutral identity and the all-NaN positions are restored afterwards. No warning, and
    # the identity never reaches the output.
    if how == "mean":
        counts = defined.sum(axis=1)
        collapsed = np.where(defined, score, 0.0).sum(axis=1) / np.maximum(counts, 1)
    else:
        collapsed = np.where(defined, score, -np.inf).max(axis=1)
    return np.where(all_nan, np.nan, collapsed).astype(np.float32)


def score_anomalies(
    windows: WindowSet,
    model: LoadedMoment,
    loss: str = DEFAULT_LOSS,
    channel_aggregation: str = DEFAULT_AGGREGATION,
    batch_size: int = 8,
    warmup: bool = True,
) -> AnomalyResult:
    """Score every window by its unmasked self-reconstruction residual.

    Runs the same pinned reconstruction entry point as `imputation.reconstruct` with no
    extra hiding mask, so the model sees every observed point — including the ones it is
    being scored on. That is the RFC's v1 definition, and `MODEL_CARD.md` records what it
    costs.

    Args:
        loss: `"mae"` (default) or `"mse"`; recorded on the result and in provenance.
        channel_aggregation: `"none"` (default), `"mean"` or `"max"`. Collapses the
            channel axis only.
        warmup: as `imputation.reconstruct` — a discarded forward pass so
            `latency_seconds` excludes lazy allocation. Pass False in production.

    Raises:
        AnomalyConfigError: unsupported `loss` or `channel_aggregation`, refused before
            the model runs rather than after a forward pass.
    """
    if loss not in ANOMALY_LOSSES:
        raise AnomalyConfigError(
            f"loss={loss!r} is not supported; choose one of {list(ANOMALY_LOSSES)}"
        )
    if channel_aggregation not in CHANNEL_AGGREGATIONS:
        raise AnomalyConfigError(
            f"channel_aggregation={channel_aggregation!r} is not supported; choose one of "
            f"{list(CHANNEL_AGGREGATIONS)}"
        )

    started = time.perf_counter()
    # mask=None is the unmasked self-reconstruction: the only positions hidden are the
    # ones the source did not contain, which `reconstruct` hides for us and which the
    # scored domain then excludes anyway.
    reconstructed = reconstruct(
        windows, model, mask=None, batch_size=batch_size, warmup=warmup
    )
    result = score_from_reconstruction(
        windows, reconstructed, loss=loss, channel_aggregation=channel_aggregation
    )
    return _with_latency(result, time.perf_counter() - started)


def score_from_reconstruction(
    windows: WindowSet,
    reconstructed: ReconstructionResult,
    loss: str = DEFAULT_LOSS,
    channel_aggregation: str = DEFAULT_AGGREGATION,
) -> AnomalyResult:
    """Derive scores from a reconstruction that has already been computed.

    Split out from `score_anomalies` so the scoring arithmetic — the scored domain, the
    residual, the aggregation — is testable against a `WindowSet` without 454 MB of
    weights, and so a caller who already holds a `ReconstructionResult` does not pay for a
    second forward pass.
    """
    error = residual(windows.x_enc, reconstructed.reconstruction, loss)

    # Non-padded AND visible to the model. `model_mask` is the patch-expanded mask that
    # `reconstruct` actually handed to momentfm, so this is what the encoder saw, not what
    # the caller hoped it saw.
    visible = windows.input_mask * reconstructed.model_mask
    scored = np.broadcast_to(visible[:, None, :], error.shape).astype(np.float32)

    valid = np.broadcast_to(windows.input_mask[:, None, :], error.shape)
    prefilled = (valid == 1) & (windows.point_mask == 0)
    hidden_but_observed = (valid == 1) & (windows.point_mask == 1) & (scored == 0)

    score = np.where(scored == 1, error, np.nan).astype(np.float32)
    score = aggregate_channels(score, channel_aggregation)
    scored_mask = visible.astype(np.float32) if channel_aggregation != "none" else scored

    denominator = float(valid.sum())
    return AnomalyResult(
        reconstruction=reconstructed.reconstruction,
        reconstruction_error=error,
        anomaly_score=score,
        scored_mask=scored_mask,
        point_mask=windows.point_mask,
        model_mask=reconstructed.model_mask,
        series_ids=windows.series_ids,
        window_ids=windows.window_ids,
        channels=windows.channels,
        loss=loss,
        channel_aggregation=channel_aggregation,
        latency_seconds=reconstructed.latency_seconds,
        scored_point_count=int((scored == 1).sum()),
        unscored_prefilled_count=int(prefilled.sum()),
        unscored_hidden_by_patch_count=int(hidden_but_observed.sum()),
        scored_point_fraction=(float((scored == 1).sum()) / denominator) if denominator else 0.0,
        masked_point_fraction=reconstructed.masked_point_fraction,
        masked_patch_fraction=reconstructed.masked_patch_fraction,
        timestamps=windows.timestamps,
    )


def _with_latency(result: AnomalyResult, seconds: float) -> AnomalyResult:
    """Total wall time for `score_anomalies`, which is the reconstruction plus the
    scoring arithmetic — not the reconstruction alone."""
    return dataclasses.replace(result, latency_seconds=seconds)


__all__ = [
    "ANOMALY_LOSSES",
    "CHANNEL_AGGREGATIONS",
    "DEFAULT_AGGREGATION",
    "DEFAULT_LOSS",
    "AnomalyConfigError",
    "AnomalyResult",
    "aggregate_channels",
    "residual",
    "score_anomalies",
    "score_from_reconstruction",
]
