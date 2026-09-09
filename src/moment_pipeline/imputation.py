"""Task 2 — reconstruction plus the v1 user-facing imputation contract.

MOMENT-1-base exposes pretrained reconstruction weights. DIMER uses that primitive
for imputation but keeps the product semantics explicit:

* canonical ``x_enc`` is finite before model entry; raw NaN never reaches MOMENT;
* the model mask is always explicit and patch-quantized;
* **default imputed output preserves every observed source value** and replaces
  only source-missing values or points the caller deliberately hid;
* full reconstruction remains separately available;
* artificial-mask evaluation scores only deliberately hidden points for which
  ground truth existed in the source. Source-missing points are never invented
  into the denominator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .canonical import (
    WindowSet,
    expand_patch_view,
    hidden_position_fraction,
    masked_patch_fraction_of,
    missing_point_fraction,
    to_patch_view,
)
from .embedding import TaskMismatchError
from .model import LoadedMoment


class DegenerateMaskError(ValueError):
    """A window would be handed to MOMENT with nothing visible at all."""


@dataclass(frozen=True)
class ImputationMetrics:
    """Masked-point-only evaluation for deliberately hidden ground-truth points."""

    n: int
    mae: float
    rmse: float


@dataclass(frozen=True)
class ReconstructionResult:
    """Reconstruction, imputed series, masks and accounting for one call.

    ``masked_point_fraction`` is source missingness. ``model_masked_point_fraction``
    is the channel-collapsed fraction hidden from MOMENT after source missingness and
    any caller mask are combined. ``requested_visible_mask`` records only the caller's
    extra mask before patch expansion, so artificial-mask evaluation can distinguish
    requested targets from neighbouring points hidden solely because MOMENT works by
    patches.

    The final five fields have defaults for backwards compatibility with tests that
    construct a minimal result only to exercise provenance serialization.
    """

    reconstruction: np.ndarray
    point_mask: np.ndarray
    model_mask: np.ndarray
    patch_mask: np.ndarray
    masked_point_fraction: float
    masked_point_count: int
    model_masked_point_fraction: float
    model_masked_point_count: int
    masked_patch_fraction: float
    masked_patch_count: int
    series_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    channels: tuple[str, ...]
    latency_seconds: float
    input_values: np.ndarray | None = None
    input_mask: np.ndarray | None = None
    timestamps: tuple[np.ndarray, ...] = ()
    requested_visible_mask: np.ndarray | None = None
    imputed: np.ndarray | None = None

    def _require_export_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.input_values is None
            or self.input_mask is None
            or self.requested_visible_mask is None
            or self.imputed is None
        ):
            raise ValueError(
                "this ReconstructionResult was constructed without the v1 imputation export "
                "state; call reconstruct()/impute() rather than instantiating it manually"
            )
        return (
            self.input_values,
            self.input_mask,
            self.requested_visible_mask,
            self.imputed,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return one row per non-padded (window, channel, timestamp) cell.

        ``original_value`` is NaN only where the source itself was missing. A
        deliberately hidden point retains its ground truth in ``original_value`` so
        tutorial/evaluation code can audit what was withheld, while ``imputed_value``
        contains the model-derived replacement at that point.
        """

        input_values, input_mask, requested_visible, imputed = self._require_export_state()
        rows: list[dict[str, Any]] = []
        for window_index, (series_id, window_id) in enumerate(
            zip(self.series_ids, self.window_ids, strict=True)
        ):
            stamps = self.timestamps[window_index]
            for channel_index, channel in enumerate(self.channels):
                for position in np.flatnonzero(input_mask[window_index] == 1):
                    source_observed = bool(self.point_mask[window_index, channel_index, position])
                    requested_hidden = bool(requested_visible[window_index, position] == 0)
                    rows.append(
                        {
                            "series_id": series_id,
                            "window_id": window_id,
                            "timestamp": pd.Timestamp(stamps[position]),
                            "channel": channel,
                            "original_value": (
                                float(input_values[window_index, channel_index, position])
                                if source_observed
                                else np.nan
                            ),
                            "imputed_value": float(imputed[window_index, channel_index, position]),
                            "reconstruction": float(
                                self.reconstruction[window_index, channel_index, position]
                            ),
                            "source_observed": source_observed,
                            "source_missing": not source_observed,
                            "requested_hidden": requested_hidden,
                            "model_hidden": bool(self.model_mask[window_index, position] == 0),
                        }
                    )
        return pd.DataFrame(rows)


def mask_accounting(
    windows: WindowSet, visible_points: np.ndarray, patch_mask: np.ndarray
) -> dict[str, float | int]:
    """Every reported masking number for one reconstruction call, in one place."""

    valid_points = np.broadcast_to(windows.input_mask[:, None, :], windows.point_mask.shape)
    valid_patches = to_patch_view(windows.input_mask, windows.patch_length)
    return {
        "masked_point_fraction": missing_point_fraction(windows.point_mask, windows.input_mask),
        "masked_point_count": int(((valid_points == 1) & (windows.point_mask == 0)).sum()),
        "model_masked_point_fraction": hidden_position_fraction(
            visible_points, windows.input_mask
        ),
        "model_masked_point_count": int(
            ((windows.input_mask == 1) & (visible_points == 0)).sum()
        ),
        "masked_patch_fraction": masked_patch_fraction_of(
            patch_mask, windows.input_mask, windows.patch_length
        ),
        "masked_patch_count": int(((valid_patches == 1) & (patch_mask == 0)).sum()),
    }


def _requested_visible_mask(windows: WindowSet, mask: np.ndarray | None) -> np.ndarray:
    """Normalize only the caller's extra mask, independent of source missingness."""

    if mask is None:
        return np.ones_like(windows.input_mask, dtype=np.float32)
    requested = np.asarray(mask, dtype=np.float32)
    if requested.ndim == 3:
        if requested.shape != windows.point_mask.shape:
            raise ValueError(
                f"mask shape {requested.shape} != point_mask shape {windows.point_mask.shape}"
            )
        # MOMENT has no channel axis in its reconstruction mask. If one channel is
        # deliberately hidden at a timestamp, the model must treat that timestamp as
        # hidden for every channel; record the same conservative collapse explicitly.
        requested = requested.min(axis=1)
    if requested.shape != windows.input_mask.shape:
        raise ValueError(f"mask shape {requested.shape} != expected {windows.input_mask.shape}")
    if not np.isin(np.unique(requested), (0.0, 1.0)).all():
        raise ValueError("mask must be binary (1 = visible to the model, 0 = hidden)")
    return requested.astype(np.float32)


def _combine_masks(windows: WindowSet, requested_visible: np.ndarray) -> np.ndarray:
    """Visible-point mask: source observed in every channel AND caller-visible."""

    return np.minimum(windows.model_point_mask, requested_visible).astype(np.float32)


def masked_point_metrics(result: ReconstructionResult) -> ImputationMetrics:
    """Evaluate only deliberately hidden source-observed points.

    Source missing values have no truth and are excluded. Positions hidden only by
    patch expansion are also excluded: the tutorial metric answers how well the model
    reconstructed the points the evaluator intentionally held out, not every neighbour
    MOMENT had to hide internally to satisfy its patch contract.
    """

    input_values, input_mask, requested_visible, imputed = result._require_export_state()
    valid = np.broadcast_to(input_mask[:, None, :], result.point_mask.shape) == 1
    deliberate = np.broadcast_to(
        requested_visible[:, None, :] == 0, result.point_mask.shape
    )
    scorable = valid & (result.point_mask == 1) & deliberate
    n = int(scorable.sum())
    if n == 0:
        raise ValueError(
            "no deliberately hidden source-observed points are available for imputation "
            "evaluation; pass an artificial mask to reconstruct()/impute()"
        )
    error = imputed[scorable] - input_values[scorable]
    return ImputationMetrics(
        n=n,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
    )


def reconstruct(
    windows: WindowSet,
    model: LoadedMoment,
    mask: np.ndarray | None = None,
    batch_size: int = 8,
    warmup: bool = True,
) -> ReconstructionResult:
    """Run the pinned reconstruction path and construct the v1 imputed product.

    Args:
        windows: canonical windows; ``x_enc`` is finite by construction.
        model: pipeline loaded with ``task='reconstruction'``.
        mask: optional extra hiding mask, ``(batch, seq_len)`` or
            ``(batch, channels, seq_len)``. One means visible. It is combined with
            source missingness and then patch-quantized for MOMENT.
        warmup: run one discarded forward pass before timing the scored call.
    """

    import torch

    if model.identity.task != "reconstruction":
        raise TaskMismatchError(
            f"reconstruct() needs a pipeline loaded with task='reconstruction', got "
            f"{model.identity.task!r}"
        )
    if not np.isfinite(windows.x_enc).all():
        raise ValueError(
            "x_enc contains non-finite values; MOMENT.reconstruct has no nan_to_num and "
            "would propagate them. Build windows through moment_pipeline.canonical."
        )

    requested_visible = _requested_visible_mask(windows, mask)
    visible_points = _combine_masks(windows, requested_visible)
    patch_mask = to_patch_view(visible_points, windows.patch_length)
    model_mask = expand_patch_view(patch_mask, windows.patch_length)

    blind = np.flatnonzero(
        (patch_mask.sum(axis=1) == 0) & (windows.input_mask.sum(axis=1) > 0)
    )
    if blind.size:
        raise DegenerateMaskError(
            "[DEGENERATE_MASK] no patch is visible in window(s) "
            f"{[windows.window_ids[i] for i in blind.tolist()]}; MOMENT would normalize "
            "against an all-zero mask and return NaN. This is a masking fault, not a "
            "pre-fill fault: every value in x_enc is finite."
        )

    device = model.identity.device
    x_all = torch.from_numpy(windows.x_enc)
    input_all = torch.from_numpy(windows.input_mask)
    mask_all = torch.from_numpy(model_mask)

    def _run() -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, windows.n_windows, batch_size):
                stop = start + batch_size
                out = model.pipeline.reconstruct(
                    x_enc=x_all[start:stop].to(device),
                    input_mask=input_all[start:stop].to(device),
                    mask=mask_all[start:stop].to(device),
                )
                chunks.append(out.reconstruction.detach().float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    if warmup:
        _run()
    started = time.perf_counter()
    reconstruction = _run()
    latency = time.perf_counter() - started

    if not np.isfinite(reconstruction).all():
        raise ValueError(
            "reconstruction contains non-finite values despite finite pre-fill; "
            "this violates the RFC M-2 contract"
        )

    # Replacement semantics are intentionally narrower than model_mask. A source
    # missing value is replaced. A source-observed value is replaced only when the
    # caller deliberately requested it hidden. Neighbouring observed points that MOMENT
    # had to hide merely because they share an 8-step patch remain untouched.
    valid_cells = np.broadcast_to(
        windows.input_mask[:, None, :] == 1, windows.point_mask.shape
    )
    deliberate_cells = np.broadcast_to(
        requested_visible[:, None, :] == 0, windows.point_mask.shape
    )
    replacement = valid_cells & ((windows.point_mask == 0) | deliberate_cells)
    imputed = np.where(replacement, reconstruction, windows.x_enc).astype(np.float32)

    return ReconstructionResult(
        reconstruction=reconstruction,
        point_mask=windows.point_mask,
        model_mask=model_mask,
        patch_mask=patch_mask,
        series_ids=windows.series_ids,
        window_ids=windows.window_ids,
        channels=windows.channels,
        latency_seconds=latency,
        input_values=windows.x_enc,
        input_mask=windows.input_mask,
        timestamps=windows.timestamps,
        requested_visible_mask=requested_visible,
        imputed=imputed,
        **mask_accounting(windows, visible_points, patch_mask),
    )


def impute(
    windows: WindowSet,
    model: LoadedMoment,
    mask: np.ndarray | None = None,
    batch_size: int = 8,
    warmup: bool = True,
) -> ReconstructionResult:
    """User-facing alias for :func:`reconstruct` emphasizing imputed output semantics."""

    return reconstruct(
        windows,
        model,
        mask=mask,
        batch_size=batch_size,
        warmup=warmup,
    )
