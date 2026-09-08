"""Long-format -> canonical MOMENT tensor conversion.

Documented policy for the pinned `AutonLab/MOMENT-1-base` checkpoint (seq_len 512,
patch_len 8, stride 8):

* **Channel ordering** — the sorted unique channel names of the *whole* input. Every
  window therefore shares one channel axis, and the order does not depend on row order.
* **Windowing (Phase 1)** — exactly ONE window per `series_id`: the last 512 distinct
  timestamps of that series, right-aligned.
* **Truncation** — a series with more than 512 distinct timestamps keeps only the last
  512; `WindowSet.truncated` flags it, so the loss is disclosed before inference.
* **Padding** — a series with fewer than 512 timestamps is LEFT-padded. Padded positions
  get `input_mask = 0` (MOMENT's RevIN normalizer ignores them) and `point_mask = 0`.
* **Missing values** — mandatory finite pre-fill. Missing payloads are replaced with
  `MomentConfig.prefill_value` (default 0.0) *before* the tensor exists; missingness
  survives only in `point_mask`. Raw NaN never reaches MOMENT (RFC M-2, rules 11-12).
* **Frequency** — irregular spacing is surfaced in the validation report, never
  interpolated (RFC rule 7).
* **Normalization** — delegated to MOMENT's internal RevIN, which is driven by
  `input_mask`. This layer performs no scaling of its own.

Mask vocabulary (1 = usable, 0 = not), all `float32`:

| array              | shape             | meaning                                        |
|--------------------|-------------------|------------------------------------------------|
| `input_mask`       | (b, 512)          | position belongs to the series (not padding)   |
| `point_mask`       | (b, c, 512)       | value observed rather than pre-filled          |
| `model_point_mask` | (b, 512)          | `point_mask` collapsed over channels (min)     |
| `patch_mask`       | (b, 64)           | patch fully observed (upstream patch view)     |
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MomentConfig
from .validation import ValidationError, ValidationReport, validate_long_frame


def to_patch_view(point_mask: np.ndarray, patch_length: int = 8) -> np.ndarray:
    """Reproduce `momentfm.utils.masking.Masking.convert_seq_to_patch_view`.

    A patch counts as observed only when all `patch_length` of its points are observed
    (upstream: ``(mask.unfold(...).sum(dim=-1) == patch_len)``). This is the quantization
    that makes one missing point cost a whole patch (RFC M-3).
    """
    if point_mask.ndim != 2:
        raise ValueError(f"expected a (batch, seq_len) mask, got shape {point_mask.shape}")
    batch, seq_len = point_mask.shape
    if seq_len % patch_length:
        raise ValueError(f"seq_len {seq_len} is not a multiple of patch_length {patch_length}")
    folded = point_mask.reshape(batch, seq_len // patch_length, patch_length)
    return (folded.sum(axis=-1) == patch_length).astype(np.float32)


def expand_patch_view(patch_mask: np.ndarray, patch_length: int = 8) -> np.ndarray:
    """Expand a (batch, n_patches) patch mask back to (batch, seq_len) point granularity."""
    return np.repeat(patch_mask, patch_length, axis=1).astype(np.float32)


@dataclass(frozen=True)
class WindowSet:
    """The canonical DIMER-facing representation handed to MOMENT."""

    x_enc: np.ndarray
    input_mask: np.ndarray
    point_mask: np.ndarray
    series_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    channels: tuple[str, ...]
    truncated: tuple[bool, ...]
    padded: tuple[bool, ...]
    n_source_timestamps: tuple[int, ...]
    timestamps: tuple[np.ndarray, ...] = field(repr=False, default=())
    prefill_value: float = 0.0
    patch_length: int = 8
    validation: ValidationReport | None = field(repr=False, default=None)

    @property
    def n_windows(self) -> int:
        return int(self.x_enc.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.x_enc.shape[1])

    @property
    def sequence_length(self) -> int:
        return int(self.x_enc.shape[2])

    @property
    def model_point_mask(self) -> np.ndarray:
        """`point_mask` collapsed over channels: a position is observed only if it is
        observed in every channel. MOMENT's `mask` argument has no channel axis, so the
        collapse is conservative and explicit rather than silently per-channel."""
        return self.point_mask.min(axis=1).astype(np.float32)

    @property
    def patch_mask(self) -> np.ndarray:
        return to_patch_view(self.model_point_mask, self.patch_length)

    def patch_quantized_mask(self) -> np.ndarray:
        """The (batch, seq_len) mask actually passed to `pipeline.reconstruct`.

        It is the patch view expanded back to points, so what the caller sees is exactly
        what the model uses — no hidden widening inside momentfm.
        """
        return expand_patch_view(self.patch_mask, self.patch_length)

    @property
    def valid_point_count(self) -> int:
        return int(self.input_mask.sum() * self.n_channels)

    @property
    def masked_point_fraction(self) -> float:
        """Fraction of *non-padded* (window, channel, position) cells that were missing."""
        valid = np.broadcast_to(self.input_mask[:, None, :], self.point_mask.shape)
        denom = float(valid.sum())
        if denom == 0.0:
            return 0.0
        return float(((valid == 1) & (self.point_mask == 0)).sum()) / denom

    @property
    def masked_patch_fraction(self) -> float:
        """Fraction of *non-padded* patches the model will treat as unobserved."""
        valid_patches = to_patch_view(self.input_mask, self.patch_length)
        denom = float(valid_patches.sum())
        if denom == 0.0:
            return 0.0
        masked = ((valid_patches == 1) & (self.patch_mask == 0)).sum()
        return float(masked) / denom

    @property
    def padded_fraction(self) -> float:
        return float((self.input_mask == 0).sum()) / float(self.input_mask.size)


def to_windows(
    df: pd.DataFrame,
    config: MomentConfig | None = None,
    report: ValidationReport | None = None,
    frame: pd.DataFrame | None = None,
) -> WindowSet:
    """Validate (unless a report+frame is supplied) and convert to the canonical tensors."""
    config = config or MomentConfig()
    if report is None or frame is None:
        report, frame = validate_long_frame(df, config)

    channels = list(report.channels)
    series_ids = list(report.series_ids)
    seq_len = config.sequence_length

    if len(series_ids) > config.limits.max_windows:
        raise ValidationError(
            "LIMIT_EXCEEDED",
            f"{len(series_ids)} windows exceeds max_windows={config.limits.max_windows}",
            {
                "limit": "max_windows",
                "actual": len(series_ids),
                "allowed": config.limits.max_windows,
            },
        )

    x_rows: list[np.ndarray] = []
    input_rows: list[np.ndarray] = []
    point_rows: list[np.ndarray] = []
    stamp_rows: list[np.ndarray] = []
    truncated: list[bool] = []
    padded: list[bool] = []
    n_source: list[int] = []

    grouped = {sid: sub for sid, sub in frame.groupby("series_id", sort=True)}
    for series_id in series_ids:
        subset = grouped[series_id]
        stamps = np.sort(subset["timestamp"].unique().astype("datetime64[ns]"))
        n_source.append(int(stamps.size))
        truncated.append(bool(stamps.size > seq_len))
        padded.append(bool(stamps.size < seq_len))
        kept = stamps[-seq_len:]

        wide = (
            subset.set_index(["channel", "timestamp"])["value"]
            .unstack("timestamp")
            .reindex(index=channels, columns=pd.DatetimeIndex(kept))
        )
        grid = wide.to_numpy(dtype="float64", na_value=np.nan)
        observed = np.isfinite(grid)
        if not observed.any():
            raise ValidationError(
                "ALL_MISSING_WINDOW",
                f"series {series_id!r} has no observed value in its 512-point window",
                {"series_id": series_id, "n_timestamps": int(stamps.size)},
            )

        filled = np.where(observed, grid, config.prefill_value).astype(np.float32)
        pad = seq_len - kept.size
        if pad > 0:
            filled = np.concatenate(
                [np.full((len(channels), pad), config.prefill_value, np.float32), filled], axis=1
            )
            observed = np.concatenate(
                [np.zeros((len(channels), pad), bool), observed], axis=1
            )
            window_mask = np.concatenate(
                [np.zeros(pad, np.float32), np.ones(kept.size, np.float32)]
            )
            window_stamps = np.concatenate(
                [np.full(pad, np.datetime64("NaT", "ns")), kept]
            )
        else:
            window_mask = np.ones(seq_len, np.float32)
            window_stamps = kept

        x_rows.append(filled)
        input_rows.append(window_mask)
        point_rows.append(observed.astype(np.float32))
        stamp_rows.append(window_stamps)

    x_enc = np.stack(x_rows).astype(np.float32)
    if not np.isfinite(x_enc).all():  # pragma: no cover - defensive, pre-fill guarantees this
        raise ValidationError(
            "NON_FINITE_TENSOR",
            "canonical x_enc contains non-finite values after pre-fill",
            {},
        )

    return WindowSet(
        x_enc=x_enc,
        input_mask=np.stack(input_rows).astype(np.float32),
        point_mask=np.stack(point_rows).astype(np.float32),
        series_ids=tuple(series_ids),
        window_ids=tuple(f"{sid}::w0" for sid in series_ids),
        channels=tuple(channels),
        truncated=tuple(truncated),
        padded=tuple(padded),
        n_source_timestamps=tuple(n_source),
        timestamps=tuple(stamp_rows),
        prefill_value=float(config.prefill_value),
        patch_length=config.patch_length,
        validation=report,
    )
