"""Task 2 — the Phase 1 reconstruction primitive.

Phase 1 scope is deliberately one function. It is the smoke path that proves the
finite pre-fill + explicit-mask contract end to end:

* `x_enc` is already finite (the canonical converter pre-filled it) — `MOMENT.reconstruct`
  has **no** `nan_to_num` (unlike `MOMENT.embed`), so a raw NaN there propagates through
  `PatchEmbedding` (`mask * linear(x) + (1 - mask) * mask_embedding`) and out into the
  reconstruction. `tests/test_integration_model.py` keeps a negative control that proves it.
* `mask` is always explicit; `mask=None` is never passed. Upstream would substitute
  `torch.ones_like(input_mask)`, silently discarding every missing-value marker.
* The mask handed to momentfm is already patch-quantized, so the caller sees exactly the
  masking the model applies (RFC M-3).

Not in Phase 1: the imputed export that preserves observed points, masked-point-only MAE
/ RMSE (Phase 3) and anomaly scoring (Phase 4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .canonical import WindowSet, expand_patch_view, to_patch_view
from .embedding import TaskMismatchError
from .model import LoadedMoment


@dataclass(frozen=True)
class ReconstructionResult:
    """Raw reconstruction plus every mask the caller needs to interpret it."""

    reconstruction: np.ndarray
    point_mask: np.ndarray
    model_mask: np.ndarray
    patch_mask: np.ndarray
    masked_point_fraction: float
    masked_patch_fraction: float
    masked_point_count: int
    masked_patch_count: int
    series_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    channels: tuple[str, ...]
    latency_seconds: float


def _combine_masks(windows: WindowSet, mask: np.ndarray | None) -> np.ndarray:
    """Visible-point mask: observed AND (optionally) not deliberately hidden."""
    visible = windows.model_point_mask
    if mask is None:
        return visible
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        if mask.shape != windows.point_mask.shape:
            raise ValueError(
                f"mask shape {mask.shape} != point_mask shape {windows.point_mask.shape}"
            )
        mask = mask.min(axis=1)
    if mask.shape != visible.shape:
        raise ValueError(f"mask shape {mask.shape} != expected {visible.shape}")
    if not np.isin(np.unique(mask), (0.0, 1.0)).all():
        raise ValueError("mask must be binary (1 = visible to the model, 0 = hidden)")
    return np.minimum(visible, mask).astype(np.float32)


def reconstruct(
    windows: WindowSet,
    model: LoadedMoment,
    mask: np.ndarray | None = None,
    batch_size: int = 8,
) -> ReconstructionResult:
    """Run the pinned reconstruction entry point with an explicit patch-level mask.

    Args:
        windows: canonical windows; `x_enc` is finite by construction.
        model: a pipeline loaded with `task="reconstruction"`.
        mask: optional extra hiding mask, `(batch, seq_len)` or `(batch, channels, seq_len)`,
            1 = visible. Combined with the observed mask by `min`, then patch-quantized.
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

    visible_points = _combine_masks(windows, mask)
    patch_mask = to_patch_view(visible_points, windows.patch_length)
    model_mask = expand_patch_view(patch_mask, windows.patch_length)

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

    _run()  # warm-up before latency measurement
    started = time.perf_counter()
    reconstruction = _run()
    latency = time.perf_counter() - started

    if not np.isfinite(reconstruction).all():
        raise ValueError(
            "reconstruction contains non-finite values despite finite pre-fill; "
            "this violates the RFC M-2 contract"
        )

    valid_points = np.broadcast_to(windows.input_mask[:, None, :], windows.point_mask.shape)
    hidden_points = np.broadcast_to(visible_points[:, None, :], windows.point_mask.shape)
    masked_point_count = int(((valid_points == 1) & (hidden_points == 0)).sum())
    valid_patches = to_patch_view(windows.input_mask, windows.patch_length)
    masked_patch_count = int(((valid_patches == 1) & (patch_mask == 0)).sum())
    point_denom = float(valid_points.sum()) or 1.0
    patch_denom = float(valid_patches.sum()) or 1.0

    return ReconstructionResult(
        reconstruction=reconstruction,
        point_mask=windows.point_mask,
        model_mask=model_mask,
        patch_mask=patch_mask,
        masked_point_fraction=masked_point_count / point_denom,
        masked_patch_fraction=masked_patch_count / patch_denom,
        masked_point_count=masked_point_count,
        masked_patch_count=masked_patch_count,
        series_ids=windows.series_ids,
        window_ids=windows.window_ids,
        channels=windows.channels,
        latency_seconds=latency,
    )
