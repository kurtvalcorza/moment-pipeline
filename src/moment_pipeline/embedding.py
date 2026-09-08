"""Task 1 — pretrained-encoder embeddings.

Phase 1 exposes the minimum defensible contract: one pooled vector per window from the
*pretrained encoder*, with the reduction recorded in provenance.

Reduction semantics (RFC M-5), read off the installed
`momentfm/models/moment.py::MOMENT.embed`:

* `reduction="mean"` first averages over the **channel** axis
  (`enc_out.mean(dim=1)`), then takes the `input_mask`-weighted mean over patches.
  A multichannel window therefore yields ONE vector in which channels are averaged —
  it is not a per-channel representation and must not be described as one.
* No task head is involved: the embedding loader uses `nn.Identity`, so nothing about
  this path is freshly initialized. The upstream warning "Only reconstruction head is
  pre-trained…" fires for any non-reconstruction task and refers to *heads only*.

**Known limitation — embeddings are not missingness-aware.** Upstream
`MOMENT.embed(self, *, x_enc, input_mask=None, reduction='mean', **kwargs)` takes no
per-point observedness mask; `input_mask` is the *padding* mask, and momentfm feeds it to
both RevIN (`moment.py:254`) and the patch embedding (`moment.py:262`). The finite pre-fill
is therefore consumed as observed data: a window that is half missing produces a vector
byte-identical to the same window with literal zeros in those positions, and both differ
from the clean window. `point_mask` cannot reach the model on this path -- there is no
parameter to carry it. This module therefore records the missingness fractions on the
result and in provenance, which are the only signal an export carries about it. See
`MODEL_CARD.md` and `tests/test_integration_model.py::
test_embeddings_are_missingness_blind_known_limitation`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import WindowSet
from .model import LoadedMoment

REDUCTION = "mean"

#: Stated on every embedding result and in every embedding provenance block. The point is
#: that the recorded fractions are the *only* place missingness survives on this path.
MISSINGNESS_POLICY = (
    "NOT missingness-aware: upstream MOMENT.embed accepts no per-point observedness mask "
    "(input_mask is the padding mask), so pre-filled positions are seen by the encoder as "
    "observed values and enter RevIN and the patch embedding as data. An embedding of a "
    "window containing missing data equals the embedding of the same window with the "
    "prefill_value written into those positions. masked_point_fraction / "
    "masked_patch_fraction are the only record that any of it was fabricated"
)
CHANNEL_POLICY = (
    "channels are averaged inside momentfm before patch pooling (reduction='mean'); "
    "the result is one vector per window, not one per channel"
)


class TaskMismatchError(RuntimeError):
    """The loaded pipeline was built for a different task than the call requires."""


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: np.ndarray
    series_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    d_model: int
    latency_seconds: float
    reduction: str = REDUCTION
    channel_policy: str = CHANNEL_POLICY
    n_channels: int = 1
    truncated: tuple[bool, ...] = field(default=())
    padded: tuple[bool, ...] = field(default=())
    #: Source missingness of the windows these vectors were computed from. Same definition
    #: as `WindowSet.masked_point_fraction` / `.masked_patch_fraction` (R-5): the point
    #: fraction's denominator is non-padded cells x channels, the patch fraction's is
    #: non-padded patches. Nothing about them reached the model -- see MISSINGNESS_POLICY.
    masked_point_fraction: float = 0.0
    masked_point_count: int = 0
    masked_patch_fraction: float = 0.0
    missingness_policy: str = MISSINGNESS_POLICY
    missingness_visible_to_model: bool = False

    def to_frame(self) -> pd.DataFrame:
        """`series_id, window_id, embedding_0 … embedding_n` (RFC Task 1 output)."""
        columns = {f"embedding_{i}": self.embeddings[:, i] for i in range(self.embeddings.shape[1])}
        return pd.DataFrame(
            {"series_id": list(self.series_ids), "window_id": list(self.window_ids), **columns}
        )


def embed(
    windows: WindowSet,
    model: LoadedMoment,
    batch_size: int = 8,
    warmup: bool = True,
) -> EmbeddingResult:
    """Pooled embeddings for every window. Deterministic in eval mode.

    Args:
        warmup: run one discarded forward pass before the timed one so
            `latency_seconds` excludes lazy-allocation cost. Defaults to True for an
            honest latency number; pass False in production, where it otherwise
            doubles the cost of every call (R-9).
    """
    import torch

    if model.identity.task != "embedding":
        raise TaskMismatchError(
            f"embed() needs a pipeline loaded with task='embedding', got "
            f"{model.identity.task!r}; MOMENT replaces its head per task, so reuse of a "
            "reconstruction instance would silently change the computation"
        )
    if windows.sequence_length != model.identity.seq_len:
        raise ValueError(
            f"windows are {windows.sequence_length} long, model expects "
            f"{model.identity.seq_len}"
        )

    device = model.identity.device
    x_all = torch.from_numpy(windows.x_enc)
    mask_all = torch.from_numpy(windows.input_mask)

    def _run() -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, windows.n_windows, batch_size):
                stop = start + batch_size
                out = model.pipeline.embed(
                    x_enc=x_all[start:stop].to(device),
                    input_mask=mask_all[start:stop].to(device),
                    reduction=REDUCTION,
                )
                chunks.append(out.embeddings.detach().float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    if warmup:
        _run()  # discarded; keeps latency_seconds free of lazy-allocation cost
    started = time.perf_counter()
    embeddings = _run()
    latency = time.perf_counter() - started

    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values; the pre-fill contract failed")

    return EmbeddingResult(
        embeddings=embeddings,
        series_ids=windows.series_ids,
        window_ids=windows.window_ids,
        d_model=int(embeddings.shape[1]),
        latency_seconds=latency,
        n_channels=windows.n_channels,
        truncated=windows.truncated,
        padded=windows.padded,
        masked_point_fraction=windows.masked_point_fraction,
        masked_point_count=windows.masked_point_count,
        masked_patch_fraction=windows.masked_patch_fraction,
    )
