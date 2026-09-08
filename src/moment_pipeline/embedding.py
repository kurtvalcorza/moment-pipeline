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
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import WindowSet
from .model import LoadedMoment

REDUCTION = "mean"
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

    def to_frame(self) -> pd.DataFrame:
        """`series_id, window_id, embedding_0 … embedding_n` (RFC Task 1 output)."""
        columns = {f"embedding_{i}": self.embeddings[:, i] for i in range(self.embeddings.shape[1])}
        return pd.DataFrame(
            {"series_id": list(self.series_ids), "window_id": list(self.window_ids), **columns}
        )


def embed(windows: WindowSet, model: LoadedMoment, batch_size: int = 8) -> EmbeddingResult:
    """Pooled embeddings for every window. Deterministic in eval mode."""
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

    _run()  # warm-up before latency measurement (RFC runtime requirements)
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
    )
