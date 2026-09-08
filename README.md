# MOMENT — DIMER Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/moment-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat)](https://huggingface.co/AutonLab/MOMENT-1-base)
[![Upstream](https://img.shields.io/badge/Upstream-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white)](https://github.com/moment-timeseries-foundation-model/moment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A DIMER pipeline that turns [MOMENT-1-base](https://huggingface.co/AutonLab/MOMENT-1-base),
an open-weight time-series foundation model, into a reproducible service. You supply a
long-format table of `(series_id, timestamp, channel, value)` rows; the pipeline validates
it, converts it deterministically into MOMENT's canonical tensors, and runs the pretrained
encoder.

**Status: Phase 1 — foundation.** The RFC contract is versioned in-repo, the runtime is
locked, the loader is integrity-verified, and the canonical converter plus the
finite-pre-fill and masking contract are implemented and covered by unit and CPU
integration tests. Exports, tutorials, sample datasets and anomaly scoring are later phases.

## What v1 exposes — and what it does not

The released MOMENT-1-base checkpoint is a **reconstruction** model. Only that path carries
pretrained weights, so v1 exposes only what the base weights support:

1. time-series embeddings / representation extraction;
2. imputation / reconstruction;
3. reconstruction-based anomaly scoring (Phase 4).

**Short-horizon forecasting and classification are NOT in the public API.** `momentfm`
would happily build those heads, but it builds them *freshly initialized* — they are
adaptation workflows, not pretrained capabilities, and this pipeline refuses them at the
loader. DIMER's zero-shot forecasting service is Chronos-2.

See [MODEL_CARD.md](MODEL_CARD.md) for the pretrained-vs-adapted semantics, patch-masking
quantization, licences, and the full supply-chain constants.

## Quickstart

```bash
uv sync --locked          # exact dependency graph, same one CI installs
export HF_HUB_DISABLE_SYMLINKS_WARNING=1   # Windows/WSL caches cannot symlink
uv run pytest -m "not integration"         # unit suite: no network, no weights
uv run pytest -m integration               # CPU suite: downloads ~454 MB once
```

```python
import pandas as pd
from moment_pipeline import build_provenance, embed, load_moment, to_windows

frame = pd.read_csv("my_series.csv")  # series_id, timestamp, channel, value
windows = to_windows(frame)           # validate + canonicalize to (b, c, 512)

model = load_moment(task="embedding", device="auto")
result = embed(windows, model)

print(result.embeddings.shape)        # (n_windows, 768)
print(result.to_frame().head())       # series_id, window_id, embedding_0 … embedding_767
provenance = build_provenance(model, windows, result)
```

For imputation/reconstruction, load the **reconstruction** task — MOMENT swaps its head per
task, so the instances are deliberately separate:

```python
from moment_pipeline import reconstruct

model = load_moment(task="reconstruction", device="cpu")
result = reconstruct(windows, model)          # mask is always explicit, never None
print(result.masked_point_fraction, result.masked_patch_fraction)
```

## Canonical data contract

Input is long format:

```csv
series_id,timestamp,channel,value
A,2026-01-01T00:00:00,signal_1,0.12
A,2026-01-01T01:00:00,signal_1,0.18
```

The converter's documented policy (Phase 1):

| Policy | Behaviour |
|---|---|
| Channel ordering | sorted unique channel names of the whole input |
| Windowing | one window per `series_id` — the last 512 distinct timestamps |
| Alignment | right-aligned |
| Padding | shorter series left-padded, `input_mask = 0` on the padding |
| Truncation | longer series keep the last 512, `truncated=True` disclosed |
| Missing values | finite pre-fill (default `0.0`); missingness lives only in the masks |
| Frequency | irregular spacing surfaced, never interpolated |
| Normalization | delegated to MOMENT's internal RevIN, driven by `input_mask` |

Missing values are quantized to patches: one missing point makes its whole 8-step patch
unobserved to the model. Both `masked_point_fraction` and `masked_patch_fraction` are
reported.

## Repository layout

```text
docs/rfc/0001-moment-base.md   verbatim RFC copy — the mission anchor
src/moment_pipeline/
  config.py       runtime config + resource limits
  validation.py   the common validation contract (12 rules)
  canonical.py    long format -> (x_enc, input_mask, point_mask, patch_mask)
  model.py        pinned, digest-verified, safetensors-proved loader
  embedding.py    Task 1 — pooled encoder embeddings
  imputation.py   Task 2 — the reconstruction primitive
  anomaly.py      Task 3 — deferred to Phase 4 (deliberately behaviour-free)
  provenance.py   model / runtime / inference export metadata
tests/            unit (no network) + integration (real weights, CPU)
```

## Contract and provenance

- The approved contract is [`docs/rfc/0001-moment-base.md`](docs/rfc/0001-moment-base.md),
  a verbatim copy of [issue #1](https://github.com/kurtvalcorza/moment-pipeline/issues/1)
  captured at `2026-09-08T04:34:15Z`. The issue body is mutable; the file is the anchor.
- `momentfm` is installed from upstream commit `38f7310ad594100747ca2a8357e9c7ca7d323e0e`;
  the version string `0.1.5` does not exist on PyPI.
- Every export records the pinned revision, both digests, the weight file the loader
  *proved* it was using, the full runtime version set, device and dtype.

## Licence

Pipeline code: MIT, Copyright (c) 2026 Kurt Valcorza — see [LICENSE](LICENSE).
Model weights and upstream `momentfm` code carry their own MIT terms; the distinction is
recorded in [MODEL_CARD.md](MODEL_CARD.md).
