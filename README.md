# MOMENT — DIMER Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/moment-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat)](https://huggingface.co/AutonLab/MOMENT-1-base)
[![Upstream](https://img.shields.io/badge/Upstream-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white)](https://github.com/moment-timeseries-foundation-model/moment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Open embeddings tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_embeddings_colab.ipynb)
[![Open imputation tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_imputation_colab.ipynb)
[![Open anomaly tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_anomaly_detection_colab.ipynb)
[![Open classification adaptation tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_classification_colab.ipynb)

A reproducible DIMER pipeline around the pinned **AutonLab/MOMENT-1-base** checkpoint. Input is a long-format table of `(series_id, timestamp, channel, value)` rows; the pipeline validates it, canonicalizes it into MOMENT's 512-step representation, resolves and integrity-checks the approved checkpoint, and exposes only capabilities defensible from the pretrained base weights.

## Status: v1 tutorial / developer preview

The v1 capability set is implemented:

1. pretrained time-series embeddings;
2. reconstruction-backed imputation;
3. raw reconstruction-residual anomaly scoring.

The repository also includes deterministic synthetic tutorial assets, four fresh-Colab notebooks (three inference tutorials and one end-to-end classification adaptation), portable result/provenance exports, and CI that executes the notebooks against the real pinned checkpoint on `main` and manual dispatch.

This is **not yet a production-serving release**. Stable backend dispatch, explicit serving maxima, latency/SLO instrumentation, and DIMER backend packaging remain Phase 5 work tracked by the release-completion RFC.

The authoritative contracts are:

- [`docs/rfc/0001-moment-base.md`](docs/rfc/0001-moment-base.md) / issue #1 — model and task semantics;
- [`MODEL_CARD_SPEC.md`](MODEL_CARD_SPEC.md) — the DIMER-wide contract for what [`MODEL_CARD.md`](MODEL_CARD.md) must
  contain (intended uses and users, out-of-scope uses, factors, metrics and decision
  thresholds, ethical considerations), with the pre-flight checklist to run before a release;
- issue #5 — tutorial/release completion and remaining serving-readiness work.

## What v1 exposes — and what it does not

The released MOMENT-1-base checkpoint is a **reconstruction** model. Only that path carries a pretrained task head.

### Supported

- **Embeddings:** pooled pretrained encoder representations, one vector per canonical window.
- **Imputation:** full reconstruction plus a default imputed series that preserves observed source values and replaces only source-missing or deliberately hidden cells.
- **Anomaly scoring:** raw per-element reconstruction residuals (`mae` or `mse`) with explicit scored-domain accounting.
- **Classification adaptation (caller-trained head):** `adaptation.adapt` trains a linear head on the frozen pooled embeddings (frozen policy) and optionally the last encoder blocks with it (unfrozen policy), selected against the probe by validation log-loss; `evaluate` scores held-out accuracy and macro-F1 against a majority floor and a cosine k-NN vote; `save_artifact` / `load_artifact` export and reload the head plus any trained blocks as safetensors bound to the pinned base. No pretrained head is loaded.

### Deliberately unsupported

- classification with a pretrained head (none exists; the adaptation contract trains the caller's own);
- zero-shot forecasting from MOMENT;
- a universal anomaly threshold;
- claims that a pooled embedding is missingness-aware.

DIMER's zero-shot forecasting path is Chronos-2. Forecasting adaptation for MOMENT would need its own artifact/training contract; the classification adaptation contract is `moment_pipeline.adaptation` (see `MODEL_CARD.md`, capability 4).

## Live tutorials

The four notebooks in [`tutorials/`](tutorials/README.md) are **standalone** under DIMER Notebook Specification 2.0 (§4) — three are declared `TASK-INFERENCE` and one, the classification adaptation, `E2E` (mode `GUIDED`): generated by `tools/build_notebook.py` from four templates, each carries the package's modules (ten for the inference notebooks, twelve for the adaptation notebook), the model identity (`AutonLab/MOMENT-1-base` at the immutable revision `9fea447e740eb968a9e8d80c7562ae122bdb5dde`), the manifest digests and the runtime pins (`momentfm` as the commit-pinned source reference), so the exported notebooks run without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`). Each regenerates its deterministic sample in code, validates it into an input manifest, runs its task through the public API, writes an evaluation report (`not-measurable` for embeddings, `sample-sanity` with `masked_point_metrics` for imputation, `sample-sanity` with `top_k_recall` for the labelled anomaly sample), and exports outputs plus provenance.

| Notebook | Capability |
|---|---|
| `tutorials/moment_embeddings_colab.ipynb` | pooled pretrained embeddings |
| `tutorials/moment_imputation_colab.ipynb` | reconstruction-backed imputation with a known-truth patch holdout |
| `tutorials/moment_anomaly_detection_colab.ipynb` | raw reconstruction-residual anomaly ranking |
| `tutorials/moment_classification_colab.ipynb` | `E2E`: the inference contract on real windows, then a linear probe vs a bounded unfreeze of the last two encoder blocks on 179 UCI HAPT motion windows (six activities, 30 volunteers, CC BY 4.0, fetched at run time; split by volunteer), selected on validation and scored by held-out accuracy / macro-F1 against a majority floor and a k-NN vote; adapter export and reload parity |

See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — all four standalone tutorials have a clean-runtime execution of their exact committed blob recorded in `docs/release-verification.md` and `STATUS.md`: the three `TASK-INFERENCE` notebooks on Kaggle CPU (2026-09-14) and the `E2E` classification notebook blob `7c5c3732` (committed at `ac2f9ef`) on a clean Kaggle Tesla T4 runtime on 2026-09-19 (20/20 ok (1 restart after install cell), 294.3 s). Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted runs are. A later change to the carried modules or a notebook returns that notebook to Candidate until re-verified.

## Installation and tests

Requires Python 3.12 and `uv`.

```bash
uv sync --locked
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
uv run ruff check .
uv run pytest -m "not integration" -q
uv run pytest -m integration -q
```

`uv.lock` is the environment contract. `requirements.lock.txt` is the human-readable `uv export` rendering used by Colab bootstrap and CI parity checks. It is **not pip-installable** with ordinary `pip install -r`: the graph mixes hashed requirements with the exact unhashed Git source requirement for `momentfm`. Use `uv sync --locked` for a repository checkout. The Colab notebooks use `uv pip` deliberately rather than plain pip while consuming the parity-checked export.

The upstream source pin is:

```text
moment-timeseries-foundation-model/moment
38f7310ad594100747ca2a8357e9c7ca7d323e0e
```

The model checkpoint is pinned separately to Hugging Face revision:

```text
AutonLab/MOMENT-1-base
9fea447e740eb968a9e8d80c7562ae122bdb5dde
```

See [MODEL_CARD.md](MODEL_CARD.md) for file digests, safetensors load proof, runtime semantics, and limitations.

## Quickstart: embeddings

```python
import pandas as pd
from moment_pipeline import build_provenance, embed, load_moment, to_windows

frame = pd.read_csv("my_series.csv")
windows = to_windows(frame)
model = load_moment(task="embedding", device="cpu")
result = embed(windows, model)

embeddings = result.to_frame()
provenance = build_provenance(model, windows, result)
```

`reduction="mean"` averages channels inside upstream before patch pooling, so the result is one vector per window, not one vector per channel.

**Embedding missingness limitation:** upstream `MOMENT.embed` has no per-point observedness mask. A pre-filled missing value is visible to the encoder. The result and provenance therefore report missingness fractions explicitly; they do not claim the representation ignored missing values.

## Quickstart: imputation

```python
import numpy as np
from moment_pipeline import impute, load_moment, masked_point_metrics, to_windows

windows = to_windows(frame)
visible = np.ones_like(windows.input_mask, dtype=np.float32)
visible[:, 448:456] = 0.0  # deliberately hide one complete 8-step patch

model = load_moment(task="reconstruction", device="cpu")
result = impute(windows, model, mask=visible)
imputed = result.to_frame()
metrics = masked_point_metrics(result)
```

The default imputed product has three important semantics:

- source-observed values remain unchanged unless the caller explicitly hid them;
- source-missing or deliberately hidden cells receive reconstructed values;
- points hidden only because MOMENT expanded a mask to an 8-step patch are **not** overwritten merely because the model could not see them.

`masked_point_metrics()` scores only deliberately hidden source-observed cells. Source-missing values have no ground truth and never enter MAE/RMSE.

## Quickstart: anomaly scoring

```python
from moment_pipeline import load_moment, score_anomalies, to_windows

windows = to_windows(frame)
model = load_moment(task="reconstruction", device="cpu")
result = score_anomalies(windows, model, loss="mae", channel_aggregation="none")
scores = result.to_frame()
```

The score is an **unmasked self-reconstruction residual**, not a forecast residual. MOMENT sees the point it scores. A drift it reconstructs faithfully may therefore score low.

There is **no universal binary threshold in v1**. `AnomalyResult.threshold_policy` and exported provenance state this explicitly. Any threshold must be calibrated by the downstream application on an appropriate reference segment.

## Quickstart: classification adaptation

```python
from moment_pipeline import (
    adapt, evaluate, classify, knn_baseline, majority_baseline, class_names,
    fetch_sample_dataset, check_split_disjoint, save_artifact, load_artifact,
)
from moment_pipeline.model import load_moment

splits = fetch_sample_dataset()                       # 107 / 36 / 36 {id, x, label} windows, split by volunteer (79.6 MB fetch, digest-checked)
check_split_disjoint(splits)
model = load_moment(task="embedding", weights_dir="weights/moment-1-base")
classes = class_names(splits["train"])
print(majority_baseline([r["label"] for r in splits["train"]], [r["label"] for r in splits["test"]], classes)["accuracy"])
print(knn_baseline(model, splits["train"], splits["test"])["accuracy"])     # frozen embeddings, no training
adapter = adapt(model, splits["train"], splits["validation"])              # probe, then the last two blocks; selected on validation
print(adapter.policy, evaluate(model, adapter, splits["test"])["accuracy"])
print(classify(model, adapter, splits["test"][:3])["labels"])
save_artifact(model, adapter, "outputs/moment_adapter")                    # adapter.safetensors + manifest.json
fresh = load_moment(task="embedding", weights_dir="weights/moment-1-base")
reloaded = load_artifact(fresh, "outputs/moment_adapter")
```

Every window goes through the same long-format validation and canonical windowing as inference input (`records_to_long_frame` → `validate_long_frame` → `to_windows` → `embed`). The HAPT archive is CC BY 4.0 and is never redistributed by this repository.

## Canonical data contract

Input is long format:

```csv
series_id,timestamp,channel,value
A,2026-01-01T00:00:00,vibration,0.12
A,2026-01-01T00:15:00,vibration,0.18
```

| Policy | Behaviour |
|---|---|
| Channel ordering | sorted unique channel names |
| Windowing | one window per series, final 512 distinct timestamps |
| Alignment | right-aligned |
| Padding | shorter series left-padded; padding mask = 0 |
| Truncation | longer series keep the final 512; truncation is disclosed |
| Missing values | finite pre-fill, default `0.0`; missingness retained in masks |
| Patch semantics | 8-step non-overlapping patches; one missing point can hide the entire patch |
| Frequency | irregular spacing is surfaced, never silently interpolated |
| Normalization | delegated to MOMENT RevIN |

For multichannel windows, MOMENT's reconstruction mask has no channel axis. The public converter therefore collapses observedness conservatively across channels before patch quantization and records both point- and model-side masking fractions.

## CI and live-notebook evidence

Pull requests must pass:

- lock parity;
- Ruff for Python source/tests;
- no-network unit/contract suite;
- JSON parsing and Python compilation of every tutorial code cell.

Pushes to `main` and manual workflow dispatch additionally run:

- CPU integration tests against the real pinned `model.safetensors`;
- the exact code cells from all four tutorial notebooks, top-to-bottom (the classification notebook fetches the 79.6 MB HAPT archive and trains for several minutes on the runner);
- output-artifact upload (`moment-live-tutorial-outputs`).

Static notebook JSON validation alone is not considered release evidence.

## Repository layout

```text
docs/rfc/0001-moment-base.md
src/moment_pipeline/
  config.py
  validation.py
  canonical.py
  model.py
  embedding.py
  imputation.py
  anomaly.py
  provenance.py
examples/sample-data/
  DATASET_CARD.md
  SHA256SUMS
  generate_samples.py
tutorials/
  moment_embeddings_colab.ipynb
  moment_imputation_colab.ipynb
  moment_anomaly_detection_colab.ipynb
scripts/run_notebook.py
tests/
.github/workflows/ci.yml
```

## Licence

Pipeline code: MIT, Copyright (c) 2026 Kurt Valcorza — see [LICENSE](LICENSE).

Model weights and upstream `momentfm` code carry their own MIT terms. Their identities and provenance are documented separately in [MODEL_CARD.md](MODEL_CARD.md).

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
