---
license: mit
tags:
  - time-series
  - time-series-foundation-model
  - representation-learning
  - imputation
  - anomaly-detection
base_model: AutonLab/MOMENT-1-base
---

# MOMENT-1-base — DIMER profile

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat)](https://huggingface.co/AutonLab/MOMENT-1-base)
[![Upstream](https://img.shields.io/badge/Upstream-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white)](https://github.com/moment-timeseries-foundation-model/moment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary

MOMENT-1-base is an open-weight pretrained time-series foundation model from the Auton Lab at Carnegie Mellon University. The released base checkpoint is a **reconstruction model**: its pretrained task head reconstructs masked time-series patches. DIMER therefore exposes only capabilities defensible directly from those weights:

1. pretrained encoder embeddings;
2. reconstruction-backed imputation;
3. raw reconstruction-residual anomaly scoring.

Classification and forecasting are not exposed as pretrained capabilities. Upstream can instantiate those heads, but they are freshly initialized and require a separate training/adaptation contract. DIMER's zero-shot forecasting path is Chronos-2.

## Model details

| Field | Value |
|---|---|
| Model | `AutonLab/MOMENT-1-base` |
| Developer | Auton Lab, Carnegie Mellon University |
| Family | MOMENT-1 |
| Released task | reconstruction |
| Architecture | patch-based encoder using a FLAN-T5-base encoder backbone |
| Approx. parameters | ~113.5M |
| Sequence length | 512 timesteps |
| Patch length / stride | 8 / 8 |
| Patches per full window | 64 |
| Effective representation dimension | 768 |
| Standard weight format | `model.safetensors` |
| License | MIT |

The effective representation dimension is read from the loaded model rather than inferred from the repository `config.json`, whose `d_model` field is not itself a reliable public representation-size contract.

## Checkpoint and source provenance

The DIMER path is immutable at both model and source-code layers.

| Item | Pin / assertion |
|---|---|
| Hugging Face revision | `9fea447e740eb968a9e8d80c7562ae122bdb5dde` |
| `model.safetensors` SHA-256 | `1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825` |
| `model.safetensors` size | 453,940,120 bytes |
| `config.json` SHA-256 | `f1c66c2bb845229c0ed27a1600dbcc956b85ab21f9e5fd8a1663e6641bed7755` |
| Upstream `momentfm` source | `moment-timeseries-foundation-model/moment` |
| Upstream source commit | `38f7310ad594100747ca2a8357e9c7ca7d323e0e` |
| Runtime lock | `uv.lock` + parity-checked `requirements.lock.txt` |

The loader's standard path downloads only the approved snapshot files, verifies the config and safetensors digests, verifies the expected byte size, and records the file identity in load proof/provenance. Pickle-format fallback is not part of the public path.

A `pytorch_model.bin` artifact exists upstream at the same model revision, so the safetensors-only acquisition rule is material rather than cosmetic. The DIMER loader does not silently substitute it.

## Public capability 1 — pretrained embeddings

`moment_pipeline.embed()` exposes the pretrained encoder representation using the upstream `reduction="mean"` behavior.

### Representation semantics

- one vector is produced per canonical window;
- channels are averaged inside upstream before patch pooling;
- the result is therefore **not** one embedding per channel;
- the reduction and channel policy are recorded on the result and in provenance;
- no random classification or forecasting head is involved.

### Missingness limitation

Embeddings are **not missingness-aware** in the current upstream path. `MOMENT.embed` accepts the padding `input_mask`, but no per-point observedness mask. Finite pre-filled missing values are therefore visible to the encoder as values.

DIMER does not hide this limitation. `EmbeddingResult` and provenance report source missingness fractions and explicitly state that missingness was not visible to the model. For sensitive downstream clustering, retrieval, or classification, either use clean windows or perform an explicit imputation step first.

## Public capability 2 — imputation / reconstruction

`moment_pipeline.impute()` uses the pretrained reconstruction path while applying a DIMER-owned user-facing product contract.

### Mask semantics

MOMENT works on 8-step patches. A point-level missing or caller-hidden position can therefore hide the entire containing patch from the model. DIMER reports separately:

| Field | Meaning |
|---|---|
| `masked_point_fraction` | source-missing non-padded cells |
| `model_masked_point_fraction` | non-padded positions hidden from the model after source missingness + caller mask |
| `masked_patch_fraction` | non-padded patches hidden from the model |

For multichannel input, the model mask is conservatively collapsed across channels because upstream's reconstruction mask has no channel axis.

### Imputed-series semantics

The default exported imputed series is intentionally narrower than full reconstruction:

- source-observed values are preserved unchanged unless the caller deliberately hid them;
- source-missing values are replaced by reconstruction;
- deliberately hidden source-observed values are replaced by reconstruction;
- observed neighbours hidden only because they share a model patch are **not** overwritten merely because the model could not see them;
- full reconstruction is exported separately.

This distinction prevents a tutorial imputation workflow from silently rewriting good observed data just because MOMENT's internal mask is patch-quantized.

### Evaluation semantics

`masked_point_metrics()` computes MAE/RMSE only on deliberately hidden source-observed cells. It excludes:

- source-missing cells, because no ground truth exists;
- observed neighbour cells hidden only by patch expansion, because they were not deliberate evaluation targets.

## Public capability 3 — anomaly scoring

`moment_pipeline.score_anomalies()` returns raw reconstruction residuals. The standard v1 primitive is an **unmasked self-reconstruction residual**.

### What the score means

The model sees the point it is scoring. The residual therefore measures how difficult that point/window is for MOMENT to reproduce, not how surprising it would have been before observation. It is not a forecast residual and should not be interpreted as one.

A slowly drifting pattern that MOMENT reconstructs faithfully may score low. Conversely, a local shape the model reconstructs poorly may score high even if it is operationally benign.

### Loss and aggregation

Supported residual losses:

- `mae` — absolute reconstruction error;
- `mse` — squared reconstruction error.

Channel aggregation is explicit and DIMER-owned:

- `none` — default; preserves per-channel scores;
- `mean`;
- `max`.

The selected loss and aggregation are recorded in provenance.

### No universal threshold

**The v1 pipeline ships no binary anomaly threshold.** There is no hidden default and no thresholded classifier output. `threshold_policy` explicitly records that raw scores only were produced.

Any binary decision boundary must be calibrated by the downstream application on an appropriate reference/calibration segment and evaluated separately. The deterministic tutorial sample includes injected spikes only to make ranking behavior inspectable; those labels are not a universal calibration set.

### Scored domain

A score is defined only where a position is non-padded and visible to the model. DIMER excludes:

- padding;
- source pre-filled positions whose residual would describe the sentinel rather than the series;
- observed points inside a patch hidden because of another missing value, because those residuals are imputation residuals rather than self-reconstruction residuals.

Unscored positions are marked `NaN` by construction. They are not NaNs propagated from the model; raw reconstruction error remains available separately for diagnostics.

## Canonical input contract

Preferred user input is long-format data:

```csv
series_id,timestamp,channel,value
A,2026-01-01T00:00:00,vibration,0.12
A,2026-01-01T00:15:00,vibration,0.18
```

Canonical conversion is deterministic:

- channel order: sorted unique channel names;
- one window per series in v1;
- final 512 distinct timestamps retained;
- shorter series left-padded;
- longer series truncated to the final 512 timestamps with truncation disclosed;
- missing payloads finite-pre-filled before tensor construction;
- source missingness retained in masks;
- irregular timestamp spacing surfaced, never silently interpolated;
- normalization delegated to upstream MOMENT RevIN.

Raw NaN is not passed to the reconstruction path. The finite pre-fill + explicit-mask rule is required because multiplying NaN by a zero mask does not make the value numerically safe inside the upstream model.

## Task-instance separation

MOMENT changes task heads according to task configuration. DIMER therefore uses task-specific loaded instances:

- `task="embedding"` for embeddings;
- `task="reconstruction"` for imputation and anomaly scoring.

The public loader refuses unsupported task names rather than giving users freshly initialized classifier/forecaster outputs that could be mistaken for pretrained predictions.

## Tutorials

The v1 repository contains three executable Colab tutorials:

- [`tutorials/moment_embeddings_colab.ipynb`](tutorials/moment_embeddings_colab.ipynb)
- [`tutorials/moment_imputation_colab.ipynb`](tutorials/moment_imputation_colab.ipynb)
- [`tutorials/moment_anomaly_detection_colab.ipynb`](tutorials/moment_anomaly_detection_colab.ipynb)

The default tutorial assets are deterministic synthetic series generated by [`examples/sample-data/generate_samples.py`](examples/sample-data/generate_samples.py). [`SHA256SUMS`](examples/sample-data/SHA256SUMS) records the expected materialized CSV digests and [`DATASET_CARD.md`](examples/sample-data/DATASET_CARD.md) documents their purpose and provenance.

The anomaly sample includes three explicitly injected vibration spikes for ranking demonstration. The labels do not establish a calibrated threshold or real-world benchmark accuracy.

## Reproducibility and CI

Pull requests must pass:

- lock parity;
- Python lint;
- no-network unit/contract tests;
- parse/compile checks for every tutorial code cell.

Pushes to `main` and manual workflow dispatch additionally run:

- real CPU integration tests against the pinned checkpoint;
- all three tutorial notebooks top-to-bottom using the same public library calls a user sees;
- output artifact upload for inspection.

Static notebook JSON validation alone is not accepted as evidence that the tutorials work.

## Intended use

Appropriate v1 use cases include:

- feature extraction from clean or explicitly preprocessed time-series windows;
- reconstruction-backed imputation where patch-level masking semantics are acceptable;
- exploratory/raw anomaly-score ranking where downstream calibration and domain interpretation remain explicit.

## Limitations

- Fixed 512-step canonical context in the current DIMER v1 converter.
- Patch length 8 means point missingness expands to patch-level model masking.
- Embeddings cannot receive per-point missingness masks upstream.
- No pretrained classification head is exposed.
- No MOMENT zero-shot forecasting claim is made.
- Anomaly scores are uncalibrated self-reconstruction residuals, not universal anomaly probabilities.
- No universal anomaly threshold is provided.
- Fairness, robustness, calibration, and domain-transfer behavior are application-dependent and not established by the tutorial samples.
- The current release boundary is a tutorial/developer preview; stable production-serving contracts remain follow-on work.

## License

Pipeline code in this repository is MIT licensed. The pinned MOMENT model and upstream source also declare MIT terms; their identities are recorded separately because model artifacts, upstream code, and this wrapper are distinct provenance layers.

## References

- Hugging Face model: `AutonLab/MOMENT-1-base`
- Upstream code: `moment-timeseries-foundation-model/moment`
- DIMER model/task contract: [`docs/rfc/0001-moment-base.md`](docs/rfc/0001-moment-base.md)
- Release-completion tracking: issue #5
