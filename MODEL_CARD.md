---
license: mit
model_card_spec: "1.1"
tags:
  - time-series
  - time-series-foundation-model
  - representation-learning
  - imputation
  - anomaly-detection
base_model: AutonLab/MOMENT-1-base
---

# MOMENT-1-base (v1.0) — Time-Series Foundation Model (Embeddings, Imputation & Anomaly Detection)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat)](https://huggingface.co/AutonLab/MOMENT-1-base)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white)](https://github.com/moment-timeseries-foundation-model/moment)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2402.03885-b31b1b.svg)](https://arxiv.org/abs/2402.03885)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pipeline](https://img.shields.io/badge/Pipeline-moment--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/moment-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides three ready-to-run interactive Google Colab notebooks, one per task head, each resolving the pinned `AutonLab/MOMENT-1-base` revision and exercising the repository's public API on bundled or your own series:

- **Embeddings Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_embeddings_colab.ipynb) [`moment_embeddings_colab.ipynb`](https://github.com/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_embeddings_colab.ipynb)  
  *Pretrained time-series representation extraction: encode windows into MOMENT embeddings and export them for downstream use.*

- **Imputation Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_imputation_colab.ipynb) [`moment_imputation_colab.ipynb`](https://github.com/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_imputation_colab.ipynb)  
  *Reconstruction-backed imputation: mask missing values, reconstruct them with the pretrained model, and score the reconstruction against ground truth.*

- **Anomaly Detection Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_anomaly_detection_colab.ipynb) [`moment_anomaly_detection_colab.ipynb`](https://github.com/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_anomaly_detection_colab.ipynb)  
  *Reconstruction-residual anomaly ranking: score each timestep by residual and rank anomalies; raw scores, no threshold is fitted.*

> [!NOTE]
> All three notebooks run on the default CPU runtime; no GPU is required.

---

###### Description

MOMENT-1-base is an open-weights time-series foundation model from the Auton Lab at Carnegie Mellon University (`AutonLab/MOMENT-1-base`, pinned revision `9fea447e740eb968a9e8d80c7562ae122bdb5dde`), packaged by this repository for representation learning, imputation, and anomaly scoring. Built on a patch-based encoder over a FLAN-T5-base backbone (~113.5M parameters in `model.safetensors`), it segments multi-channel time series into non-overlapping 8-step patches across a fixed 512-timestep window (64 patches, effective dimension 768) and maps them into latent representations. The released base checkpoint is fundamentally a self-reconstruction model (`task_name: reconstruction` with a pretrained `PretrainHead`), while classification and forecasting heads are untrained and deliberately refused. This repository provides a verified, robust DIMER pipeline wrapper: strict safetensors loading, supply-chain verification enforcing digest and byte-size checks, total exclusion of legacy pickle checkpoints, canonical data transformation with mandatory finite pre-fill, explicit patch-quantized masking contracts, raw anomaly residual scoring, and complete provenance tracking.

#### Intended Use and Limitations

###### Primary Intended Uses

The primary intended uses of this pipeline comprise three distinct zero-shot time-series tasks:
1. Multi-channel representation learning: Generating per-patch and pooled latent embeddings (`moment_pipeline.embedding.embed`) for downstream clustering, classification, or vector search.
2. Missing-value imputation: Reconstructing missing or masked time-series segments (`moment_pipeline.imputation.reconstruct`) via patch-quantized self-attention.
3. Reconstruction-based anomaly scoring: Emitting continuous, unthresholded reconstruction residuals (`moment_pipeline.anomaly.score_anomalies`) to highlight unexpected patterns.
Concrete application domains include industrial machine vibration telemetry, ECG and biometric monitoring, environmental sensor network recovery, and data center metrics. The pipeline serves as a standardized feature extractor and anomaly detector within the DIMER platform.

###### Primary Intended Users

Primary intended users are machine learning researchers, data scientists, industrial automation engineers, and MLOps professionals building time-series analytics workflows. Users are expected to understand patch-quantized masking mechanics—specifically that a single missing point masks an entire 8-step patch across all channels—and to understand that base embeddings are missingness-blind (finite pre-fill moves the vector). Users must also recognize that anomaly scores represent continuous reconstruction discrepancies rather than binary labels, and understand that supervised classification and forecasting require explicit task-head adaptation.

###### Out-of-scope use cases

1. **Capability boundaries:** Zero-shot forecasting and classification are strictly out of scope. The upstream checkpoint carries no pretrained forecasting or classification heads; upstream `momentfm` initializes them randomly, so exposing them without fine-tuning produces invalid outputs (for zero-shot forecasting, use `chronos-2-forecasting-pipeline`).
2. **Input boundaries:** Inputs must conform strictly to 512-timestep windows. Series shorter than 512 steps must be padded, and longer series must be sliced into 512-step windows. Unformatted, irregular, non-numeric, or infinite values are refused.
3. **Decision boundaries:** Autonomous, unmonitored decision-making based on raw anomaly scores—such as automatic shutdown of life-support systems, emergency grid tripping, or automated financial transactions—without human operator verification is strictly prohibited.

---

#### Factors

###### Groups

MOMENT-1-base is a numerical sequence model trained on the Timeseries-PILE benchmark, a broad multi-domain collection of synthetic, industrial, physical, and environmental time series. It does not model demographic or phenotypic human groups natively. However, the pretraining corpus was not demographically audited by its authors. When operators apply this pipeline to human biometric telemetry (e.g., ECG, photoplethysmography, gait analysis, or wearable health monitors), performance may vary across demographic categories such as age, biological sex, skin tone, or health status. Operators deploying on human subjects are obligated to perform independent subgroup validation and fairness audits on their own data.

###### Instrumentation

Training and evaluation data for MOMENT originate from diverse instrumentation, including industrial accelerometers, medical electrocardiographs, climate monitoring stations, server monitors, and laboratory sensors. Sensor sampling rate, frequency response, electrical noise, analog filtering, and ADC resolution directly influence the recorded time series. Because MOMENT operates on patches of 8 steps, high-frequency sensor noise or sudden phase jitter propagates through the patch projection layer into latent embeddings. While the pipeline sanitizes missing data via finite prefill and validates shapes, it cannot identify physical sensor drift or loss of calibration.

###### Environment

1. **Operating environment:** Requires Python 3.12, PyTorch >=2.1.2, and `transformers`. The pipeline executes on CPU using ~454 MB for weights, supporting `device="auto"`, `device="cpu"`, and `device="cuda"`. The pipeline enforces `float32` precision exclusively; half-precision dtypes (`float16`, `bfloat16`) are refused on the public API due to instability and unsupported CPU operations.
2. **Data environment:** Assumes regular, 512-step windows normalized via Reversible Instance Normalization (RevIN). The model degrades when applied to non-stationary series with abrupt variance explosions, windows containing >50% masked patches, or data whose sampling frequency drastically departs from typical physical phenomena.

---

#### Metrics

###### Performance Measures

For reconstruction and imputation tasks, performance is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE) computed strictly over masked target points (`masked_point_mae`, `masked_point_rmse`). For anomaly scoring, per-element residuals are computed under MSE or MAE. For embeddings, representation quality is judged downstream via silhouette scores, retrieval precision, or linear probe classification accuracy. Evaluating metrics over masked positions only is essential; including observed positions in reconstruction metrics artificially deflates error and masks poor imputation fidelity.

###### Decision thresholds

The anomaly scoring module deliberately ships **no binary threshold**—neither as a default, a constant, nor a keyword argument. Residual scales vary across different physical series, and a hardcoded threshold would be falsely interpreted as an empirical decision boundary. Output residuals are provided unthresholded, accompanied by explicit `threshold_policy` metadata. Downstream operators own threshold calibration against clean, holdout reference windows based on the asymmetric operational costs of false positives (unnecessary alarms) versus false negatives (missed failures).

###### Approaches to uncertainty and variability

Inference for embedding, imputation, and anomaly scoring is completely deterministic on CPU under standard runtime execution. The model uses no dropout or stochastic sampling at inference time. Output anomaly scores and reconstruction residuals are raw scalar distances, not statistical probabilities, p-values, or calibrated confidence intervals. Operators requiring calibrated uncertainty must apply conformal prediction, extreme value theory, or empirical quantiles over domain-specific validation splits.

---

#### Ethical considerations and biases

###### Data

MOMENT-1-base was pretrained on the Timeseries-PILE, an extensive compilation of publicly available time-series datasets spanning multiple domains. The upstream authors have not published a complete instance-level inventory of every private or sensitive artifact that might have been included in the public crawls. This repository distributes code, pipeline adapters, and tests; model weights are cached from Hugging Face Hub and never committed. Operators supplying inference data are responsible for auditing payloads to prevent accidental transmission of confidential, classified, or protected health information.

###### Human Life

MOMENT-1-base is not certified, tested, or approved for life-critical applications or high-stakes decisions concerning human life, safety, or health. It must not be deployed as an autonomous diagnostic device, clinical patient monitor, emergency safety trip, or automated criminal justice assessment tool. Use in any human-adjacent domain requires rigorous external clinical validation, fail-safe redundant systems, and continuous human clinician or expert supervision.

###### Mitigations

The pipeline implements extensive architectural and supply-chain mitigations:
1. **Supply-chain security:** Pins immutable revision `9fea447e…`, validates SHA-256 digests for `config.json` (`f1c66c2b…`) and `model.safetensors` (`1a436826…`), validates weight byte count (`453,940,120`), and strictly forbids legacy pickle files (`pytorch_model.bin`) via `allow_patterns` and snapshot inspection.
2. **Tensor identity proof:** At load time, all 116 non-head and head tensors are verified against `model.safetensors` using `torch.equal`.
3. **Numerical sanitization:** Implements mandatory finite prefill (`prefill_value`) to prevent NaN propagation during patch embedding, and explicitly surfaces `masked_point_fraction` and `masked_patch_fraction`.
4. **Head refusal:** Rejects untrained classification and forecasting heads at API boundary.
5. **Reproducibility:** Locks runtime dependencies via `uv.lock` and exports structured provenance with every result.

###### Risks and harms

Key risks include:
1. **Missingness blindness in embeddings:** Pre-filling missing values with a sentinel moves embedding vectors without leaving traces in the vector itself; downstream consumers must inspect provenance fractions to detect input degradation.
2. **Patch-edge anomaly artifacts:** Patch quantization (8 points per patch) can produce residual spikes at gap boundaries if unhandled; the pipeline sets unscored positions to NaN by construction.
3. **Automation bias:** Downstream operators interpreting raw reconstruction residuals as infallible anomaly alarms without establishing baseline calibration.

###### Use cases

Prohibited use cases include:
1. Deceptive surveillance, biometric tracking, or covert monitoring of individuals.
2. Automated workplace monitoring, employee productivity scoring, or predictive disciplinary action.
3. Autonomous deployment in lethal weapons systems or hazardous chemical/nuclear infrastructure.
4. Any usage violating the upstream MIT license or applicable regional AI regulatory mandates.

---

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

The loader's standard path:

1. downloads with `allow_patterns=["config.json", "model.safetensors", "README.md"]`, which cannot match `*.bin`;
2. refuses any snapshot directory containing a `.bin` file, before it checks anything else;
3. verifies the resolved commit, both digests and the expected weight byte size;
4. compares all 116 tensors against `model.safetensors` with `torch.equal`.

**What guarantees which file was loaded is (1) and (2), not (4).** `pytorch_model.bin` at this revision is a value-identical serialization of the same checkpoint, so a `.bin` load would satisfy the tensor comparison byte for byte. The comparison is still worth having — it discriminates a freshly initialized head, a partial or truncated load and a tampered file — but the file-identity claim rests on exclusion controls and digest checks. Pickle-format fallback is not part of the public path.

A `pytorch_model.bin` artifact exists upstream at the same model revision, so the safetensors-only acquisition rule is material rather than cosmetic. The DIMER loader does not silently substitute it.

## Public capability 1 — pretrained embeddings

`moment_pipeline.embed()` exposes the pretrained encoder representation using the upstream `reduction="mean"` behavior.

### Representation semantics

- one vector is produced per canonical window;
- channels are averaged inside upstream before patch pooling;
- the result is therefore **not** one embedding per channel;
- the reduction and channel policy are recorded on the result and in provenance;
- no random classification or forecasting head is involved.

### Embeddings are NOT missingness-aware

Embeddings are **NOT missingness-aware** in the current upstream path. Upstream's `MOMENT.embed` accepts the padding `input_mask`, but has no per-point observedness parameter. Finite pre-filled missing values are therefore visible to the encoder as values. RFC common-validation rule 12 is therefore not satisfiable through upstream `embed`.

DIMER does not hide this limitation. `EmbeddingResult` and provenance report `masked_point_fraction` and explicitly state that missingness was not visible to the model. For sensitive downstream clustering, retrieval, or classification, either use clean windows or perform an explicit imputation step first.

## Public capability 2 — imputation / reconstruction

`moment_pipeline.impute()` uses the pretrained reconstruction path while applying a DIMER-owned user-facing product contract.

### Mask semantics

MOMENT works on 8-step patches. A point-level missing or caller-hidden position can therefore hide the entire containing patch from the model. DIMER reports separately:

| Field | Meaning | Denominator |
|---|---|---|
| `masked_point_fraction` | source-missing non-padded cells | non-padded cells x channels |
| `model_masked_point_fraction` | non-padded positions hidden from the model after source missingness + caller mask | non-padded (window, position) pairs |
| `masked_patch_fraction` | non-padded patches hidden from the model | non-padded patches |

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
