<!-- Source issue: https://github.com/kurtvalcorza/moment-pipeline/issues/1 -->
<!-- Captured at: 2026-09-08T04:34:15Z -->
<!-- Verbatim copy; the issue is mutable, this file is the anchor. -->

## Status

**RFC v2 / Integrator-amended implementation proposal** — revised after Agent Relay review. This version is the builder contract pending one final reviewer pass.

## Summary

DIMERify **MOMENT-1-base** as a shared open-weight pretrained time-series foundation-model pipeline.

MOMENT is **not** treated as a turnkey zero-shot classifier or long-horizon forecaster. The released base checkpoint is a pretrained reconstruction/encoder model. DIMER v1 therefore exposes only capabilities that are defensible from the pretrained base weights.

### Approved v1 capability set

1. **time-series embeddings / representation extraction**;
2. **imputation / reconstruction**;
3. **reconstruction-based anomaly scoring**.

### Deferred from v1

- short-horizon forecasting;
- classification;
- long-horizon forecasting;
- trained anomaly classifiers/thresholds.

Classification and forecasting remain later **adaptation** workflows using explicit task heads / probes / fine-tuning.

---

## Integrator decisions from RFC review

The following reviewer findings are accepted into this RFC:

- **M-1 CONFIRMED:** only the reconstruction head is pretrained; classification and standard forecasting heads require adaptation. The core base-vs-adapter architecture is approved.
- **M-2:** missing values MUST be pre-filled with finite values before model entry and represented through masks; the standard imputation/anomaly entry points must never receive raw NaNs.
- **M-3:** missingness is patch-quantized; expose/report both masked-point fraction and masked-patch fraction and construct tutorial masking at patch granularity.
- **M-4:** short-horizon forecasting is **removed from v1** and deferred. Chronos-2 remains DIMER's zero-shot forecasting service.
- **M-5:** embedding reduction/pooling is part of the public contract and must be recorded; multichannel semantics must be explicit.
- **M-6:** anomaly scores are DIMER-defined per-channel/per-timestep reconstruction residuals; raw scores are the v1 output and there is no universal default binary threshold.
- **M-7:** `momentfm==0.1.5` alone is not an immutable runtime pin; lock the complete runtime and prove which weight file is loaded.
- **M-8:** embedding mode may replace the reconstruction head with identity; task-specific loaders/instances must be explicit. The notebook should pre-empt misleading upstream warnings in embedding mode.
- **M-9:** the model identity below is concretely pinned and the model-side license/metadata have been independently rechecked.
- **X-1:** tests must discriminate the intended behavior, not merely assert attribute presence.
- **X-2:** the first implementation PR must copy this RFC into `docs/rfc/0001-moment-base.md` and link the issue to the resulting commit.

---

## Upstream model identity and supply chain

### Model

- **Model:** `AutonLab/MOMENT-1-base`
- **Family:** MOMENT-1
- **Task of released base checkpoint:** reconstruction
- **Parameters:** ~113.5M
- **Base architecture:** patch-based time-series encoder using a FLAN-T5-base encoder backbone
- **Sequence length:** 512
- **Patch length:** 8
- **Stride:** 8
- **License:** MIT
- **Preferred weight:** `model.safetensors`
- **Pinned Hugging Face revision:** `9fea447e740eb968a9e8d80c7562ae122bdb5dde`
- **`model.safetensors` SHA-256:** `1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825`
- **`model.safetensors` size:** 453,940,120 bytes

### Supply-chain invariants

The implementation MUST:

1. resolve only the pinned official Hugging Face revision in tutorial/production-facing paths;
2. verify the revision commit;
3. verify `model.safetensors` SHA-256 and byte size;
4. calculate and record `config.json` SHA-256 before Phase 1 merge;
5. record the model-repository license/source at that revision;
6. use `model.safetensors`; the standard path MUST NOT silently load `pytorch_model.bin`;
7. include a runtime test proving which weight file was actually loaded;
8. record revision + config digest + weight digest in `MODEL_CARD.md` and exported provenance.

The immutable model revision is the primary integrity anchor; file digests are secondary assertions.

---

## Runtime/dependency reproducibility

Upstream project metadata may report `momentfm` version `0.1.5`, but that string is not sufficient as a source-code identity.

Required:

- if using PyPI: pin `momentfm==0.1.5` and the exact wheel/hash;
- if using source: pin an exact upstream commit SHA;
- explicitly pin Hugging Face Hub, Transformers, PyTorch, NumPy, and the complete resolved dependency graph;
- commit a reproducible lock (`uv.lock`, hashed requirements, or equivalent);
- CI and Colab install from the same lock;
- record Python, `momentfm`, `huggingface_hub`, Transformers, PyTorch, device, and dtype in provenance.

Do not install mutable upstream `main` in production-facing tutorials.

---

## Critical semantic distinction: base model vs adapted task heads

The released base checkpoint uses reconstruction weights.

DIMER MUST NOT:

- present a freshly initialized classification head as a pretrained classifier;
- present a freshly initialized forecasting head as a pretrained zero-shot forecaster;
- export untrained task-head outputs as production predictions.

DIMER MAY:

- expose pretrained encoder embeddings;
- use the pretrained reconstruction path for imputation;
- derive reconstruction-error anomaly scores;
- later add separately trained/adapted task heads under their own artifact/version contracts.

---

## Scope

### In scope for v1

- one pinned `MOMENT-1-base` checkpoint;
- reusable Python loader/validator outside notebooks;
- embeddings;
- imputation/reconstruction;
- reconstruction-based anomaly scoring;
- long-format BYOD conversion into the canonical tensor representation;
- CPU/GPU runtime selection where supported;
- deterministic model acquisition/integrity verification;
- structured provenance;
- sample datasets + dataset cards;
- fresh-Colab tutorials for the three approved v1 tasks.

### Explicitly out of scope for v1

- short-horizon forecasting;
- long-horizon forecasting;
- classification inference without adaptation;
- full backbone fine-tuning;
- supervised/calibrated anomaly classifier;
- universal anomaly threshold;
- multivariate anomaly aggregation semantics beyond the explicit v1 policy below;
- arbitrary ragged/irregular semantics without explicit padding/masking;
- DIMER backend/server implementation beyond a stable portable inference contract.

### Later RFC / phase candidates

- frozen-embedding linear-probe classification;
- MOMENT classification-head training;
- PEFT/LoRA classification adaptation;
- long-horizon forecasting head training;
- calibrated anomaly detection;
- optional revisiting of short-horizon reconstruction forecasting after dedicated validation.

---

## Canonical DIMER-facing data representation

### Long-format input

Preferred user format:

```csv
series_id,timestamp,channel,value
A,2026-01-01T00:00:00,signal_1,0.12
A,2026-01-01T01:00:00,signal_1,0.18
```

### Internal canonical tensor

Convert deterministically to:

```text
x_enc:      (batch, channels, sequence_length)
input_mask: (batch, sequence_length)
```

The conversion layer MUST document:

- deterministic channel ordering;
- windowing policy;
- right/left alignment policy;
- padding convention;
- truncation behavior;
- sequence-length target (`512` for the pinned base config);
- missing-value handling;
- timestamp/frequency assumptions;
- normalization delegated to/performed by MOMENT.

---

## Common validation contract

Required:

1. required columns exist;
2. IDs are non-null;
3. timestamps parse when provided;
4. no duplicate `(series_id, channel, timestamp)` rows;
5. deterministic channel ordering;
6. numeric/coercible values under an explicit policy;
7. frequency irregularity is surfaced; no silent interpolation;
8. sequence/window padding/truncation is disclosed before inference;
9. empty channels or all-missing windows are rejected;
10. resource limits cover batch size, windows, channels, and sequence counts;
11. raw NaNs are **not** passed to MOMENT reconstruction/anomaly paths;
12. missing positions are finite-pre-filled and represented only through explicit masks.

### Mandatory finite pre-fill contract

For missing/masked values:

- replace missing numerical payloads with a finite sentinel appropriate to the normalized model input, defaulting to `0.0` unless the tested preprocessing contract requires otherwise;
- carry missingness exclusively via the mask;
- test that NaN-bearing BYOD input yields finite model outputs after canonical preprocessing;
- never rely on multiplying NaNs by zero masks.

---

## Task 1 — embeddings

### Goal

Expose the pretrained MOMENT encoder as a reusable feature extractor.

### Public contract

Input: validated windows.

Output:

```text
series_id
window_id
embedding_0 ... embedding_n
```

plus provenance.

### Embedding reduction semantics

The v1 implementation MUST choose and document an explicit representation policy.

Recommended v1 default:

- use a documented pooled embedding mode that produces one vector per window;
- record `reduction` / pooling strategy in provenance;
- for multichannel inputs, explicitly state whether channels are averaged, pooled independently, or combined by a DIMER-owned policy.

Do not imply the chosen pooled embedding is identical to the representation consumed by a later native classification head unless proven.

### Acceptance/oracle

- eval mode;
- identical input/config => deterministic embedding within defined numeric tolerance;
- expected output dimension asserted for the pinned config;
- no random classification/forecasting task head introduced;
- task-specific loader/instance semantics tested.

The tutorial/model card should explain any harmless upstream warning associated with embedding task initialization so users are not told the embeddings themselves require fine-tuning.

---

## Task 2 — imputation / reconstruction

### Goal

Use the pretrained reconstruction path to fill deliberately missing/masked points.

### Mask semantics

The pinned model operates at **patch length 8**. One missing point can cause a whole patch to be treated as unobserved by the model.

Therefore DIMER MUST:

- expose `masked_point_fraction`;
- expose `masked_patch_fraction`;
- explain patch quantization in `MODEL_CARD.md` and tutorial;
- create tutorial artificial masks at patch granularity by default;
- if arbitrary point masks are accepted, disclose the resulting patch masking expansion.

### Required output

- original observed series;
- imputed/reconstructed values;
- explicit point mask;
- explicit derived patch mask or summary;
- masked-point count/fraction;
- masked-patch count/fraction;
- provenance/runtime metadata.

Observed points remain unchanged in the default exported **imputed** series. Full reconstruction may be exposed separately.

### Evaluation

When generating tutorial missingness:

- retain ground truth only for scoring;
- remove/mask values before model inference;
- score **only masked target points**;
- report MAE and RMSE on masked points;
- report masking fractions alongside the metric.

### Required discriminating tests

- NaN-containing BYOD input is finite-pre-filled and produces finite reconstruction;
- the implementation calls one pinned reconstruction entry point with an explicit mask;
- a missing point's patch expansion is correctly reflected in `masked_patch_fraction`;
- observed values remain unchanged in the default imputed export.

---

## Task 3 — anomaly scoring

### Goal

Use MOMENT reconstruction error as an anomaly **score**, not a universally calibrated anomaly classifier.

### Raw score semantics

For v1, define the primitive score as per-element reconstruction residual:

```text
(batch, channel, timestep)
```

with a documented loss, e.g. absolute or squared reconstruction error.

DIMER MUST document that this is an **unmasked self-reconstruction residual** for the standard anomaly path. It is not a forecast residual and does not guarantee sensitivity to slowly reconstructable drift.

### Output

```text
series_id
timestamp
channel
reconstruction
reconstruction_error
anomaly_score
```

Raw scores are mandatory.

### Channel aggregation

Any aggregate score is DIMER-owned and must be explicit, e.g.:

- max over channels;
- mean over channels;
- no aggregation (default safest mode).

Record aggregation strategy in metadata.

### Threshold policy

**No universal binary threshold in v1.**

If tutorial calibration is demonstrated, it must be an explicit optional step using a designated calibration/reference segment and must not include held-out labelled anomalies used for evaluation.

Potential later strategies:

- user threshold;
- percentile calibrated on a reference segment;
- documented robust-statistics rule.

### Evaluation when labels exist

- AUROC/AUPRC where appropriate;
- precision/recall/F1 only after an explicitly calibrated threshold;
- event-level metrics only in a later domain-specific extension.

### Required discriminating tests

- NaN-bearing input produces finite anomaly scores after finite pre-fill;
- score shape matches `(batch, channel, timestep)` before optional aggregation;
- no hidden/default binary threshold is applied;
- aggregation strategy changes only the documented aggregation dimension.

---

## Deferred — short-horizon forecasting

Short-horizon reconstruction forecasting is **not a v1 capability**.

Reason for deferral:

- effective context is reduced in patch-sized increments;
- horizon validation is insufficient upstream;
- padded windows can be semantically mishandled;
- Chronos-2 already provides the intended DIMER zero-shot forecasting service.

A future RFC may revisit it only with explicit horizon bounds, full-length/right-aligned window rules, padding rejection/handling, and effective-context provenance.

---

## Deferred — classification adaptation

Preferred later order:

### Option A — frozen MOMENT embeddings + classical classifier

Safest first supervised adaptation path.

### Option B — frozen backbone + trained MOMENT classification head

Requires explicit train/validation/test split and serialized head artifact.

### Option C — PEFT/LoRA

Only after A/B are reproducible.

For any adapted classifier:

- group/subject leakage controls where relevant;
- train-only fit of adaptation/scaling;
- adapter/head artifact stored separately from immutable base identity;
- model version records base revision + adapter hash + training provenance.

---

## Output/provenance contract

Every v1 mode exports structured metadata including:

```json
{
  "model": {
    "name": "AutonLab/MOMENT-1-base",
    "revision": "9fea447e740eb968a9e8d80c7562ae122bdb5dde",
    "config_sha256": "<recorded during Phase 1>",
    "weights_sha256": "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825",
    "license": "MIT"
  },
  "runtime": {
    "python": "...",
    "momentfm": "...",
    "huggingface_hub": "...",
    "transformers": "...",
    "torch": "...",
    "device": "cpu|cuda",
    "dtype": "..."
  },
  "inference": {
    "task": "embedding|imputation|anomaly",
    "sequence_length": 512,
    "patch_length": 8,
    "n_series": 32,
    "n_channels": 1,
    "latency_seconds": 0.42
  }
}
```

Task-specific metadata additionally records:

- embedding reduction/pooling;
- masked point + patch fractions;
- anomaly loss and channel aggregation.

---

## Sample datasets and tutorials

Provide at least one defensible sample per approved v1 task, reusing one dataset across tasks only if the pedagogy remains clear.

Each sample MUST have:

- source URL;
- license/redistribution basis;
- row/series/channel counts;
- frequency;
- preprocessing;
- intended task;
- deterministic archive generation if zipped;
- SHA-256 verification if remotely fetched.

Fresh-Colab tutorials:

1. `moment_embeddings_colab.ipynb`
2. `moment_imputation_colab.ipynb`
3. `moment_anomaly_detection_colab.ipynb`

Each must run top-to-bottom and use the same locked runtime as CI.

---

## Runtime / serving requirements

- `device=auto|cpu|cuda` where supported;
- no hard CUDA requirement;
- actual device/dtype recorded;
- warm-up before latency measurement;
- latency includes preprocessing needed by the public contract;
- explicit resource guards;
- no raw user-series logging by default;
- task-specific pipeline instances/loaders are explicit because changing MOMENT task configuration may replace heads;
- reusable inference modules exist outside notebooks.

---

## Security requirements

- immutable official model revision only;
- verified `model.safetensors` load path;
- reject silent `.bin` fallback;
- revision + config + weight integrity checks;
- safe archive extraction;
- file type/size limits;
- no arbitrary code execution from datasets;
- provenance preserved in every export.

---

## Required tests

### Supply-chain / loader

- exact HF revision resolves;
- config + weight digest verification;
- loaded file is proven to be `model.safetensors`;
- mutable/unapproved source rejected in standard path;
- environment lock reproduced in CI.

### Common validation

- duplicate IDs/channel/timestamps;
- bad numeric values;
- irregular frequency surfaced;
- empty/all-missing windows rejected;
- finite pre-fill transforms NaNs without losing missingness mask;
- deterministic channel/window ordering.

### Embeddings

- deterministic oracle;
- output dimension oracle;
- explicit reduction metadata;
- no random task head.

### Imputation

- finite outputs with original NaNs;
- explicit mask actually used;
- patch expansion oracle;
- masked-point-only metric;
- observed points preserved in imputed export.

### Anomaly

- finite outputs with original NaNs;
- raw score shape oracle;
- documented loss;
- no implicit threshold;
- aggregation behavior tested.

### Integration

- real CPU tiny-sample smoke path for all three tasks;
- GPU smoke optional;
- bundled sample resolver/hash path;
- BYOD path;
- fresh-Colab smoke execution where feasible.

Static notebook JSON checks alone are insufficient.

---

## Repository structure

```text
moment-pipeline/
├── README.md
├── MODEL_CARD.md
├── pyproject.toml / runtime lock
├── docs/rfc/0001-moment-base.md
├── src/moment_pipeline/
│   ├── config.py
│   ├── validation.py
│   ├── model.py
│   ├── provenance.py
│   ├── embedding.py
│   ├── imputation.py
│   └── anomaly.py
├── tutorials/
├── examples/sample-data/
├── tests/
└── .github/workflows/ci.yml
```

---

## Implementation phases

### Phase 1 — foundation

- version this RFC in-repo;
- lock runtime;
- verified safetensors loader;
- model card;
- canonical data converter;
- finite pre-fill + masking contract;
- CPU smoke test.

### Phase 2 — embeddings

- pooled representation contract;
- deterministic embedding tests;
- sample + tutorial + BYOD;
- export/provenance.

### Phase 3 — imputation

- explicit point/patch masks;
- patch-granularity tutorial masking;
- masked-point evaluation;
- export/provenance.

### Phase 4 — anomaly scoring

- reconstruction residual score;
- optional documented aggregation;
- raw-score tutorial;
- optional calibration demonstration separated from core inference.

### Phase 5 — serving readiness

- stable task dispatch / task-specific instances;
- resource limits;
- latency instrumentation;
- packaging contract for later DIMER integration.

Adapted classification/forecasting require separate follow-on RFCs.

---

## Acceptance criteria

- [ ] This RFC is copied to `docs/rfc/0001-moment-base.md` and commit-linked from this issue.
- [ ] HF revision `9fea447e740eb968a9e8d80c7562ae122bdb5dde` resolves and is recorded.
- [ ] Weight SHA-256/size above are verified.
- [ ] `config.json` SHA-256 is recorded and verified.
- [ ] MIT model provenance is documented separately from code provenance where needed.
- [ ] Complete environment is reproducibly locked.
- [ ] Standard loader proves `model.safetensors` was loaded and rejects silent `.bin` fallback.
- [ ] Base-vs-adapter semantics are explicit in code/docs.
- [ ] v1 exposes only embeddings, imputation, and anomaly scoring.
- [ ] Short-horizon forecasting is absent from v1 public APIs/tutorials.
- [ ] Missing values use finite pre-fill + explicit masks.
- [ ] NaN regression tests produce finite imputation/anomaly outputs.
- [ ] Patch masking semantics and both point/patch fractions are documented/tested.
- [ ] Embedding pooling/reduction is explicit and serialized.
- [ ] Imputation scores only masked target points.
- [ ] Anomaly output defaults to raw reconstruction scores with no universal threshold.
- [ ] Real CPU smoke tests pass for all v1 tasks.
- [ ] Bundled samples are licensed, deterministic, and hash-verified.
- [ ] BYOD tutorials run top-to-bottom in fresh Colab.
- [ ] `MODEL_CARD.md` documents pretrained-vs-adapted semantics, patch masking, anomaly-score limitations, runtime requirements, provenance, and future/deferred task scope.
- [ ] CI exercises runtime/sample/model-loading paths rather than static notebook structure only.

## Integrator disposition

**Base-model architecture approved. MOMENT v1 is intentionally narrowed to embeddings, imputation, and reconstruction-based anomaly scoring. Builder may start only after the final reviewer confirms this amended RFC contract.**
