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

# MOMENT-1-base

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat)](https://huggingface.co/AutonLab/MOMENT-1-base)
[![Upstream](https://img.shields.io/badge/Upstream-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white)](https://github.com/moment-timeseries-foundation-model/moment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

MOMENT-1-base is a pretrained time-series foundation model from the Auton Lab at Carnegie
Mellon University. It is a patch-based encoder built on a FLAN-T5-base backbone: a series
is cut into non-overlapping patches of 8 timesteps across a fixed 512-step context, each
patch is embedded, and a transformer encoder produces one representation per patch per
channel.

**The released base checkpoint is a reconstruction model.** Its `config.json` declares
`task_name: reconstruction`, and the only task head in `model.safetensors` is the
pretrained `PretrainHead` (`head.linear.weight`, `head.linear.bias`). That single fact
governs everything below.

## Model details

| | |
|---|---|
| Model identifier | `AutonLab/MOMENT-1-base` |
| Developer | Auton Lab, Carnegie Mellon University |
| Code repository | [moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment) |
| Family | MOMENT-1 |
| Task of the released checkpoint | reconstruction |
| Architecture | patch-based encoder over a FLAN-T5-base encoder backbone |
| Parameters | ~113.5M |
| Context (`seq_len`) | 512 |
| Patch length / stride | 8 / 8 (64 patches per window) |
| Effective `d_model` | 768 |
| Tensors in `model.safetensors` | 116 |

`config.json` declares `d_model: null`. The value is derived from the backbone when the
model is constructed; this pipeline reads the effective value off the loaded module
(768 — confirmed by executing `load_moment`) and records it in provenance rather than
hardcoding it from the config file.

## Pretrained vs adapted: what this pipeline will and will not expose

Only the reconstruction path carries pretrained weights.

**Exposed in v1** — defensible directly from the base checkpoint:

- pretrained encoder embeddings (`moment_pipeline.embedding.embed`);
- imputation / reconstruction (`moment_pipeline.imputation.reconstruct`);
- reconstruction-based anomaly **scoring** (Phase 4; the module is deliberately empty today).

**Not exposed, at any phase, without a separate adaptation contract:**

- classification — `momentfm` builds a **freshly initialized** `ClassificationHead`;
- forecasting (short or long horizon) — likewise a freshly initialized head. DIMER's
  zero-shot forecasting service is Chronos-2, not MOMENT.

`moment_pipeline.model.load_moment` accepts only `task="embedding"` or
`task="reconstruction"` and raises `ModelSourceError` for anything else, so an untrained
head cannot be reached through the public API by accident.

### The embedding-mode warning is about heads, not embeddings

Loading the embedding task emits:

> Only reconstruction head is pre-trained. Classification and forecasting heads must be
> fine-tuned.

`momentfm/models/moment.py::MOMENT._get_head` (L172-175 in the pinned build) raises this
for *every* non-reconstruction task name, including `embedding`. For the embedding task
the head that replaces `PretrainHead` is `nn.Identity` — there is nothing to fine-tune,
and the encoder producing the embeddings is fully pretrained. The warning is harmless
here and must not be read as "these embeddings require fine-tuning". This pipeline
asserts `type(pipeline.head) is nn.Identity` and that the module exposes **no** `head.*`
parameters in embedding mode.

## Masking is patch-quantized

MOMENT sees patches, not points. `momentfm/utils/masking.py::Masking.convert_seq_to_patch_view`
marks a patch observed only when **all 8** of its points are observed
(`mask.unfold(...).sum(dim=-1) == patch_len`).

One missing point therefore costs a whole patch. In a single-channel window a NaN at
index 13 masks patch 1: `masked_point_fraction = 1/512`, `masked_patch_fraction = 1/64`.

For multichannel windows the point mask is collapsed over channels with a `min`: MOMENT's
`mask` argument has no channel axis, so a point missing in **any** channel is treated as
unobserved. That is conservative and explicit -- and it means two honestly different
numbers exist. They have two different names:

| Field | Counts | Denominator |
|---|---|---|
| `masked_point_fraction` | source missingness, per (window, channel, position) cell | non-padded cells x channels |
| `model_masked_point_fraction` | positions hidden from the model: source missingness collapsed over channels **plus** any caller-supplied mask | non-padded (window, position) pairs |
| `masked_patch_fraction` | patches the model treats as unobserved | non-padded patches |

`masked_point_fraction` is defined once, in `canonical.missing_point_fraction`, and
`WindowSet`, `EmbeddingResult` and `ReconstructionResult` all report that same number for
the same windows. One missing point in `c1` of a 2-channel 512-step window is
`masked_point_fraction = 1/1024` and `model_masked_point_fraction = 1/512`; before this
was unified, both were called `masked_point_fraction` and read 1/1024 and 2/1024.

Every fraction is exposed on the result and recorded in provenance, and the mask handed to
`pipeline.reconstruct` is already expanded to patch granularity so callers see exactly the
masking the model applies.

## Missing values: mandatory finite pre-fill

`MOMENT.reconstruct` has **no** `nan_to_num` (unlike `MOMENT.embed`, which calls it at
L255), and `PatchEmbedding.forward` computes `mask * linear(x) + (1 - mask) * mask_embedding`
— multiplying a NaN by a zero mask yields NaN, so a raw NaN survives masking and reaches
the output. The repository keeps a negative control
(`tests/test_integration_model.py::test_negative_control_raw_nan_reaches_the_output`) that
executes this and asserts the output contains NaN.

The canonical converter therefore replaces every missing payload with a finite sentinel
(default `0.0`) before a tensor exists, and keeps missingness in `point_mask`. On the
**reconstruction** path that mask reaches the model: `pipeline.reconstruct` takes an
explicit patch-quantized `mask`, and a pre-filled position is embedded as
`mask_embedding` rather than as its sentinel value.

### Embeddings are NOT missingness-aware

On the **embedding** path the mask cannot reach the model, and this pipeline cannot make
it. Upstream's signature is

```python
MOMENT.embed(self, *, x_enc, input_mask=None, reduction="mean", **kwargs)
```

There is no per-point observedness parameter. `input_mask` is the *padding* mask, and
momentfm passes it to RevIN (`moment.py:254`) and to the patch embedding
(`moment.py:262`) as the only mask there is. The finite pre-fill is therefore consumed as
genuine observed data: it enters the RevIN mean and standard deviation and is embedded as
a value, not replaced by `mask_embedding`.

The consequences, all executed on the real weights and pinned by
`tests/test_integration_model.py::test_embeddings_are_missingness_blind_known_limitation`:

- the embedding of a window with missing values is **byte-identical** to the embedding of
  the same window with `prefill_value` written into those positions;
- both differ from the embedding of the clean window, so the sentinel demonstrably moves
  the vector;
- nothing in the vector distinguishes a fabricated stretch from a real flat one.

RFC common-validation rule 12 -- "missingness carried exclusively via the mask" -- is
therefore **not satisfiable through upstream `embed`**, and it is not satisfied here. What
this pipeline does instead is refuse to hide it: `EmbeddingResult` and every embedding
provenance block carry `masked_point_fraction`, `masked_point_count`,
`masked_patch_fraction`, `missingness_visible_to_model: false` and a
`missingness_policy` string. Those fractions are the **only** record that any part of the
input was fabricated, so a consumer clustering or retrieving on these vectors must read
them.

Whether v1 should keep embedding such windows with this disclosure, refuse windows above a
missingness threshold, or require a DIMER-owned imputation first is an open contract
decision for the repository owner; nothing in Phase 1 forecloses any of the three.

## Limitation of the future anomaly path

The Phase 4 anomaly score will be an **unmasked self-reconstruction residual**: the model
reconstructs a window it can see in full, and the residual is scored. It is not a forecast
residual, and it offers no guarantee of sensitivity to drift the model reconstructs easily.
Raw per-element scores `(batch, channel, timestep)` are the v1 output; there is no
universal binary threshold, and any channel aggregation is DIMER-owned and recorded.

## Supply chain

| Constant | Value |
|---|---|
| Pinned revision | `9fea447e740eb968a9e8d80c7562ae122bdb5dde` |
| `config.json` SHA-256 | `f1c66c2bb845229c0ed27a1600dbcc956b85ab21f9e5fd8a1663e6641bed7755` (949 bytes) |
| `model.safetensors` SHA-256 | `1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825` |
| `model.safetensors` size | 453,940,120 bytes |
| Weight file loaded | `model.safetensors` (guaranteed by the exclusion controls below) |

**`pytorch_model.bin` also exists at this revision** (453,978,525 bytes, SHA-256
`23c3d65bbb6dcd323352029e9fbe4ee3a3da0fff55b45ee4e00f38fff4e9bfb9`). A silent pickle
fallback is a live hazard, not a hypothetical one, so the standard path:

1. downloads with `allow_patterns=["config.json", "model.safetensors", "README.md"]`, which
   cannot match `*.bin`;
2. refuses any snapshot directory containing a `.bin` file, before it checks anything else;
3. verifies both digests and the weight byte size, and checks the snapshot directory name
   against the pinned commit — on **every** load, including when the caller passes its own
   `VerifiedSnapshot`. The identity written into provenance is rebuilt from the digests
   recomputed at that moment, so an export records what was verified, never what a caller
   asserted;
4. compares every non-`head.*` tensor in the file — and, for the reconstruction task, both
   `head.*` tensors as well, 116 of 116 — against `safetensors.safe_open` entries with
   `torch.equal`.

**What guarantees which file was loaded is (1) and (2), not (4).** `pytorch_model.bin` at
this revision is a value-identical serialization of the same checkpoint, so a `.bin` load
would satisfy the tensor comparison byte for byte. The comparison is still worth having —
it discriminates a freshly initialized head, a partial or truncated load and a tampered
file — but the *file identity* claim rests on `allow_patterns` never fetching a pickle
artifact and on the snapshot being refused outright if one is present. Both of those are
mutation-tested; the load proof carries the same statement in its
`file_identity_basis` field.

**The revision is pinned by the digests, not by an independent commit lookup.** The only
revision check is `verify_snapshot_dir`'s comparison of the snapshot directory name against
`PINNED_REVISION`. `assert_pinned_source` has already forced the *requested* revision to be
exactly that commit, and `huggingface_hub` names the snapshot directory after the commit the
request resolved to, so for a SHA request the two agree by construction and the check cannot
fail. It is a consistency assertion, not an oracle.

What actually carries the integrity claim here is the pair of SHA-256 digests: content that
hashes to the pinned values *is* the pinned revision's content, whatever a hub says about
it. The sibling `chronos-2-forecasting-pipeline` additionally asks the Hub which commit the
pin resolves to (`HfApi().model_info(...).sha`) and records whether that confirmation
actually ran. This pipeline does not, and its exported `model.revision_basis` says so rather
than implying a check that did not happen. Adding the lookup is a Phase-2 change; it buys a
second, independent witness, not a stronger content guarantee.

Mutable references (`main`, `latest`), other commits, other repositories, local directories
and `s3://` / `https://` sources are refused before any network call.

## Licences — weights and code are separate claims

- **Weights:** MIT, declared as `license: mit` in the Hugging Face model-card metadata at
  revision `9fea447e…`. **There is no `LICENSE` file in the model repository at that
  revision** (its files are `.gitattributes`, `README.md`, `config.json`,
  `model.safetensors`, `pytorch_model.bin`), so the licence claim rests on card metadata.
- **Upstream `momentfm` code:** MIT, CMU Auton Lab, installed from commit
  `38f7310ad594100747ca2a8357e9c7ca7d323e0e`.
- **This pipeline's code:** MIT, Copyright (c) 2026 Kurt Valcorza — see [LICENSE](LICENSE).

## Runtime requirements

- Python `>=3.12,<3.13`.
- CPU is sufficient; there is no hard CUDA requirement. `device="auto"` resolves to `cuda`
  only when it is actually available.
- Dependencies install from the committed lock (`uv sync --locked`); `requirements.lock.txt`
  is a hashed export for CI/Colab parity.
- `momentfm` is installed from source at an exact commit. Its version string is `0.1.5`,
  which **does not exist on PyPI** — the PyPI release is 0.1.4 and hard-pins
  `transformers==4.33.3`. Provenance records the commit alongside the version string.
- ~454 MB of weights are downloaded once into the Hugging Face cache. On Windows set
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

## Provenance

Every export carries `model`, `runtime`, `inference` and `load_proof` blocks: the pinned
revision and both digests, the proved weight file, the momentfm version *and* source
commit, `huggingface_hub` / `transformers` / `torch` / `numpy` versions, device and dtype,
and task-specific fields — `reduction` and the channel-averaging policy for embeddings,
masked point/patch counts and fractions for reconstruction.

## Deferred / out of scope

Short-horizon forecasting, long-horizon forecasting, classification inference without
adaptation, full backbone fine-tuning, a supervised or calibrated anomaly classifier, any
universal anomaly threshold, and multivariate anomaly aggregation beyond the explicit v1
policy. Classification adaptation is a later RFC in the order: frozen embeddings + a
classical classifier, then a trained MOMENT head, then PEFT/LoRA.

Phase 1 additionally does **not** ship the embedding export/tutorial (Phase 2), the
imputed export that preserves observed points or masked-point metrics (Phase 3), anomaly
scoring (Phase 4), serving latency instrumentation beyond one warm-up-corrected field
(Phase 5), sample datasets or Colab notebooks.

## Upstream references

- Model card and weights: <https://huggingface.co/AutonLab/MOMENT-1-base>
- Upstream code: <https://github.com/moment-timeseries-foundation-model/moment>
  (pinned at commit `38f7310ad594100747ca2a8357e9c7ca7d323e0e`)
- RFC contract for this pipeline: [`docs/rfc/0001-moment-base.md`](docs/rfc/0001-moment-base.md)
