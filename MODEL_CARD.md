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

One missing point therefore costs a whole patch. A single NaN at index 13 masks patch 1:
`masked_point_fraction = 1/512`, `masked_patch_fraction = 1/64`. Both fractions are
exposed on every reconstruction result and recorded in provenance, and the mask handed to
`pipeline.reconstruct` is already expanded to patch granularity so callers see exactly the
masking the model applies.

For multichannel windows the point mask is collapsed over channels with a `min`: MOMENT's
`mask` argument has no channel axis, so a point missing in **any** channel is treated as
unobserved. That is conservative and explicit.

## Missing values: mandatory finite pre-fill

`MOMENT.reconstruct` has **no** `nan_to_num` (unlike `MOMENT.embed`, which calls it at
L255), and `PatchEmbedding.forward` computes `mask * linear(x) + (1 - mask) * mask_embedding`
— multiplying a NaN by a zero mask yields NaN, so a raw NaN survives masking and reaches
the output. The repository keeps a negative control
(`tests/test_integration_model.py::test_negative_control_raw_nan_reaches_the_output`) that
executes this and asserts the output contains NaN.

The canonical converter therefore replaces every missing payload with a finite sentinel
(default `0.0`) before a tensor exists, and carries missingness only in the masks. MOMENT
normalizes internally via RevIN driven by `input_mask`.

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
| Weight file loaded | `model.safetensors` (proved at load time) |

**`pytorch_model.bin` also exists at this revision** (453,978,525 bytes, SHA-256
`23c3d65bbb6dcd323352029e9fbe4ee3a3da0fff55b45ee4e00f38fff4e9bfb9`). A silent pickle
fallback is a live hazard, not a hypothetical one, so the standard path:

1. downloads with `allow_patterns=["config.json", "model.safetensors", "README.md"]`, which
   cannot match `*.bin`;
2. refuses any snapshot directory containing a `.bin` file, before it checks anything else;
3. verifies the resolved commit, both digests and the weight byte size;
4. compares live encoder tensors — and, for the reconstruction task, both `head.*` tensors —
   against `safetensors.safe_open` entries with `torch.equal`. That comparison, not a
   filename, is the proof of which file was loaded.

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
