# MOMENT tutorial registry

**DIMER Notebook Specification:** `1.0`

The three user-facing notebooks intentionally remain separate because embeddings, imputation, and anomaly scoring have different input/output and evaluation semantics. Each notebook is a `TASK-INFERENCE` workflow: it consumes the pretrained, immutable `AutonLab/MOMENT-1-base` checkpoint and performs no gradient training, fine-tuning, in-context conditioning, or fitted preprocessing.

| Notebook | Profile | Capability | Default runtime | BYOD | Release gate |
|---|---|---|---|---|---|
| [`moment_embeddings_colab.ipynb`](moment_embeddings_colab.ipynb) | `TASK-INFERENCE` | pooled pretrained embeddings | CPU / float32 | Yes | current revision must pass live-notebook CI |
| [`moment_imputation_colab.ipynb`](moment_imputation_colab.ipynb) | `TASK-INFERENCE` | reconstruction-backed imputation | CPU / float32 | Yes | current revision must pass live-notebook CI |
| [`moment_anomaly_detection_colab.ipynb`](moment_anomaly_detection_colab.ipynb) | `TASK-INFERENCE` | raw reconstruction-residual anomaly ranking | CPU / float32 | Yes | current revision must pass live-notebook CI |

## Common learning contract

All three notebooks use the repository's production-facing API, the locked dependency graph, deterministic repository-generated samples by default, optional gated BYOD, explicit input validation/canonicalization, the pinned model identity and integrity-verification path, machine-readable result/provenance exports, and a closing statement of what successful execution proves and does not prove.

The canonical BYOD schema is long format with columns `series_id`, `timestamp`, `channel`, and `value`. Operational limits and padding/truncation behavior are surfaced before inference in every notebook. BYOD contents are not sent by this pipeline to an external inference service; users remain responsible for whether their notebook runtime is authorized to process sensitive data.

## References

- [Repository README](../README.md)
- [Model card](../MODEL_CARD.md)
- [Tutorial sample dataset card](../examples/sample-data/DATASET_CARD.md)
- [Upstream MOMENT code](https://github.com/moment-timeseries-foundation-model/moment)
- [AutonLab/MOMENT-1-base](https://huggingface.co/AutonLab/MOMENT-1-base)

## Release verification

Static JSON parsing and Python-cell compilation run on pull requests, but they are not execution evidence. Pushes to `main` and manual workflow dispatch execute the notebooks top-to-bottom on CPU against the real pinned `model.safetensors` and upload the generated tutorial outputs. A notebook revision is release-grade only after that clean-runtime gate succeeds for the revision being released.

Applicable `SHOULD` deviations must be recorded in the pull request that introduces them and kept discoverable here or in another durable release note after merge. None are intentionally declared by this registry.
