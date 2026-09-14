# Release verification

The three tutorial notebooks (`moment_embeddings_colab.ipynb`, `moment_imputation_colab.ipynb`,
`moment_anomaly_detection_colab.ipynb`, all `TASK-INFERENCE`) are **release candidates** until each exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for the notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the three notebooks against its own template:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the three notebooks, each named in `tutorials/README.md` with its `TASK-INFERENCE` profile, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and
  `generated_from` (repository, module commit, the ten carried modules, their combined SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary path;
  one cell tagged `embedded_module` per module of `src/moment_pipeline/` (ten, in dependency order: `config`, `model`,
  `validation`, `canonical`, `csvio`, `embedding`, `provenance`, `imputation`, `anomaly`, `roles`), each equal to its module
  after the generator's documented rewrites (the `DEFAULT_WEIGHTS_DIR` rule plus the removal of package-relative imports);
  the inline `MANIFEST` equal to the committed `weights/moment-1-base/dimer-base-manifest.json` (3 files); the inline `PINS`
  equal to the `pyproject.toml` runtime pins with `momentfm` carried as the `[tool.uv.sources]` commit-pinned direct reference;
  each notebook byte-identical to `tools/build_notebook.py` output from its template; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest, which each
  notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same identity
  string appears in `README.md` and `MODEL_CARD.md` with no stray revisions (the `momentfm` source commit is whitelisted);
- the profile-specific public-API calls per notebook (`stage_missing_files`, `verify_snapshot`, `load_moment(task=...,
  weights_dir=...)`, `validate_long_frame`, `to_windows`, `validate_inputs`, then `embed` / `impute` + `masked_point_metrics` /
  `score_anomalies` + `top_k_recall`, `evaluation_report`, `build_provenance`), the ceiling print (`ResourceLimits`, the 512-step
  window, the 8-step patch), the exports, the learner-facing statements (no adaptation, upstream-vs-repository split, raw CSV
  header defence, representation / masked-evaluation / raw-residual semantics, no threshold) and the gated-off BYOD default
  listed in the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import
  on the primary path, an unpinned `git+https://` dependency, a mutable `revision='main'`, direct `momentfm` /
  `MOMENTPipeline` / `snapshot_download` / `from huggingface_hub import` / `from transformers import` use **outside the carried
  module cells**, any `worker.run(` / `worker_cli(` / `subprocess.run([` outside the generator-owned install cell,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the checkpoint provenance section.

CI also installs the locked environment (`uv sync --locked`), runs `ruff`, `tools/build_notebook.py --check` for each
template, and the offline unit suite (`tests/`, including `test_fleet_snapshot.py`, `test_role_helpers.py`,
`test_notebook_parity.py`; no weights, injected downloader). These are source/provenance and unit checks. They are **not**
execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; a notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebooks are standalone) |
| Repository CI integration job (`scripts/run_notebook.py`) | GitHub-hosted Ubuntu runner, the locked `uv` environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes each standalone notebook's code cells sequentially against the real pinned weights in a scratch working directory; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`, for **each** notebook:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle executor above) with
   **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults for the
   sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`,
   `momentfm` at the pinned source commit);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access other than the commit-pinned `momentfm` source install;
   - the ten carried module cells execute (define `load_moment`, `validate_inputs`, `evaluation_report` and the rest) with no
     import of the repository package;
   - pinned `AutonLab/MOMENT-1-base` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/moment-1-base/`, `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reports all three manifest entries (`README.md`, `config.json`, `model.safetensors`) on a clean runtime,
     `verify_snapshot` returns the manifest dict with the digests equal to the pinned constants, and `load_moment(task=...,
     weights_dir=WEIGHTS_DIR)` returns a `LoadedMoment` whose identity names that revision and whose live-weight proof passes;
   - the synthetic sample regenerated in code with SHA-256 equal to the repository's `examples/sample-data/SHA256SUMS`
     (`moment_clean.csv` 34fc4758… for embeddings/imputation; `moment_anomaly.csv` 58855d89… + labels f35ec637… for anomaly);
   - `validate_inputs` writes `outputs/<stem>_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     empty-channel probe) and the ceilings are printed;
   - the task path runs (`embed` / `impute` with the 8-step artificial holdout / `score_anomalies`);
   - `evaluation_report` writes `outputs/<stem>_evaluation_report.json` — `not-measurable` (embeddings), `sample-sanity` with
     `masked_point_metrics` and the interpolation baseline (imputation), `sample-sanity` with `top_k_recall` (anomaly);
   - the task exports (`moment_embeddings.csv` + provenance; `moment_imputed_series.csv`, metrics, provenance, SVG;
     `moment_anomaly_scores.csv`, provenance, SVG) and `outputs/<stem>_result.json` written with `NOTEBOOK_SOURCE`, model
     revision, model licence, runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, transformers, device), model identifier and
   immutable revision, whether the model cache was clean, outcome, produced outputs, and any warning or applicable `SHOULD`
   deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with `git rev-parse <commit>:tutorials/<notebook>`).
Wall times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model
download; they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Notebook | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|---|
| 2026-09-14 | `31ddb06` / `505e74302b0d` | Kaggle CPU (`kurtvalcorza/dimer-nb2-moment-imputation` v1) | Default sample path | 252.7 s | **PASSED** — 18/18 ok code cells executed cleanly, 9 files, 454 MB staged |
| | `moment_imputation_colab.ipynb` | | | Default sample path | | pending — queued to the GPU lane |
| | `moment_anomaly_detection_colab.ipynb` | | | Default sample path | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of any standalone notebook has been recorded yet; the runs are **pending** and queued to the
GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
cell, and the offline unit suite passed on the tutorial sources at the candidate revision, which is necessary but not
sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run against each notebook blob
under review and an integrator promotes it; promotion is not performed by the builder. Three facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real `hf_hub_download` fetch of
all three manifest entries into a fresh `weights/moment-1-base/` has not been executed); `load_moment(weights_dir=...)` was
exercised only with a stand-in `momentfm` module (the real `MOMENTPipeline.from_pretrained` on the manifest-described
directory has not been executed); and the standalone carrier itself — executing the ten carried module cells in a runtime
that has no repository checkout — has been validated statically only (parity PASS, carrier probe up to the fetch), never run.
The earlier repository-installing notebooks did run in CI against the real weights at `9fea447e740eb968a9e8d80c7562ae122bdb5dde`; that evidence predates the
standalone carrier and the fleet snapshot scheme and does not transfer to it.