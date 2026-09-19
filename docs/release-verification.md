# Release verification

The three tutorial notebooks (`moment_embeddings_colab.ipynb`, `moment_imputation_colab.ipynb`,
`moment_anomaly_detection_colab.ipynb`, all `TASK-INFERENCE`) are **release candidates** until each exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for the notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the four notebooks (three `TASK-INFERENCE`, one `E2E`) against its own template:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the four notebooks, each named in `tutorials/README.md` with its profile (`TASK-INFERENCE` ×3, `E2E` for the
  classification adaptation), the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile,
  spec `2.0`, `standalone: true` and `generated_from` (repository, module commit, the carried modules — ten, or twelve
  for the classification notebook — their combined SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary path;
  one cell tagged `embedded_module` per carried module of `src/moment_pipeline/` (ten, in dependency order: `config`, `model`,
  `validation`, `canonical`, `csvio`, `embedding`, `provenance`, `imputation`, `anomaly`, `roles`; the classification
  notebook adds `samples` and `adaptation`), each equal to its module
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
  `score_anomalies` + `top_k_recall`, `evaluation_report`, `build_provenance`; for the classification notebook `fetch_corpus`,
  `read_corpus`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `user_summary`, `records_to_long_frame`,
  `majority_baseline`, `knn_baseline`, `adapt(... trainable_blocks=0)` and `adapt(... trainable_blocks=TRAINABLE_BLOCKS ...)`,
  `evaluate`, `classify`, `save_artifact`, `load_artifact` and the reload-parity assertion), the ceiling print (`ResourceLimits`,
  the 512-step window, the 8-step patch), the exports, the learner-facing statements (no adaptation for the inference notebooks;
  the frozen / unfrozen policies, the majority floor, the k-NN vote, lowest validation log-loss, no dispersion estimate and the
  CC BY 4.0 licence for the classification notebook; upstream-vs-repository split, raw CSV header defence, representation /
  masked-evaluation / raw-residual semantics, no threshold) and the gated-off BYOD default
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
   - the carried module cells execute (define `load_moment`, `validate_inputs`, `evaluation_report` and the rest; for the
     classification notebook also `fetch_corpus`, `read_corpus`, `adapt`, `evaluate`, `save_artifact`, `load_artifact`) with no
     import of the repository package;
   - pinned `AutonLab/MOMENT-1-base` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/moment-1-base/`, `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reports all three manifest entries (`README.md`, `config.json`, `model.safetensors`) on a clean runtime,
     `verify_snapshot` returns the manifest dict with the digests equal to the pinned constants, and `load_moment(task=...,
     weights_dir=WEIGHTS_DIR)` returns a `LoadedMoment` whose identity names that revision and whose live-weight proof passes;
   - the synthetic sample regenerated in code with SHA-256 equal to the repository's `examples/sample-data/SHA256SUMS`
     (`moment_clean.csv` 34fc4758… for embeddings/imputation; `moment_anomaly.csv` 58855d89… + labels f35ec637… for anomaly);
     for the classification notebook, `fetch_corpus` fetching the UCI HAPT archive (79,596,192 bytes, SHA-256 `4ac4ae06…`)
     into `weights/hapt/`, `read_corpus` cutting 179 windows from the raw files of 30 volunteers, `build_sample_dataset` drawing
     18 / 6 / 6 volunteers (107 / 36 / 36 windows) with `check_split_disjoint` reporting no shared window and no shared
     volunteer, the three dataset digests `7ec96e15…` / `4490b2f9…` / `725d8711…`, `outputs/…_train.csv` written and the
     four dataset refusal probes each raising `ValueError`;
   - `validate_inputs` writes `outputs/<stem>_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     empty-channel probe) and the ceilings are printed;
   - the task path runs (`embed` / `impute` with the 8-step artificial holdout / `score_anomalies`); for the classification
     notebook: `embed` on three real test windows with the four sanity checks `True`, `evaluation_report` `not-measurable`,
     `build_provenance`; the majority floor (16.7 %), the cosine 5-NN vote (≈ 72 %) and the frozen policy
     (`adapt(trainable_blocks=0)`, ≈ 72 % / macro-F1 ≈ 0.71 on the test split) with the cell's assertion that the probe beats
     the floor; `adapt` printing epoch 0 as the probe (validation log-loss ≈ 0.746) then 3 unfreeze epochs of the last two
     blocks (14,158,848 trainable of 109,635,456 parameters plus the 4,614-parameter head) with validation log-loss /
     accuracy each epoch (0.746 → 0.703 → 0.751 → 0.605 in the recorded run) and the selected policy `unfrozen last 2 blocks
     + linear head` (`best_epoch` 3); `evaluate` on the validation and test splits with the four-way comparison (the cell
     asserts the selected model beats the floor — ≈ 78 % versus 16.7 % on the sample; the delta over the probe is reported,
     not asserted); six windows classified by the selected model and by the probe adapter over a fresh `load_moment`
     instance; `save_artifact` (20 tensors, about 56.7 MB) and `load_artifact` on a fresh instance with identical
     probabilities and test accuracy (asserted);
   - `evaluation_report` writes `outputs/<stem>_evaluation_report.json` — `not-measurable` (embeddings), `sample-sanity` with
     `masked_point_metrics` and the interpolation baseline (imputation), `sample-sanity` with `top_k_recall` (anomaly);
   - the task exports (`moment_embeddings.csv` + provenance; `moment_imputed_series.csv`, metrics, provenance, SVG;
     `moment_anomaly_scores.csv`, provenance, SVG; `moment_classification_train.csv`, `_predictions.csv`, the `_adapter/`
     directory) and `outputs/<stem>_result.json` written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime
     versions, device and dtype;
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
| 2026-09-14 | `moment_anomaly_detection_colab.ipynb` (`TASK-INFERENCE`) | `31ddb06` / `3c11d4ae` (blob unchanged at `ac2f9ef`) | Kaggle CPU (`kurtvalcorza/dimer-nb2-moment-anomaly-detection` v1; image torch 2.10.0+cpu) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 264.0 s | **PASSED** — 18/18 code cells ok (1 restart after install cell); 9 files, 454 MB staged from the Hub; run summary archived under `.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-moment-anomaly-detection/` in the workspace |
| 2026-09-14 | `moment_embeddings_colab.ipynb` (`TASK-INFERENCE`) | `31ddb06` / `b294013f` (blob unchanged at `ac2f9ef`) | Kaggle CPU (`kurtvalcorza/dimer-nb2-moment-embeddings` v1; image torch 2.10.0+cpu) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 221.1 s | **PASSED** — 17/17 code cells ok (1 restart after install cell); 9 files, 454 MB staged from the Hub; run summary archived under `.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-moment-embeddings/` in the workspace |
| 2026-09-14 | `moment_imputation_colab.ipynb` (`TASK-INFERENCE`) | `31ddb06` / `505e7430` (blob unchanged at `ac2f9ef`) | Kaggle CPU (`kurtvalcorza/dimer-nb2-moment-imputation` v1; image torch 2.10.0+cpu) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 252.7 s | **PASSED** — 18/18 code cells ok (1 restart after install cell); 9 files, 454 MB staged from the Hub; run summary archived under `.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-moment-imputation/` in the workspace |
| 2026-09-19 | `moment_classification_colab.ipynb` (`E2E`) | `ac2f9ef` / `7c5c3732` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-moment-classification` v1; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 5.16.1` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 294.3 s | **PASSED** — 20/20 code cells ok (1 restart after install cell); 10 files, 534 MB staged from the Hub into a clean cache; comparison {accuracy: {majority_floor: 0.1667, knn5: 0.7222, frozen_policy: 0.7222, selected_policy: 0.75}, macro_f1: {majority_floor: 0.0476, knn5: 0.7083, frozen_policy: 0.7145, selected_policy: 0.741}, log_loss: {frozen_policy: 0.6965, selected_policy: 0.5634}, per_class_recall: {laying: {knn5: 0.17, frozen: 0.33, selected: 0.33}, sitting: {knn5: 0.5, frozen: 0.33, selected: 0.5}, standing: {knn5: 0.67, frozen: 0.67, selected: 0.67}, walking: {knn5: 1, frozen: 1, selected: 1}, walking_downstairs: {knn5: 1, frozen: 1, selected: 1}, walking_upstairs: {knn5: 1, frozen: 1, selected: 1}}, delta_vs_frozen: {accuracy: 0.0278, macro_f1: 0.0265}, selected_policy: unfrozen last 2 blocks + linear head}; reload parity {probabilities_identical: True, accuracy_in_memory: 0.75, accuracy_reloaded: 0.75, classes_identical: True}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-moment-classification/v1/evidence/` in the workspace |
| 2026-09-19 | `moment_classification_colab.ipynb` (`E2E`) | `b0020d9` / `38bb74fc` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cpu`, `transformers 5.16.1`, `momentfm` at the pinned commit; snapshot and HAPT archive pre-staged) | Default sample path (install skipped via `DIMER_NOTEBOOK_CI_PREINSTALLED=1` → twelve carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 3 entries because the snapshot was pre-staged → `verify_snapshot` 3 files → `load_moment(task="embedding")` on CPU with the live-weight proof → `fetch_corpus` served from the pre-staged cache after its digest check → 179 windows cut from 30 volunteers, 107 / 36 / 36 drawn by whole volunteers (18 / 6 / 6) with `check_split_disjoint` clean, digests `7ec96e15…` / `4490b2f9…` / `725d8711…` → four dataset refusals → `validate_inputs` on three test windows with the duplicate-row refusal recorded → `embed` (0.34 s) with all four sanity checks `True`, `evaluation_report` `not-measurable`, `build_provenance` → majority floor 16.7 % → 5-NN vote 72.2 % / macro-F1 0.708 (18.7 s) → frozen-policy probe 72.2 % / 0.715 (log-loss 0.697; 76.8 s incl. features) → `adapt` with the unfreeze: 14,158,848 block + 4,614 head parameters, 3 epochs at 3e-4, 307.5 s, validation log-loss 0.746 (probe) → 0.703 → 0.751 → 0.605 with accuracy 69.4 / 69.4 / 61.1 / 77.8 %, selected `unfrozen last 2 blocks + linear head` at `best_epoch` 3 → **selected policy on the test split 77.8 % / macro-F1 0.776, log-loss 0.565 (Δ +0.056 accuracy, +0.062 macro-F1 vs the probe)**; per-class recall 0.50 / 0.50 / 0.67 / 1.0 / 1.0 / 1.0 (laying, sitting, standing, walking, downstairs, upstairs) → six predictions printed, one changed (`test-003`, standing → laying, the gold) → adapter 56,656,160 B / 20 tensors, SHA-256 `03acb36f…` → reload parity exact on a fresh `load_moment` instance (probabilities identical, test accuracy 0.777778 both ways) → six exports written | 430 s | **PASSED** — 20/20 code cells; pre-flight only, **not** promotion evidence; hosted clean-runtime run still required |

## Current status

**Release-grade.** All four standalone notebooks have a clean-runtime execution of their exact committed blob recorded above: the three `TASK-INFERENCE` notebooks on Kaggle CPU on 2026-09-14 (their blob ids are unchanged at `ac2f9ef`, so that evidence still identifies the shipped notebooks) and the `E2E` classification-adaptation notebook blob `7c5c3732` (committed at `ac2f9ef`) on a clean Kaggle Tesla T4 runtime on 2026-09-19 (20/20 ok (1 restart after install cell), 294.3 s, 10 files, 534 MB fetched from the Hub and digest-verified inside the notebook, the HAPT archive fetched and digest-verified by `fetch_corpus`) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight row above is what preceded it and remains history. Any later change to the carried modules or to a notebook produces a new blob, and the registry returns to **Candidate** for that notebook until a clean run of the new blob is recorded here.
