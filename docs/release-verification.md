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
  each notebook byte-identical to `tools/build_notebook.py` output from its template; the single kernel cell that builds (or reuses, by lock digest) the isolated hash-locked uv environment and routes every later cell to it, with no `pip install` into the kernel and no restart request; `NOTEBOOK_SOURCE` recorded in exports;
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

### Hosted Colab CLI runs (isolated environment)

Executor: Colab CLI 0.7.4 sequential execution (`colab exec -f`) on a fresh Colab Tesla T4 VM per notebook, driven by
the workspace serial suite, which fetches the notebook byte-exact at the commit and refuses it unless its Git blob
matches. This is not a browser Run all: the executed files carry no execution counts, and cell order is evidenced by
`exec.log` ("Executing cell k/N"). Only the default path ran; the upload/BYOD branches and the optional "Change one
thing" activities were not exercised. The executed notebook, `run_summary.json` and `exec.log` of each run are
recorded byte-for-byte under `docs/execution-evidence/2026-10-07-3b0cee5/<notebook>/`. The imputation notebook's
first attempt lost the Colab connection before cell 1 (0 cells run, infrastructure) and was retried once on a fresh VM.

| Date (UTC) | Notebook | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|---|
| 2026-10-07 | `moment_anomaly_detection_colab.ipynb` (`TASK-INFERENCE`) | `3b0cee5` / `fe88efc016c8` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-moment-3b0cee5-bdc0`) | default labelled synthetic sample, `USE_BYOD = False`, `SCORE_CHANNEL` empty; isolated env (61 locked packages, Python 3.12.12) | 115.8 s | **one pass, no restart, 0 errors** — 19/19 code cells in order per `exec.log`; vibration top-3 recall 1.0 vs a 1.0 |z-score| baseline on the same channel; pooled ranks [1, 21, 45] of 512; temperature median residual 1.62 vs vibration 0.26; verdict `sample-sanity`; evidence `docs/execution-evidence/2026-10-07-3b0cee5/moment_anomaly_detection_colab/` (SHA-256: executed notebook `8b7c1207ab8e91793865724218a0d7d8833bd5e5535912d92e7534841ad856ed`, `run_summary.json` `fc925092aa727f3b8700999a7d6099b5471a084c5b93b78e40f92d6ecc71c937`, `exec.log` `bbd00bc78a09512ebcb4b68f3d61fcb02cb595e0deaaae15f174ac921c17e353`) |
| 2026-10-07 | `moment_classification_colab.ipynb` (`E2E`) | `3b0cee5` / `7b70813b2417` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-moment-3b0cee5-d81f`) | default HAPT split, `USE_BYOD = False`, default `TRAINABLE_BLOCKS`; isolated env (61 locked packages, Python 3.12.12) | 247.7 s | **one pass, no restart, 0 errors** — 21/21 code cells in order per `exec.log`; test (36 windows) accuracy / macro-F1: majority floor 16.7 % / 0.048, 5-NN 72.2 % / 0.708, frozen probe 72.2 % / 0.715, selected (unfrozen last 2 blocks, epoch 3) 75.0 % / 0.741, `delta_windows` 1 of 36; reload parity identical; evidence `docs/execution-evidence/2026-10-07-3b0cee5/moment_classification_colab/` (SHA-256: executed notebook `96dcf6d46150b453988f9af9a11adb31976dc4100da76b29303197d68739bec8`, `run_summary.json` `50117826e94da71f211ad7059ade2857d9aa008cecd2607bb2b543b651600bae`, `exec.log` `10b2220cb04305cfeec8ed3a6dbbc184247330c3ae8f6d59a75a4070516096ef`) |
| 2026-10-07 | `moment_embeddings_colab.ipynb` (`TASK-INFERENCE`) | `3b0cee5` / `7bfc3151e54e` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-moment-3b0cee5-cb4e`) | default synthetic sample, `USE_BYOD = False`; isolated env (61 locked packages, Python 3.12.12) | 104.9 s | **one pass, no restart, 0 errors** — 19/19 code cells in order per `exec.log`; embedding shape (1, 768), finite; padded 1/1, truncated 0/1; Section 6b cosine to the clean vector 0.907 (missing) vs 0.9999998 (interpolated), missing vs zeros-written max difference 0.0; verdict `not-measurable`; evidence `docs/execution-evidence/2026-10-07-3b0cee5/moment_embeddings_colab/` (SHA-256: executed notebook `e85935e0849d8911f5345ebb74b3bb2cdf08931d8c672cd2439a06836ddd7502`, `run_summary.json` `d37a27b9d6f79a4dd0e5b8f2997ddfcb993da8e3811bfc61f5e061bceeac554b`, `exec.log` `b8a981509350c3f9f10ac1ec8f13233ff260b358bbfb149fa3146b8cf8ec774c`) |
| 2026-10-08 | `moment_imputation_colab.ipynb` (`TASK-INFERENCE`) | `3b0cee5` / `50746e514d6b` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-moment-3b0cee5-82d8`) | default synthetic sample, `USE_BYOD = False`, interior holdout (positions 384-391); isolated env (61 locked packages, Python 3.12.12) | 110.5 s | **one pass, no restart, 0 errors** — 19/19 code cells in order per `exec.log`; pooled masked-point MAE 0.6045 vs 0.0523 for linear interpolation (model worse by 11.5x, as the notebook warns); 16 held-out points; verdict `sample-sanity`; evidence `docs/execution-evidence/2026-10-07-3b0cee5/moment_imputation_colab/` (SHA-256: executed notebook `14b022ccc2a3763ac324a88f02271353198f3c8ccfaa9d17283c2d295ed1cf54`, `run_summary.json` `134b87a1d8ecbd0776c41955498370b51080c32a14ec8ebf255920c4963d4170`, `exec.log` `1e69fee2c2dd8fecd01a191fdb035638e5bb160d381032635f9b7f818318f831`) |

## Current status

**Candidate.** The four notebook blobs at `3b0cee5` (generator /2.2: fleet-sweep and notebook-review fixes, the
isolated `uv` environment, worker `google.colab` stubs with module specs) are new, so the earlier Kaggle evidence above
no longer identifies the shipped notebooks, and those runs also needed one restart after the install cell. Each new blob
passed a Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 in one pass, with no restart and 0 errors
(2026-10-07/08; see "Hosted Colab CLI runs"): anomaly detection `fe88efc016c8`, classification `7b70813b2417`,
embeddings `7bfc3151e54e`, imputation `50746e514d6b`. Those runs exercised the default path only, not a browser Run
all; the BYOD branches and the optional activities were not run. Promotion to `Release-grade` is a review decision
against the procedure above for the exact release revision. Any later change to the carried modules or to a notebook
yields a new blob that needs its own recorded run.
