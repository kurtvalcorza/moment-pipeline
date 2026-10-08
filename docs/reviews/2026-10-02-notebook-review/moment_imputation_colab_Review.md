# MOMENT Imputation Tutorial Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/moment-pipeline`  
**Notebook:** `tutorials/moment_imputation_colab.ipynb`  
**Reviewed commit:** `44526760c75d43a4f314252fef297562c9651a32` (`main`, confirmed with `gh api repos/kurtvalcorza/moment-pipeline/commits/main`)  
**Notebook Git blob:** `505e74302b0d7ab7b528130281e06e56cdf74683`. This is the blob executed in the recorded Kaggle CPU run of 2026-09-14 (commit `31ddb06`); the notebook has not changed since.  
**Finding prefix:** `MIM`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The plumbing is careful. The notebook digest-verifies the pinned snapshot and proves every head tensor is live, regenerates and digest-checks the synthetic sample, validates the input into a manifest with a recorded `EMPTY_CHANNEL` rejection, keeps source missingness apart from the artificial holdout, scores only withheld truth, preserves observed values in the exported series, and writes metrics, report and provenance. This review's local CPU run reproduced the hosted metrics exactly (MAE 1.121483564376831) and the hosted imputed values to 1.2e-7.

The demonstration that all this plumbing carries does not hold up:

1. **The reconstruction does not track the series, and the notebook never says so (MIM-B1).** On the hosted run, the model's MAE on the 16 withheld points is **1.121**, against **0.171** for the notebook's own baseline (6.6×). This review moved the 8-step holdout across all 32 patch positions of the sample: the model lost to the baseline at **32 of 32**, median interior MAE **0.99 vs 0.062** (16×). The reconstruction of points the model *can see* is no better than predicting the series mean (Pearson **0.087** temperature, **0.012** vibration; MAE 1.597 vs 1.581 for the mean). Called directly through upstream `reconstruct()` on a fully visible pure sine, the output is uncorrelated with the input (Pearson **−0.010**). The same weights under upstream momentfm's own `transformers==4.33.3` give the same numbers to 7 significant figures, so this is not a transformers-5 incompatibility. The notebook prints the two MAEs one line apart and moves on; nothing tells the learner that the "imputation" is noise around the mean, and the registry lists the notebook as Release-grade.
2. **No one-pass `Run all` (MIM-M1).** The recorded Kaggle run of this exact blob failed in the install cell (`numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.`) and passed only after a restart; the record calls this PASSED.
3. **The default holdout is the last 8 steps of the series (MIM-M2).** With no observed point after the gap, the task is a 2-hour extrapolation, which the notebook says it does not demonstrate (forecasting), and the "linear interpolation" baseline is a flat hold of the last observed value (it predicts 26.131526 and 1.280415 at every withheld step).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `TASK-INFERENCE` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (`metadata.dimer.notebook_spec`, opening cell) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated as such. Prerequisites: "basic Python and pandas; what a long-format time-series table is" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default; CUDA used when present; float32 only |
| Promised outcomes | pinned install; carried package (10 modules); digest-verified snapshot; deterministic sample or BYOD CSV; validation into an input manifest; one 8-step patch hidden with known truth; "pretrained reconstruction-backed time-series imputation"; evaluation only on withheld truth; a linear-interpolation baseline comparison; imputed series preserving observed values; metrics, report and provenance |
| Learning objectives | install; read the package guarantees; verify the model revision; generate the sample or bring a CSV; validate/canonicalize; "hide one complete 8-step patch with known truth; impute and evaluate only withheld truth"; "compare a simple interpolation baseline"; export |
| Explicitly not demonstrated | forecasting, classification, uncertainty intervals, production fitness |

### Existing execution evidence

- `docs/release-verification.md` line 133: Kaggle CPU (image torch 2.10.0+cpu), commit `31ddb06` / blob `505e7430` (= the reviewed blob), 2026-09-14, **PASSED — 18/18 code cells ok (1 restart after install cell)**, 252.7 s, 454 MB staged. `tutorials/README.md` lists the notebook as **Release-grade**.
- The archived run (`.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-moment-imputation/v1/evidence/`, workspace only): `run_summary.json` (`restarted_after_install_cell: true`; pass 1 `ok: false`, 211.6 s, `RuntimeError: Core dependencies changed while older modules were loaded: numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.`; pass 2 ok, 41.0 s), `executed-pass1.ipynb`, `executed.ipynb`, and the 7 output files. Hosted metrics: MAE 1.1215 / RMSE 1.5365 vs baseline 0.1707 / 0.2031, verdict `sample-sanity`.
- No hosted run covers BYOD, source-missing input, an interior holdout, or a Colab runtime.

### Evidence obtained by this review

- **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`, 4 torch threads), venv `dimer-moment` (Python 3.12.10, torch 2.14.0+cpu, numpy 2.5.3, pandas 3.0.5, transformers 5.16.1, momentfm at the pinned commit). Install cell run with the notebook's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`; the 3 snapshot files pre-staged by copy (`fetched = []`) and then verified by the notebook; `HF_HUB_OFFLINE=1`; `google.colab.files.upload` was a shim. `run_probes_t433.py` re-ran the model in an isolated scratch venv (Python 3.11, torch 2.4.1+cpu, `transformers 4.33.3`, numpy 1.25.2, momentfm at the same commit, `--no-deps`) on the same verified `model.safetensors`. **None of this is a Colab run.** Scale: the default sample at full default size (1 series, 2 channels, 256 steps); BYOD probes use small synthetic CSVs (≤ 840 rows).
- **P1 static:** JSON parses (nbformat 4.5); 18 code cells compile; no persisted outputs; blob equals the recorded-run blob. `tools/build_notebook.py --check` exit 0; `tools/validate_release_assets.py` exit 0 (PASS). Section 2 carries 124,812 characters of module code.
- **P2 default path (direct, CPU):** 18/18 cells ok in 23.1 s wall (model load 13.4 s, impute 0.3 s). Holdout positions 504–511 = the last 8 of 256 valid positions. MAE 1.121483564376831, RMSE 1.536515474319458, identical to the hosted run; imputed values match hosted to 1.2e-7. Per channel: temperature model MAE **2.047** vs baseline 0.172 (series std 1.78); vibration 0.196 vs 0.170 (std 0.26). Baseline predictions are a single constant per channel equal to the last observed value; 0 observations follow the holdout. Observed values preserved: true. Section 3 printed `{'device': None, 'source': 'local-snapshot'}`.
- **P3 holdout placement (direct, CPU, same model instance):** one 8-step patch hidden at each of the 32 valid patch starts (256…504). Model better than the baseline at **0 of 32**; interior (30 patches) median MAE **0.991** vs **0.062**. Without any artificial mask the report verdict is `not-measurable` (correct). A 512-step version of the same formulas (no padding) gives MAE 1.21 / 1.22 / 1.14 at starts 256 / 384 / 504, so left-padding is not the cause.
- **P5 reconstruction fidelity (hosted exported series, documented execution):** on the 248 visible points per channel, Pearson(reconstruction, original) = **0.087** (temperature) and **0.012** (vibration); reconstruction MAE 1.597 / 0.249 vs 1.581 / 0.226 for predicting the series mean.
- **P6 upstream `reconstruct()` direct, fully visible input (transformers 5.16.1):** `head_type = PretrainHead`; pure sine (period 32): Pearson **−0.0102**, MAE 0.688 vs 0.635 for the mean; sine + trend (period 96): Pearson 0.111, MAE 1.596 vs 1.593. `mask` all-ones and `mask` omitted give identical output.
- **P7 same weights, upstream's declared `transformers 4.33.3`:** identical to P6 to 7 significant figures (Pearson −0.010197 / 0.110801), and the tutorial sample with the trailing holdout gives masked MAE 1.1214834 (temperature 2.0472, vibration 0.1958) — the notebook's number. Interior patch 384: 0.604.
- **P4 BYOD and recovery:** a two-series, two-channel CSV (`flow`, `pressure`; A 300 h with 5 interior and 3 trailing NaN `flow` values; B 120 h) runs Sections 4–9 through `DIMER_BYOD_PATH`: holdout moved to 496–503 (the trailing source gap removes the last complete patch), source `masked_point_fraction` 0.0095, `model_masked_point_fraction` 0.038, `masked_patch_fraction` 0.058; 20 observed `pressure` points hidden from the model only because `flow` was missing at those patches (not overwritten); MAE 0.824 vs baseline 0.130; only window `A::w0` is scored; the SVG's last 96 `flow` rows all belong to series **B**, so the held-out patch in A is not in the plot. Invalid inputs: duplicate header (`DUPLICATE_COLUMNS`) in cell 27; non-numeric value (`NON_NUMERIC_VALUE`), duplicate key (`DUPLICATE_ROWS`) in cell 29; a 6-step series → `ValueError: The first canonical window has no complete 8-step source-observed patch to hold out. Provide a series with at least 8 consecutive observed timestamps.` All raise before the model runs. A cancelled upload raises `ValueError: Upload exactly one CSV …`; `USE_BYOD=True` outside Colab raises `ModuleNotFoundError: No module named 'google'`.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | Supply-chain verification, validation, mask accounting, withheld-truth scoring and observed-value preservation are correct and deterministic (local = hosted). The install forces a restart on hosted images (MIM-M1). The model output that everything feeds is uncorrelated with its input, under both transformers 5.16.1 and 4.33.3 (MIM-B1; cause not determined). |
| Promise fulfilment | "Reconstruction-backed imputation" is not delivered in any useful sense: the imputed values are no better than the series mean and lose to a flat hold everywhere (MIM-B1). The default holdout is extrapolation, not a gap (MIM-M2). Source-vs-artificial masking, withheld-truth scoring and observed-value preservation are delivered. |
| Learner experience | Section prose is precise and the limits section is honest about sample-only evidence. But the learner is never asked to read the model-vs-baseline result and is never told it is a 6.6× loss; the guided layer (prediction, checkpoints, troubleshooting, conclusion template) is thin, and there is no control for where the holdout goes (MIM-m2). |
| Spec conformance | RUN1/RUN10/ENV6/REL2/REL11 fail on the recorded run (MIM-M1). §21.4 (source vs artificial masking, withheld truth only) met. RUN8/UX1/EVAL3/EVAL15 not met for the central result (MIM-B1). EVAL10 baseline present but mislabelled (MIM-M2). SHOULD gaps: GDL1–GDL14 (partial), UX4/UX5, EXE1/EXE2/EXE5, UX12/SRC3. Declared spec 2.0, not 2.2. |

## 3. Promise and objective tracing

| Claim (cell) | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes without intervention (0) | cell 3 pip install + stale-module guard | Hosted pass 1 raises restart error | Learner must restart and rerun | **Not met** (MIM-M1) |
| Pinned, digest-verified model; effective identity and device printed (24) | cell 25 | 3 files verified; `device: None`, fallback `source` literal | Device not shown where promised | Met except display (MIM-m3) |
| Deterministic sample, digest asserted (26) | cell 27 | digest `34fc4758…` | Clear | Met |
| Validate before the model; disclose padding/missingness (28) | cell 29 | manifest; padded 1/1; `EMPTY_CHANNEL` finding | Clear | Met |
| Hide one complete 8-step patch with known truth (0, 28) | cell 29 `complete_patch_starts[-1]` | positions 504–511, the series' last 8 steps | Reads as a gap; it is a trailing extrapolation | **Misdescribed** (MIM-M2) |
| Pretrained reconstruction-backed imputation (0, 30) | cell 31 `impute` | imputed values near the series mean; visible-point Pearson ≈ 0 | Learner assumes a working imputer | **Not delivered** (MIM-B1) |
| Score only withheld truth (32) | cell 33 `masked_point_metrics` | n = 16 | Clear | Met (§21.4) |
| Compare a linear-interpolation baseline (0, 32) | cell 33 | 0.171 vs model 1.121 | Never interpreted; actually a flat hold | **Not delivered** (MIM-B1, MIM-M2) |
| Observed values preserved (30, 34) | `imputed = where(replacement, …)` | P2 true; P4 20 patch-coupled points untouched | Clear | Met |
| Export with provenance (36) | cell 37 | 7 files | Clear | Met |
| BYOD through the same path (0, 26) | cell 27 | P4: end to end; 4 coded rejections | Works via env var / Colab upload; fails on Jupyter | Met in Colab (MIM-m6) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Hide a patch and evaluate only withheld truth | Run cells; read positions and counts | Exercised (printed positions, n = 16) |
| Compare a simple interpolation baseline | Read two printed numbers | Printed, never interpreted; comparison inverted from what the lesson implies (MIM-B1) |
| Validate / canonicalize | Run cell; read manifest and rejection | Exercised |
| Read how source and model masking diverge | "Next experiment" only (BYOD with gaps) | Not exercised on the default path; this review's P4 shows it works but needs guidance (MIM-S3) |

## 4. Journeys

| Journey | Evidence basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | Precise per-section prose. Blocked at cell 3 on hosted images (restart). Shown MAE 1.12 beside baseline 0.17 with no reading guidance; the limits section frames the question as whether MOMENT "beats the interpolation baseline reliably", so the learner has no cue that it lost by 6.6×. No prediction, checkpoint or conclusion template. |
| Clean default | Documented execution (Kaggle CPU, same blob: pass 1 fails at install, pass 2 passes); direct execution (local CPU, install skipped, 18/18 ok, metrics identical to hosted) | **Not one-pass.** Passes after a restart; outputs belong to the run; the central result is a model that does not track the input (MIM-B1). |
| Active learning | Direct execution (local CPU) | No in-notebook exercise or holdout control. The suggested next experiment (BYOD with genuine gaps) runs and the three fractions diverge as described (0.0095 / 0.038 / 0.058), but cross-channel patch coupling (20 `pressure` points hidden) is not explained. This review's holdout sweep (32 positions, < 10 s CPU) is the experiment a learner would need, and it shows 0/32 wins. |
| Reuse and recovery | Direct execution (local CPU; upload shim) | BYOD via `DIMER_BYOD_PATH` reaches every stage; only window 0 is held out and scored; the plot can show a different series than the one held out. 4 invalid inputs rejected with coded or actionable messages before the model; cancelled upload actionable; `USE_BYOD=True` on Jupyter → `ModuleNotFoundError`. Colab upload widget: **not verified**. |

## 5. Findings

### Blocker

#### MIM-B1 — The reconstruction does not track the series; imputation loses to the baseline everywhere, and the notebook presents it as working

- **Cell/section:** Sections 6–7 (cells 30–33; `tools/notebook_template_imputation.py`), "Interpretation and limits" (cell 38); `tutorials/README.md` registry row; `MODEL_CARD.md` imputation sections.
- **Observed issue:** the model's withheld-point MAE is 1.121 against 0.171 for the baseline (temperature 2.047 vs 0.172, larger than the channel's standard deviation of 1.78). Across all 32 holdout positions the model never beats the baseline (interior median 0.99 vs 0.062). The reconstruction of *visible* points is no better than the series mean (Pearson 0.087 / 0.012), and upstream `reconstruct()` called directly on a fully visible sine returns output uncorrelated with it (Pearson −0.010). The same weights under upstream's declared `transformers 4.33.3` give identical numbers, so the transformers-5 runtime is not the cause. The head tensors are proven equal to the pinned file (`prove_pinned_weights_are_live`). This review did not establish why the pinned checkpoint behaves this way (candidate causes: checkpoint/revision behaviour, the upstream call convention for the base model, or something common to both stacks); no repository test checks reconstruction fidelity. The notebook prints both MAEs and never interprets them; its limits section ("does not prove … that MOMENT beats the interpolation baseline reliably") implies it does beat it here.
- **Consequence:** the central demonstration teaches the opposite of what the data show. A learner leaves believing they have run a working pretrained imputer and that the pipeline's imputed product is meaningful, when its imputed values are noise around the series mean. The recorded run is labelled Release-grade.
- **Evidence:** documented execution (hosted metrics and exported series, P5); direct execution (P2, P3, P6 on transformers 5.16.1; P7 on transformers 4.33.3); source inspection (cells 31–33, 38; `src/moment_pipeline/model.py:530`).
- **Recommended correction:** first establish the cause outside the notebook: a fidelity check in `tests/test_integration_model.py` that reconstructs a fully visible clean sinusoid and an interior-masked one and asserts a correlation/MAE bound (and compares against the upstream reference usage for MOMENT-1-base, including the large checkpoint as a control). If a loading or call-convention defect is found, fix it in `src/moment_pipeline/` and regenerate. If the checkpoint genuinely reconstructs this poorly, the notebook must say so: print the model-vs-baseline comparison as a verdict (e.g. "model worse than the flat-hold baseline by 6.6× on this sample"), add a visible-point fidelity line, rewrite the interpretation and limits so the learner reads the loss correctly, and downgrade the registry row and model card from "imputation capability" to a documented negative result. Either way, stop presenting the notebook as Release-grade until this is resolved.
- **Acceptance check:** (a) a repository test reconstructs a fully visible 512-step sine through `impute`/`reconstruct` and asserts Pearson ≥ 0.9 between reconstruction and input, and passes; **or** (b) the regenerated notebook prints, on the default path, a model-vs-baseline verdict line and a visible-point fidelity figure, its interpretation section states in words that the model did not beat the baseline on this sample, and `tutorials/README.md` / `MODEL_CARD.md` no longer describe the capability as working imputation. In both cases, the hosted rerun's printed comparison and the prose agree.
- **Spec:** RUN8, UX1, UX4, EVAL3, EVAL15, GDL14.

### Major

#### MIM-M1 — `Run all` needs a manual restart after the install cell; recorded as PASSED / Release-grade

- **Cell/section:** Section 1 (cell 3; `tools/build_notebook.py` install-cell builder); `docs/release-verification.md:133`; `tutorials/README.md:24`.
- **Observed issue:** the pinned install replaces packages the hosted kernel already imported; the guard raises `RuntimeError: … numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` (archived `executed-pass1.ipynb`, same blob). The run passed only on pass 2. The record says "PASSED — 18/18 code cells ok (1 restart after install cell)" and the registry says Release-grade, although the opening promises Run all needs no intervention. On the CPU image the pins also pulled `torch 2.14.0+cu130` and NVIDIA wheels.
- **Consequence:** a first-time learner's `Run all` stops in cell 3; a restart-dependent run is presented as release evidence.
- **Evidence:** documented execution (`run_summary.json`, `restarted_after_install_cell: true`; pass 1 error text); source inspection (cell 3).
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern: a carrier cell bootstraps uv, runs `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:` (CPU torch wheels for the CPU path), and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`. Generate it from `tools/build_notebook.py` so the four MOMENT notebooks change together. Until a one-pass hosted run exists, record the 2026-09-14 run as not Run-all conformant and set the registry status to Candidate.
- **Acceptance check:** on a fresh Colab (or Kaggle) CPU runtime, one `Run all` of the new blob completes every code cell with no error output and no restart, and `docs/release-verification.md` records that run (blob, runtime, outcome) with no restart note.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### MIM-M2 — The default holdout is a trailing extrapolation, and the "linear interpolation" baseline is a flat hold

- **Cell/section:** Section 5 (cell 29, `start = complete_patch_starts[-1]`; `tools/notebook_template_imputation.py:190`); Section 7 (cell 33, `interpolate(method="linear", limit_direction="both")`; `tools/notebook_template_imputation.py:235`); markdown cells 28 and 32.
- **Observed issue:** the cell picks the *latest* complete patch, which on the sample (and on any gap-free series) is the series' final 8 steps (504–511 of 256 valid positions). No observed point follows it, so the model is extrapolating the last 2 hours — the forecasting setting the notebook says it does not demonstrate — and `interpolate(..., limit_direction="both")` fills trailing values with the last observed value: the "linear-interpolation baseline" predicts 26.131526 (temperature) and 1.280415 (vibration) at every withheld step. The prose calls it "a linear interpolation baseline … on the same artificially withheld positions".
- **Consequence:** the learner is told the experiment is gap imputation against linear interpolation when it is tail extrapolation against last-value carry-forward. Conclusions about either method do not transfer to genuine interior gaps (for this sample the baseline's interior MAE is 0.062, not 0.171).
- **Evidence:** direct execution (P2 `baseline_shape`: one unique prediction per channel equal to the last observed value, 0 observations after the holdout; P3 per-position table); source inspection (cells 29, 33).
- **Recommended correction:** in `tools/notebook_template_imputation.py`, choose an interior complete patch with observed neighbours on both sides (e.g. the middle complete patch, or a named `HOLDOUT_START` form field defaulting to an interior patch), and name the baseline by what it computes. If a trailing holdout is kept deliberately, label it "trailing-patch extrapolation" and the baseline "last-value hold".
- **Acceptance check:** on the default sample the printed holdout has at least one source-observed timestamp after it in every channel, the baseline's predictions are not all equal within a channel, and the markdown names the holdout placement and the baseline method accurately.
- **Spec:** UX1, UX2, EVAL5, EVAL10 (baseline correctly described); §21.4 (clarity of the artificial mask).

### Minor

#### MIM-m1 — MAE/RMSE pool two channels with different units and scales

- **Cell/section:** Section 7 markdown (cell 32) and cell 33.
- **Observed issue:** the markdown says MAE is "in the original units", but the single MAE averages temperature (mean 28.3, std 1.78) and vibration (mean 1.60, std 0.26). Temperature contributes 2.05 of the pooled 1.12; vibration contributes 0.20.
- **Consequence:** the headline number is unit-less in practice and hides that the model fails much worse on one channel.
- **Evidence:** direct execution (P2 `per_channel`); source inspection.
- **Recommended correction:** print per-channel MAE/RMSE (model and baseline) beside the pooled values, or a scale-free measure (e.g. MAE divided by each channel's standard deviation), and say which the verdict uses.
- **Acceptance check:** the default path prints model and baseline MAE per channel, and the markdown no longer calls the pooled figure "original units" without qualification.
- **Spec:** EVAL3, EVAL15.

#### MIM-m2 — GUIDED layer is partial; no control over the experiment

- **Cell/section:** opening, Sections 2, 5–7, final cell.
- **Observed issue:** no explicit intended learner or How-to-use section, roadmap, Input → Model → Output line, prediction prompt, interpretation checkpoints with sample answers, troubleshooting list (install restart, Hub download, upload, validation codes, the no-complete-patch error), or conclusion template. The holdout position is hard-coded, so the learner cannot change the one variable the lesson is about. Section 2's ten carried-module cells (~125 k characters) are not labelled as infrastructure.
- **Consequence:** self-paced learners get a reference walkthrough, not a guided lesson, and cannot test the comparison themselves.
- **Evidence:** source inspection; direct execution (P3 shows a 32-position sweep runs in under 10 s on CPU).
- **Recommended correction:** add these elements in `tools/notebook_template_imputation.py` and the shared opening in `tools/build_notebook.py`; expose `HOLDOUT_START` (MIM-M2) and add one Predict → Change → Run → Explain activity around it.
- **Acceptance check:** the regenerated notebook has an audience/How-to-use block, a roadmap, an Input → Model → Output line, at least one Predict → Run → Explain prompt with a collapsible sample answer, a troubleshooting list, an Infrastructure label on Section 2, a holdout-position form field, and a conclusion template.
- **Spec:** GDL1–GDL5, GDL7, GDL9–GDL14, UX5.

#### MIM-m3 — Section 3 prints `device: None` and a hard-coded `source`

- **Cell/section:** cell 25, last line (`tools/build_notebook.py:548`).
- **Observed issue:** `LoadedMoment` has no `device` or `source` attribute, so the line prints `{'device': None, 'source': 'local-snapshot'}`. The markdown promises the effective device and weight source are printed before inference; the real device appears only in Section 5.
- **Consequence:** the learner sees `None` where the device is said to be confirmed.
- **Evidence:** direct execution (P2) and documented execution (hosted output identical).
- **Recommended correction:** print `pipe.identity.device`, `pipe.identity.dtype`, `pipe.identity.weight_file_loaded` and the snapshot directory from the generator's model-load cell.
- **Acceptance check:** Section 3 output shows a concrete device and the verified weight file and contains no `None` or fallback literal.
- **Spec:** MOD3, ENV3 (display).

#### MIM-m4 — Prerequisites misdescribe external access and download size

- **Cell/section:** Prerequisites (cell 1; `tools/build_notebook.py:441`).
- **Observed issue:** "No GitHub access … required", yet `PINS` installs `momentfm @ git+https://github.com/…@38f7310…`. The size note omits that on the Kaggle CPU image the install pulled the CUDA build of torch and NVIDIA wheels.
- **Consequence:** learners on restricted networks, or estimating time and disk, are misinformed.
- **Evidence:** source inspection; documented execution (pass 1 download log).
- **Recommended correction:** name GitHub (upstream `momentfm`, pinned commit) as an install-time source and give a measured download figure with its environment (or pin CPU wheels; see MIM-M1).
- **Acceptance check:** the Prerequisites name GitHub as an install-time source and give a measured download size with its environment.
- **Spec:** UX12, SRC3.

#### MIM-m5 — The diagnostic plot can miss the held-out patch

- **Cell/section:** Section 8 (cell 35).
- **Observed issue:** the plot shows the last 96 rows of the first channel of the concatenated frame, with no marker for the held-out positions and an index x-axis. On multi-series BYOD the last 96 `flow` rows all belonged to series B while the holdout was in series A (P4), so the plot showed no withheld point. Only one of the channels is plotted, and on the sample it is the channel where the model is worst, without saying so.
- **Consequence:** the learner cannot visually check the imputation the metrics describe.
- **Evidence:** direct execution (P4 `svg_series_mix = ['B']`); source inspection.
- **Recommended correction:** plot the held-out window's series and channel(s) around `start:stop`, shade the held-out span, and add the baseline as a third line.
- **Acceptance check:** on the default sample and on a two-series BYOD input, the plot contains the held-out timestamps of the scored window, visibly marked, with model, baseline and truth.
- **Spec:** UX3, UX11.

#### MIM-m6 — Conformance and BYOD-interface hygiene

- **Cell/section:** metadata and opening; cells 3, 27, 29.
- **Observed issue:** declares spec 2.0 (current 2.2). `DIMER_NOTEBOOK_CI_PREINSTALLED` is read but never documented. The non-interactive BYOD location is only an environment variable, not a form field. On Jupyter, `USE_BYOD=True` fails with `ModuleNotFoundError: No module named 'google'`, although Jupyter is a stated runtime. On multi-series BYOD only window 0 is held out and scored, which the prose does not say.
- **Consequence:** executors and Jupyter users have no documented field-based way to supply data; BYOD users may think every series was evaluated.
- **Evidence:** direct execution (P4); source inspection.
- **Recommended correction:** add `BYOD_PATH = ''  # @param {type:"string"}` (falling back to the environment variable), document both environment variables, say the upload widget is Colab-only, state that only the first window is evaluated (or evaluate one patch per window), and regenerate against spec 2.2.
- **Acceptance check:** cell 27 reads a form-field path without importing `google.colab`; the opening documents both environment variables; the evaluation section states which windows are scored; metadata declares the current spec.
- **Spec:** EXE1, EXE2, EXE5.

### Suggestions

- **MIM-S1:** rotate the holdout over every interior complete patch (32 positions take under 10 s on CPU) and report the median and spread for model and baseline, which turns a one-point anecdote into a small, honest comparison.
- **MIM-S2:** archive the pass-1 error text in `docs/release-verification.md` beside the outcome, so a restart-dependent pass cannot be read as one-pass.
- **MIM-S3:** in the "next experiment", explain cross-channel patch coupling: when one channel is missing, MOMENT hides that whole patch for every channel (P4: 20 observed `pressure` points hidden because `flow` was missing), which is why `model_masked_point_fraction` exceeds the source fraction.

## 6. Readiness

**Needs revision.** Open Blocker: MIM-B1 (the central imputation result is not meaningful and is presented as if it were). Open Majors: MIM-M1 (no one-pass Run all; RUN1, RUN10, ENV6, REL2, REL11 fail on the recorded run) and MIM-M2 (holdout and baseline misdescribed). Remaining gates after fixes: a root-cause determination or an explicit negative-result rewrite for MIM-B1; a one-pass hosted `Run all` of the regenerated blob recorded in `docs/release-verification.md`; a Colab upload-widget check of BYOD (not verified here).

Out of scope but worth checking by the owner: the anomaly-scoring notebook ranks reconstruction residuals from the same reconstruction path, so MIM-B1's observation (visible-point Pearson ≈ 0) bears on it too; this review did not test that notebook.

## 7. What was verified vs inferred

- **Verified by direct execution (local Windows CPU, not Colab):** default path 18/18 with install skipped and snapshot pre-staged; metrics identical to the hosted run; the 32-position holdout sweep; the 512-step no-padding control; upstream `reconstruct()` on fully visible sines under transformers 5.16.1 and 4.33.3 with identical results; the flat-hold baseline; per-channel errors; BYOD with source gaps end to end via `DIMER_BYOD_PATH`; four coded or actionable rejections; cancelled-upload and Jupyter `USE_BYOD` failures; `build_notebook.py --check` and `validate_release_assets.py` exit 0.
- **Verified from documented execution:** the hosted pass-1 restart failure and pass-2 success for this blob; the hosted metrics and exported series (visible-point Pearson ≈ 0).
- **Inferred / not verified:** *why* the pinned checkpoint's reconstruction does not track its input (not determined; the large checkpoint and the upstream reference notebook were not run); behaviour on a current Colab image; the Colab upload widget; learner understanding (no learner observation).
- **Most likely to be wrong:** MIM-B1's framing as a pipeline-wide fidelity failure rather than a property of `MOMENT-1-base` itself. If the base checkpoint genuinely reconstructs this poorly under upstream's reference usage, the defect is entirely in what the notebook tells the learner (path (b) of the acceptance check), and Kurt may judge it Major rather than Blocker.
