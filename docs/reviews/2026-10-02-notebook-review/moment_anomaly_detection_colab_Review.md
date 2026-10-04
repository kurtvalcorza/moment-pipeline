# MOMENT Anomaly-Scoring Tutorial Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/moment-pipeline`  
**Notebook:** `tutorials/moment_anomaly_detection_colab.ipynb`  
**Reviewed commit:** `44526760c75d43a4f314252fef297562c9651a32` (`main`, confirmed with `gh api repos/kurtvalcorza/moment-pipeline/commits/main`)  
**Notebook Git blob:** `3c11d4ae372196ab6b7a149410bbc24519bb54c0`. This is the blob executed in the recorded Kaggle CPU run of 2026-09-14 (commit `31ddb06`); the notebook has not changed since.  
**Finding prefix:** `MAD`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The default path is well engineered and reproducible. The notebook digest-verifies the pinned 3-file snapshot, regenerates the labelled synthetic sample and asserts its checked-in digests, validates the input into a manifest (with a recorded `EMPTY_CHANNEL` rejection), scores raw MAE residuals with no threshold, ranks the `vibration` channel, and exports scores, provenance, an evaluation report and a result file. This review's local CPU run reproduced the hosted numbers exactly: injected spikes at ranks 1, 2, 3 (scores 4.005 / 2.879 / 2.633, next 0.814), top-3 recall 1.0, verdict `sample-sanity`. BYOD through `DIMER_BYOD_PATH` reaches every stage, and six invalid inputs are rejected with coded messages before the model runs.

Two things need fixing before the notebook teaches what it says:

1. **No one-pass `Run all` (MAD-M1).** The recorded Kaggle run failed in the install cell (`numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.`) and passed only on a second pass after a restart. The release record calls this PASSED / Release-grade with "1 restart after install cell".
2. **The score reading the notebook teaches does not hold on its own sample (MAD-M2).** The notebook says "higher residual = stronger anomaly evidence" without qualification. On the default sample, the **clean** `temperature` channel (no injected anomaly) has a mean residual of 1.62, against 0.30 for `vibration`, and 42 clean temperature points score above the lowest injected spike. Across both channels the spikes rank 1, 21 and 45. The 1.0 recall holds because Section 7 hard-codes `vibration`, the channel the labels say contains the spikes. The package's own `channel_aggregation="max"` gives recall 0.33 and `"mean"` gives 0.67. The notebook never shows or explains this, and on BYOD it ranks and plots only the alphabetically first channel.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `TASK-INFERENCE` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (`metadata.dimer.notebook_spec`, opening cell) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated as such. Prerequisites: "basic Python and pandas; what a long-format time-series table is" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default; CUDA used when present; float32 only |
| Promised outcomes | pinned install; carried package (10 modules); staged and digest-verified snapshot; deterministic labelled synthetic sample (or BYOD CSV); validation into an input manifest; raw MAE residual scores with no threshold; score-direction and threshold semantics; `top_k_recall` ranking check (`sample-sanity` with labels, `not-measurable` without); diagnostic SVG; scores CSV, provenance, evaluation report and result JSON |
| Learning objectives | install; read package guarantees; verify the model revision; generate the sample or bring a CSV; validate/canonicalize; compute raw scores; "interpret score direction and threshold semantics"; check the spike ranking quantitatively; export with provenance |

### Existing execution evidence

- `docs/release-verification.md`: Kaggle CPU (image torch 2.10.0+cpu), commit `31ddb06` / blob `3c11d4ae` (= the reviewed blob), 2026-09-14, **PASSED — 18/18 code cells ok (1 restart after install cell)**, 264.0 s, 454 MB staged from the Hub. `tutorials/README.md` lists the notebook as **Release-grade**.
- The archived run (`.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-moment-anomaly-detection/v1/evidence/`) holds `executed-pass1.ipynb` and `executed.ipynb`. Pass 1 stops in cell 3 with `RuntimeError: Core dependencies changed while older modules were loaded: numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Pass 2 completes. The pinned install pulled `torch 2.14.0+cu130` plus the CUDA wheels (several downloads of 145–555 MB each) onto the CPU image.
- No hosted run covers BYOD, the suggested `loss="mse"` experiment, or a Colab runtime.

### Evidence obtained by this review

- **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`, 4 torch threads), venv `dimer-moment` (Python 3.12.10, torch 2.14.0+cpu, numpy 2.5.3, pandas 3.0.5, transformers 5.16.1, huggingface-hub 1.30.0, safetensors 0.8.0, momentfm 0.1.5 = every `PINS` entry). The install cell ran with the notebook's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`. The 3 snapshot files were pre-staged by copy (fetched = `[]`), then verified by the notebook. `google.colab.files.upload` was a shim. **This is not a Colab run.**
- **P1 static:** JSON parses (nbformat 4.5); 18 code cells compile; no persisted outputs; blob equals the recorded-run blob. `tools/build_notebook.py --check` exit 0; `tools/validate_release_assets.py` exit 0 (PASS).
- **P2 default path (direct, CPU):** 18/18 cells ok in 23 s (model load 12.5 s, scoring 0.4 s). Recall 1.0, injected ranks 1/2/3, 6 output files, values identical to the hosted outputs. Section 3 printed `{'device': None, 'source': 'local-snapshot'}`. Per-channel scored residuals: temperature mean 1.620 / max 3.766, vibration mean 0.304 / max 4.005. Pooled across channels, injected ranks are 1, 21, 45, and 42 clean temperature points outscore the lowest spike. The same figures come from the hosted `moment_anomaly_scores.csv`.
- **P3 active learning:** the suggested `loss="mse"` with Sections 6–7 rerun gives recall 1.0, ranks 1/2/3, and a top-20 order identical to MAE. `channel_aggregation="mean"` gives recall 0.667 (ranks 1, 3, 8); `"max"` gives 0.333 (ranks 1, 21, 45).
- **P4 naive baselines on the sample (no model):** |z-score| and |deviation from a centred 5-point rolling median| on `vibration` each get top-3 recall 1.0; |first difference| gets 0.667.
- **P5 BYOD:** a two-series, two-channel CSV (`flow`, `pressure`; series A 600 h → truncated; series B 120 h → padded, with an 8-step gap) runs Sections 4–9 through `DIMER_BYOD_PATH`. Verdict `not-measurable`; scored fraction 0.975; an injected +6 `flow` excursion ranks first. Only `flow` is ranked and plotted, and the SVG joins A and B into one 616-point line. A single-channel CSV completes, and the empty-channel probe records `EMPTY_CHANNEL`. Invalid inputs: duplicate header, missing `value`, Latin-1 bytes (cell 27), unparseable timestamp, non-numeric value, duplicate key (cell 29). Each raises a coded `ValidationError` before scoring. A cancelled or two-file upload raises `ValueError: Upload exactly one CSV with columns series_id,timestamp,channel,value.`

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | The default path is correct and deterministic (local and hosted outputs agree to the printed digits). Supply-chain verification, validation and export are sound. The install step forces a restart on hosted images (MAD-M1). Section 3 prints placeholder device and source values (MAD-m4). |
| Promise fulfilment | Every promised stage runs and produces its output. The objective "interpret score direction" is taught as a universal rule that the default sample contradicts across channels, and the passing check depends on a hard-coded channel choice (MAD-M2). The suggested experiment cannot change the ranking (MAD-m2). |
| Learner experience | Clear section prose with "Successful output means…" notes and a strong limits section. The GUIDED layer is thin: no roadmap, Input → Model → Output contract, predictions, checkpoints, troubleshooting or conclusion prompts. About 125 k characters of carried code are not labelled as skippable infrastructure (MAD-m1). The plot is hard to read (MAD-m6). |
| Spec conformance | RUN1/RUN10/ENV6/REL2/REL11 fail on the recorded hosted run (MAD-M1). EVAL15 and UX4 are not met for the score interpretation (MAD-M2). SHOULD gaps: GDL1–GDL14 (partial), UX5, EVAL10, EXE2/EXE5, SRC10/UX12. Declared spec is 2.0, not 2.2. |

## 3. Promise and objective tracing

| Claim (cell) | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes without intervention (0, 1) | cell 3 pip install + stale-import guard | hosted pass 1 stops with a restart instruction | learner must restart and rerun | **fails on hosted evidence** (MAD-M1) |
| No GitHub access required (1) | `PINS` include `momentfm @ git+https://github.com/…` | pip clones from GitHub | prose inaccurate | MAD-m5 |
| Effective identity, device and weight source printed before inference (24) | cell 25 `getattr(pipe, 'device', None)` | `{'device': None, 'source': 'local-snapshot'}` | device looks unknown | MAD-m4 |
| Deterministic labelled sample, digest-asserted (26) | cell 27 | digests match; 512 rows | — | verified |
| Validation and input manifest before the model, with a rejection shown (28) | cell 29 | manifest, `EMPTY_CHANNEL` finding, padding note | — | verified |
| Higher score = stronger anomaly evidence (0, 30) | cell 31 MAE residual | clean temperature residuals ≈ 5× vibration; 42 clean points above a spike | taught as universal | **misleading** (MAD-M2) |
| Falsifiable ranking check on the labelled sample (32) | cell 33 `top_k_recall` on hard-coded `vibration` | recall 1.0 | depends on the channel choice; a z-score also gets 1.0 | partially (MAD-M2, MAD-m3) |
| No threshold, no binary labels (0, 30, 38) | `threshold_policy` | none applied | — | verified |
| Diagnostic plot helps inspection (34) | cell 35 SVG | one unlabeled line, scored positions only | hard to relate to time or spikes | MAD-m6 |
| Exports with provenance (36) | cell 37 | 6 files; `sample.kind` correct for BYOD | — | verified |
| BYOD through the same stages (0, 4) | cell 27 branches | completes; ranks first channel only | other channels unseen in Sections 7–8 | partially (MAD-M2) |

| Objective | Learner activity | Evidence it is exercised |
|---|---|---|
| Interpret score direction and threshold semantics | read prose; read top-12 table | no prompt asks the learner to compare channels, predict, or explain; the sample contradicts the rule taught (MAD-M2, MAD-m1) |
| Check the spike ranking quantitatively | read printed recall | runs; no baseline for reference (MAD-m3) |
| Validate and canonicalize | read manifest and the rejection finding | exercised by the printed manifest |
| Bring your own CSV | optional `USE_BYOD` / `DIMER_BYOD_PATH` | works (P5) |
| Install, verify, export | run cells | exercised; install needs a restart on hosted runtimes (MAD-M1) |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Good per-section prose and limits. Contradicted by the sample's own clean-channel scores (MAD-M2); guided-layer gaps (MAD-m1); Section 3 prints `None` for device (MAD-m4). |
| Clean default | Documented execution (Kaggle CPU, same blob) + direct execution (local CPU) | Hosted: PASSED only after a restart (MAD-M1). Local: 18/18 in 23 s with install skipped and snapshot pre-staged; outputs identical to hosted. No Colab run. |
| Active learning | Direct execution (local CPU) | `loss="mse"` rerun gives identical ranks and order (MAD-m2). The aggregation variants show the channel dependence (MAD-M2). |
| Reuse and recovery | Direct execution (local CPU, `DIMER_BYOD_PATH` and an upload shim) | Two-series BYOD completes (`not-measurable`); 6 invalid inputs rejected with codes; a cancelled upload gives a clear message. Only one channel ranked and plotted. No hosted BYOD run. |

## 5. Findings

### Major

#### MAD-M1 — `Run all` needs a manual restart after the install cell; recorded as PASSED / Release-grade

- **Cell/section:** Section 1 (cell 3); `docs/release-verification.md` row of 2026-09-14; `tutorials/README.md` registry row.
- **Observed issue:** the pinned install replaces packages the hosted kernel has already imported. The guard then raises `RuntimeError: … numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` (archived `executed-pass1.ipynb`). The run passed only on a second pass. The record nevertheless says "PASSED — 18/18 code cells ok (1 restart after install cell)", and the registry calls the notebook Release-grade. The opening promises that Run all completes with no intervention. On the CPU image the pins also pull the CUDA build of torch 2.14.0 and its NVIDIA wheels.
- **Consequence:** a first-time learner's `Run all` stops in cell 3. A restart-dependent run is presented as release evidence.
- **Evidence:** documented execution (archived pass 1 / pass 2 notebooks, same blob `3c11d4ae`); source inspection (cell 3).
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern. A carrier cell bootstraps uv, runs `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:` (CPU torch wheels for the CPU path), and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`. Generate it from `tools/build_notebook.py` (the install cell builder), not by hand. Until a one-pass hosted run exists, record the 2026-09-14 run as not Run-all conformant and set the registry status to Candidate.
- **Acceptance check:** on a fresh Colab (or Kaggle) CPU runtime, a single `Run all` of the new blob completes every code cell with no error output and no restart, and `docs/release-verification.md` records that run (blob, runtime, outcome) with no restart note.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### MAD-M2 — "Higher residual = more anomaly evidence" is taught as universal, but on the default sample the clean channel outscores the spikes; the passing check relies on a hard-coded channel

- **Cell/section:** opening cell 0, Section 6 markdown (cell 30) and cell 31, Section 7 cell 33 (`score_channel = "vibration" if … else str(scores["channel"].iloc[0])`, `tools/notebook_template_anomaly_detection.py:218`), Section 8 cell 35.
- **Observed issue:** the clean `temperature` channel of the default sample, which contains no anomaly, gets scored residuals of mean 1.620 and max 3.766. The `vibration` channel gets mean 0.304, with the injected spikes at 4.005 / 2.879 / 2.633. 42 clean temperature points score above the lowest spike. Ranked across both channels, the spikes are 1st, 21st and 45th. The first table the learner sees (`scores.head()`, cell 31) shows clean temperature residuals of 1.06–2.07, with no comment. Section 7 then ranks `vibration` only, because the code names it. That is the channel the label table says contains the spikes, which an unlabelled user would not know. With the package's own aggregations, recall falls to 0.667 (`mean`) and 0.333 (`max`). The `score_policy` string says the residual scale depends on the series. Nothing says that it also depends on the channel, or that scores are comparable only within one channel. On BYOD, Sections 7–8 rank and plot only the alphabetically first channel (`flow` in P5), and nothing says the other channels were left out.
- **Consequence:** the learner is likely to take away that a high residual anywhere means an anomaly, and to read recall 1.0 as evidence of detection. Yet the same run shows a clean channel scoring like an anomaly. A BYOD user with several channels never sees most of their scores outside the CSV.
- **Evidence:** direct execution (P2, P3: local CPU) and documented execution (hosted `moment_anomaly_scores.csv`, same values); source inspection.
- **Recommended correction:** in the generator template, print a per-channel summary of the scored residuals (count, median, max) after cell 31, and state in Sections 6–7 that raw residuals are comparable only within one channel of one series. Explain that the check ranks `vibration` because that is where the spikes were injected, and show the pooled or aggregated ranking next to it as the contrast. Replace the hard-coded channel with a form field (`SCORE_CHANNEL = ''  # @param`, default the labelled channel). For BYOD, rank and plot every channel, or print which channels were omitted. Optionally offer a per-channel robust normalisation as a clearly labelled post-processing step.
- **Acceptance check:** the default run prints per-channel residual statistics, and the prose names temperature's higher scale and explains why. Section 7 states the channel choice, and how it was made, before the recall is printed. A pooled or aggregated ranking is shown, or its effect is stated. A two-channel BYOD CSV shows the ranking for every channel, or names the channels not shown.
- **Spec:** EVAL15, UX4, UX2, §21.5 (score direction), UNC2 (context).

### Minor

#### MAD-m1 — GUIDED layer is partial

- **Cell/section:** opening, Sections 2, 6–7, final cell.
- **Observed issue:** no explicit intended learner or How-to-use section, roadmap, or Input → Model → Output contract. No question or prediction before the scoring and ranking, no interpretation checkpoints with worked answers, no troubleshooting section (install restart, Hub download, upload, validation codes), no conclusion template. Section 2's ten carried-module cells (~125 k characters) are not labelled as Infrastructure that the learner may run without studying. Objectives use "read"/"interpret" without an observable activity.
- **Consequence:** self-paced learners get a reference walkthrough rather than a guided lesson.
- **Evidence:** source inspection.
- **Recommended correction:** add these elements in `tools/notebook_template_anomaly_detection.py` and the shared opening in `tools/build_notebook.py`. Include a prediction prompt before Section 7 (for example: "Which channel will have the larger typical residual, and why?").
- **Acceptance check:** the regenerated notebook has an audience/How-to-use block, a roadmap, an Input → Model → Output line, at least one Predict → Run → Explain prompt with a collapsible sample answer, a troubleshooting list covering the install, download, upload and validation failures, an Infrastructure label on Section 2, and a conclusion template.
- **Spec:** GDL1–GDL5, GDL7, GDL9, GDL11, GDL13, GDL14.

#### MAD-m2 — The suggested experiment (`loss="mse"`) cannot change the ranking

- **Cell/section:** "Next experiments" (cell 38, `tools/notebook_template_anomaly_detection.py:320`).
- **Observed issue:** with `channel_aggregation="none"`, MSE is the square of the per-element absolute error (`anomaly.py:192`), a monotone transform, so the ranking is identical by construction. The rerun gives recall 1.0, ranks 1/2/3 and an identical top-20 order. The notebook also gives no rerun instruction (which cells, where to edit).
- **Consequence:** the only suggested experiment teaches nothing about the ranking, and a learner may conclude the loss never matters, which is not true once channels are aggregated.
- **Evidence:** direct execution (P3); source inspection.
- **Recommended correction:** replace it with an experiment that can change the outcome: `channel_aggregation="mean"` vs `"max"` (recall 0.667 / 0.333 here), or a channel change. Give a prediction prompt and the exact cells to rerun (Sections 6–8). If MSE stays, state that it only rescales scores when the channels are not aggregated.
- **Acceptance check:** the suggested experiment, rerun as instructed, produces a different ranking or metric on the default sample, or the prose states beforehand why it will not.
- **Spec:** GDL10, UX5.

#### MAD-m3 — No naive baseline; a z-score also gets recall 1.0 on the sample

- **Cell/section:** Section 7 (cell 33); `evaluation_report(..., baseline=None)`; report `"baselines": []`.
- **Observed issue:** |z-score| and |deviation from a rolling median| each get top-3 recall 1.0 on `vibration`; |first difference| gets 0.667. `evaluation_report` accepts a `baseline` argument that the notebook does not use.
- **Consequence:** recall 1.0 has no reference point. The learner cannot tell that this sample does not separate MOMENT from a one-line statistic.
- **Evidence:** direct execution (P4).
- **Recommended correction:** compute a rolling-median or z-score baseline on the same channel, pass it to `evaluation_report(baseline=…)`, and say what the comparison does and does not show.
- **Acceptance check:** the evaluation report lists at least one naive baseline with its recall on the same channel and k, and the prose interprets the comparison.
- **Spec:** EVAL10 (SHOULD when meaningful).

#### MAD-m4 — Section 3 prints `device: None` and a hard-coded `source`

- **Cell/section:** cell 25, last line (`tools/build_notebook.py:548`).
- **Observed issue:** `LoadedMoment` has no `device` or `source` attribute, so `getattr(pipe, 'device', None)` prints `None` and `getattr(pipe, 'source', 'local-snapshot')` prints the fallback literal. The markdown promises that the effective identity, device and weight source are printed before inference. The real device (`pipe.identity.device` = `cpu`) appears only in Section 5.
- **Consequence:** the learner sees `device: None` at the point where the notebook says the device is confirmed.
- **Evidence:** direct execution (P2) and documented execution (hosted output identical).
- **Recommended correction:** print `pipe.identity.device`, `pipe.identity.dtype`, `pipe.identity.weight_file_loaded` and the snapshot directory from the generator's model-load cell.
- **Acceptance check:** Section 3 output shows a concrete device (`cpu`/`cuda`) and the verified weight file, and contains no `None` or fallback literal.
- **Spec:** MOD3, ENV3 (display, partial).

#### MAD-m5 — Prerequisites misdescribe external access and download size

- **Cell/section:** Prerequisites (cell 1; `tools/build_notebook.py:441` and the template's runtime bullet).
- **Observed issue:** "No GitHub access … required", yet `PINS` installs `momentfm @ git+https://github.com/…@38f7310…`, which pip clones from GitHub. The size note ("`torch==2.14.0` … largest download") omits that on the Kaggle CPU image the install pulled the CUDA build and NVIDIA wheels, several GB in total.
- **Consequence:** learners on restricted networks, or estimating time and disk, are misinformed.
- **Evidence:** source inspection; documented execution (pass 1 download log).
- **Recommended correction:** state that the install fetches `momentfm` from GitHub at a pinned commit and give the measured download size for the supported runtime (or pin CPU wheels for the CPU path; see MAD-M1).
- **Acceptance check:** the Prerequisites name GitHub (upstream `momentfm`, pinned commit) as an install-time source and give a measured download figure with its environment.
- **Spec:** UX12, SRC3 (stale instruction).

#### MAD-m6 — Diagnostic plot is hard to read and merges series

- **Cell/section:** Section 8 (cell 35; `tools/notebook_template_anomaly_detection.py:248-273`).
- **Observed issue:** one polyline of the scored positions of one channel, spaced by index rather than time. No axis ticks or labels, no raw signal, no markers for the injected spikes. With BYOD, series A and B are joined into one 616-point line with no boundary.
- **Consequence:** the learner cannot relate high scores to time, the signal or the labels, and multi-series BYOD plots are misleading.
- **Evidence:** direct execution (P5); source inspection.
- **Recommended correction:** plot per series (and per channel) against time, overlay the raw value or mark labelled spikes, and add a y-axis scale.
- **Acceptance check:** on the default sample, the plot marks the three injected timestamps and has a labelled score axis. On the two-series BYOD CSV, the series are drawn separately.
- **Spec:** UX11.

#### MAD-m7 — Conformance hygiene

- **Cell/section:** metadata and opening; cells 3 and 27.
- **Observed issue:** the notebook declares spec 2.0 (current 2.2). `DIMER_NOTEBOOK_CI_PREINSTALLED` is read but never documented. The non-interactive BYOD location is only an environment variable (`DIMER_BYOD_PATH`), not a form field. On Jupyter, `USE_BYOD=True` fails with `ModuleNotFoundError: google.colab`, although Jupyter is a stated runtime.
- **Consequence:** executors and Jupyter users have no documented, field-based way to supply data.
- **Evidence:** source inspection.
- **Recommended correction:** add `BYOD_PATH = ''  # @param {type:"string"}` (falling back to the environment variable), document both environment variables, say that the upload widget is Colab-only, and regenerate against spec 2.2.
- **Acceptance check:** cell 27 reads a form-field path without importing `google.colab`; the opening documents both environment variables; metadata declares the current spec.
- **Spec:** EXE1, EXE2, EXE5.

### Suggestions

- **MAD-S1:** add a "difficult normal event" (for example a slow drift or level shift) to the sample, so the limits section's claim that a faithfully reconstructed drift scores low is demonstrated rather than only stated.
- **MAD-S2:** archive the pass-1 error text in `docs/release-verification.md` beside the outcome, so a restart-dependent pass cannot be read as one-pass.
- **MAD-S3:** give a one-line reason why the temperature channel reconstructs worse (scale, period, or left-padding), backed by a short check, so the per-channel difference becomes a teaching point.

## 6. Readiness

**Needs revision.** Two Majors are open. MAD-M1 is a RUN1/RUN10/ENV6 MUST failure on the recorded hosted run. MAD-M2 means the central interpretation the notebook teaches is contradicted by its own default output. Remaining gates: a one-pass hosted `Run all` of the regenerated blob recorded in `docs/release-verification.md`; a hosted (or recorded) BYOD run for REL12; and the Minor and SHOULD items as practical.

## 7. What was verified vs inferred

- **Verified by direct execution (local CPU, not Colab):** the default path (18/18 cells); per-channel residual statistics and pooled ranks; the MSE and aggregation reruns; the naive baselines; BYOD completion and nine rejection cases (six invalid files, one single-channel probe, two upload-shim cases).
- **Verified from documented execution:** the hosted restart (pass 1 error text), the hosted default outputs (identical to local), and the CUDA-wheel install on the CPU image.
- **Inferred:** that Colab's preloaded NumPy triggers the same restart (Kaggle evidence only); learner impact (no learner observation).
- **Most likely to be wrong:** MAD-M2's severity. The notebook does label recall as `sample-sanity`, the policy string says scale "depends on the series", and the spikes do rank 1–3 within their channel. A reader who weighs those caveats could call this a Minor explanation gap rather than a misleading lesson.
