# MOMENT Classification-Adaptation Tutorial Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/moment-pipeline`  
**Notebook:** `tutorials/moment_classification_colab.ipynb`  
**Reviewed commit:** `44526760c75d43a4f314252fef297562c9651a32` (`main`, confirmed with `gh api repos/kurtvalcorza/moment-pipeline/commits/main`)  
**Notebook Git blob:** `7c5c3732c112f15d2a06baaa4cec26e781ee035d`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-19 (commit `ac2f9ef`); the notebook has not changed since (last touched by `247db73`, PR #13).  
**Finding prefix:** `MCL`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The default path is carefully engineered. It installs exact pins, carries the package as 12 digest-checked module cells, digest-verifies the pinned 3-file snapshot and the 79.6 MB UCI HAPT archive, cuts 179 windows by an a-priori rule, splits by whole volunteer with a disjointness check, shows four dataset refusals and a long-frame refusal, scores a majority floor, a cosine 5-NN vote and a linear probe, trains a bounded unfreeze selected on validation log-loss, evaluates on an untouched test split with per-class recall, and exports a safetensors adapter that reloads on a fresh base with an asserted parity check. The generator check and the static validator both pass on the reviewed commit. The recorded hosted run of this exact blob reached every stage.

Three things need fixing before the notebook teaches what it says:

1. **No one-pass `Run all` (MCL-M1).** The recorded Kaggle T4 run stopped in the install cell (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.`) and passed only on a second pass. The release record calls it PASSED / Release-grade with "1 restart after install cell".
2. **The suggested experiments and the BYOD rerun break the comparison and can fail the reload assertion (MCL-M2).** `adapt` changes the encoder inside `pipe` in place when the unfreeze wins, which it does on the default path. Rerunning Sections 6–9 (the optional experiments) or Sections 4–9 (BYOD) without reloading the model trains the "frozen" policy and the k-NN baseline on an encoder already tuned on HAPT. When the rerun selects the frozen policy, the exported artifact holds only the head, and it no longer reproduces on the pinned base. In a small-input probe of that sequence, reloaded probabilities differed by up to 0.31 and one of six labels changed, so the Section 9 parity assertion would fail. The suggested `LEARNING_RATE = 3e-5` exercise ("watch the probe win") is the case that produces this.
3. **The interpretation reports a different run's result as "the finding" (MCL-M3).** The prose fixes 77.8 % accuracy, "two windows better", laying and sitting recall rising from 0.33 to 0.50, and "partly recovering the static postures". Those numbers come from a local CPU pre-flight of an older blob (`38bb74fc`). The recorded hosted run of this blob got 75.0 %, one window better, with laying recall unchanged at 0.33. The notebook itself says one window is about 2.8 points and that "a head cannot recover information the normalisation removed". This review confirmed that the pooled embedding is invariant to a per-channel offset and scale (max difference 1.5e-8), so the unfrozen blocks cannot receive the gravity component either.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (`metadata.dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (`metadata.dimer.notebook_spec`, opening cell) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated as such. Prerequisites: "basic Python, NumPy and pandas; what a long-format time-series table is; what a linear probe is …; what accuracy and macro-F1 measure; what validation-based selection between two policies means" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default; CUDA used automatically when present; float32 only |
| Promised outcomes | pinned install; carried package (12 modules); staged and digest-verified snapshot; digest-pinned HAPT corpus, 179 windows, 107 / 36 / 36 by volunteer; inference contract on three test windows (input manifest, rejection probe, `not-measurable` report, provenance); majority floor, cosine 5-NN, frozen policy; bounded unfreeze selected on validation log-loss; held-out accuracy / macro-F1 with per-class recall and a four-way comparison; before/after predictions; safetensors adapter export with reload parity; BYOD through the same stages |
| Learning objectives | install; read package guarantees; verify the model revision; fetch, validate and split a real corpus by volunteer; push windows through the inference contract and read its outputs; read accuracy and macro-F1 beside a floor and k-NN; train a probe and a bounded unfreeze with validation-based selection; evaluate on an independent split; compare predictions before and after; export and reload an adapter with verified parity |

### Existing execution evidence

- `docs/release-verification.md`: Kaggle Tesla T4, commit `ac2f9ef` / blob `7c5c3732` (= the reviewed blob), 2026-09-19, **PASSED — 20/20 code cells ok (1 restart after install cell)**, 294.3 s. Comparison: floor 0.1667, k-NN 0.7222, frozen 0.7222, selected 0.75 (unfrozen, best epoch 3); reload parity exact. `tutorials/README.md` and `STATUS.md` list the notebook as **Release-grade**.
- The archived run (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-moment-classification/v1/evidence/`) holds `executed-pass1.ipynb` and `executed.ipynb`. Pass 1 stops in cell 3 with the restart `RuntimeError` quoted above. The pinned install pulled `torch 2.14.0+cu130` and its CUDA wheels (downloads of up to 555 MB each). Pass 2 completes.
- The same file records a local CPU pre-flight of a different blob (`b0020d9` / `38bb74fc`, 2026-09-19, 77.8 %). It is marked "not promotion evidence". No hosted run exists on CPU (the declared default runtime) or on Colab, and none covers BYOD or an optional experiment.

### Evidence obtained by this review

- **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`, 4 threads), the repository's locked environment (`uv sync --locked --no-dev` in the review worktree: Python 3.12, torch 2.14.0 CPU, numpy 2.5.3, pandas 3.0.5, transformers 5.16.1, `momentfm` at the pinned commit). The install cell ran with the notebook's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1` hook. The snapshot and the HAPT archive were pre-staged by copy and re-verified by the notebook's own code. **This is not a Colab run.**
- **P1 static:** JSON parses (nbformat 4.5); 20 code cells compile; no persisted outputs; blob equals the recorded-run blob. `tools/build_notebook.py --template tools/notebook_template_classification.py --check` exit 0; `tools/validate_release_assets.py` exit 0 (PASS). The 12 carried module cells hold 180,023 characters, with no `cellView: form` and no Infrastructure label. The Prerequisites cell shows literal `{{id, x, label}}` and `{{1,64}}`. The BYOD branch has no location field, and `DIMER_BYOD_PATH` is not read.
- **P2 clean default (direct, CPU):** code cells 3–41 in one namespace **exceeded the 590 s cap and were stopped**. Not verified by direct execution.
- **P3 stale-state mechanism (direct, CPU, stand-in inputs: 24 train / 12 validation / 6 test windows of the default split):** `adapt(trainable_blocks=1, epochs=2, lr=1e-3)` selected the unfreeze (best epoch 2) and changed 9 parameter tensors inside `pipe`. A second `adapt(trainable_blocks=0)` on the same `pipe` returned the frozen policy with 0 trainable names. Embeddings from that `pipe` differed from a fresh base by up to 0.056. `save_artifact` wrote only `head.bias` and `head.weight`. After `load_artifact` on a fresh `load_moment` instance, probabilities differed by up to 0.3135, and one of six labels changed (`walking` → `walking_upstairs`). The Section 9 assertion `probabilities_identical` would be `False`.
- **P4a normalisation (direct, CPU):** pooled embeddings of a HAPT `sitting` window, of the same window plus a per-channel constant offset, and of the window ×3 differ by at most 1.5e-8 (offset) and 2.0e-6 (scale); cosine 1.0. Mean accelerometer (x, y, z) per posture: sitting (0.90, 0.27, 0.26), standing (1.01, −0.14, 0.04), laying (0.06, 0.66, 0.57). The gravity direction separates the postures in the raw data and is invisible to the encoder.
- **P4b BYOD (direct, CPU):** a `.zip` of 72 HAPT windows from 12 volunteers with `records.csv` (`id,file,label,group`) loads, splits 48 / 12 / 12 by group with a clean disjointness check, validates, and reaches the frozen-policy `adapt` and `evaluate` (accuracy 0.75 on 12 windows, `measured-small-sample`). The unfreeze, export and reload were not run on BYOD (time cap). Invalid inputs: a `records.csv` without `group`, windows of length 500, and a `.csv` instead of a `.zip` each raise an actionable `ValueError`. `import google.colab` outside Colab raises `ModuleNotFoundError: No module named 'google'`.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | The default path is correct on the recorded hosted run: digests, volunteer-disjoint split, selection on validation only, test used once, reload parity exact. The install step forces a restart on hosted images (MCL-M1). Rerunning a section after the default run leaves `pipe` modified, which invalidates the "frozen" comparison and can fail the reload assertion (MCL-M2). Section 3 prints `device: None` (MCL-m5). |
| Scientific validity | Split, selection and baselines are sound for a tutorial, and the limits are stated honestly. The representation claim (instance normalisation removes gravity) is correct (P4a). The interpretation draws a "partly recovered postures" conclusion from a one- or two-window change in another run, contradicting its own noise statement (MCL-M3). |
| Promise fulfilment | Every promised stage runs on the default path. The before/after narrative and "the finding" describe a run the learner will not see (MCL-M3). BYOD works up to evaluation in the probe but is Colab-only (MCL-m2). |
| Learner experience | Strong section prose, "Look for / Success here means" notes, and an explicit limits section. The GUIDED layer is thin: no audience or How-to-use block, roadmap, glossary, predictions, checkpoints with sample answers, troubleshooting or conclusion scaffold. 180 k characters of carried code are not labelled as skippable infrastructure (MCL-m1). Runtime estimates conflict (MCL-m3). |
| Spec conformance | RUN1/RUN10/ENV6/REL2/REL11 fail on the recorded hosted run (MCL-M1). GDL8/GDL14/ENV8 not met for the interpretation (MCL-M3). GDL10/RUN9/VER3 are at risk on the documented rerun paths (MCL-M2). SHOULD gaps: GDL1–GDL14 (partial), EXE1/EXE2, UX12, MOD3 (display). Declared spec is 2.0, not 2.2. |

## 3. Promise and objective tracing

| Claim (cell) | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes with no intervention (0) | cell 3 pip install + stale-import guard | hosted pass 1 stops with a restart instruction | learner must restart and rerun | **fails on hosted evidence** (MCL-M1) |
| No GitHub access required (1) | `PINS` include `momentfm @ git+https://github.com/…` | pip clones from GitHub | prose inaccurate | MCL-m6 |
| Identity, device and weight source printed before inference (28) | cell 29 `getattr(pipe, 'device', None)` | hosted: `{'device': None, 'source': 'local-snapshot'}` | device looks unknown | MCL-m5 |
| Digest-pinned corpus, 179 windows, 107 / 36 / 36 by volunteer, no leakage (30) | cell 31 | hosted: as stated; digests `7ec96e15…` / `4490b2f9…` / `725d8711…` | — | verified (documented) |
| Four dataset refusals before torch (30) | cell 31 probes | hosted: all four rejected | — | verified (documented) |
| Inference contract on three windows; cosine "qualitative look" (32) | cell 33 | hosted: checks True, `not-measurable`; cosine same-activity 0.9844 < other-activity 0.9973 | no guidance on reading a reversed pair | partially (MCL-m7) |
| Floor, k-NN and frozen policy frame the result (34) | cell 35 | hosted: 0.167 / 0.722 / 0.722 | — | verified (documented) |
| Unfreeze selected on validation log-loss, probe competes (36) | cell 37 | hosted: 0.746 → 0.703 → 0.753 → 0.610, epoch 3 kept | — | verified (documented) |
| "77.8 %, two windows better, laying and sitting 0.33 → 0.50" (38, 42) | prose fixed in the template | hosted: 75.0 %, one window, laying 0.33 → 0.33 | learner reads a result they did not get | **contradicted** (MCL-M3) |
| "Partly recovering the static postures" (42) | prose | ≤ 1 window per class; embedding is offset/scale-invariant (P4a) | unsupported causal reading | **unsupported** (MCL-M3) |
| Export and fresh-reload parity (40) | cell 41 | hosted: identical probabilities, 0.75 both ways | — | verified (documented) on the default path; fails on a rerun after an unfreeze win (MCL-M2) |
| Optional experiments do not affect the default path (42) | prose | true for the first pass; reruns inherit the modified `pipe` | — | MCL-M2 |
| BYOD through the same stages (0, 30) | cell 31 `files.upload()` | probe: load → split → validate → probe → evaluate OK; Colab-only | — | partially (MCL-m2, MCL-M2) |

| Objective | Learner activity | Evidence it is exercised |
|---|---|---|
| Read accuracy and macro-F1 beside a floor and k-NN | read the four-way comparison | exercised by the printed comparison; no prompt asks the learner to compute the delta in windows or judge it against noise (MCL-M3, MCL-m1) |
| Probe and bounded unfreeze with validation-based selection | read the epoch ladder; optional LR / blocks / epochs changes | ladder printed; the suggested changes, rerun in place, run on a modified encoder (MCL-M2) |
| Validate and split by volunteer without leakage | read the summary and refusals | exercised (hosted output) |
| Compare predictions before and after | read six rows | exercised; the prose's example row (`test-003` standing → laying) differs from the hosted row (standing → walking_downstairs) (MCL-M3) |
| Export and reload with verified parity | read the parity dict | exercised on the default path |
| Install, verify | run cells | exercised; install needs a restart on hosted runtimes (MCL-M1) |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Clear per-section prose and an honest limits section. The interpretation asserts numbers and a causal reading the learner's own run will not show (MCL-M3). Guided-layer gaps and 180 k characters of unlabelled carried code (MCL-m1). Section 3 prints `None` for device (MCL-m5). |
| Clean default | Documented execution (Kaggle T4, same blob) | PASSED only after a restart (MCL-M1). Direct CPU execution exceeded the 590 s cap: **not verified** directly. No Colab run and no hosted CPU run of this blob. |
| Active learning | Direct execution (local CPU, stand-in subset; mechanism only) | After an unfreeze win, a rerun that selects the frozen policy exports a head-only artifact that does not reproduce on the base (max probability difference 0.31; 1 of 6 labels changed). The full `LEARNING_RATE = 3e-5` exercise was **not run** (time cap). |
| Reuse and recovery | Direct execution (local CPU, no Colab) | BYOD zip (72 windows, 12 groups) reaches validate → split → frozen probe → evaluate. Unfreeze, export and reload on BYOD **not run**. 3 invalid inputs rejected with actionable messages; `USE_BYOD = True` outside Colab fails with `ModuleNotFoundError`. |

## 5. Findings

### Major

#### MCL-M1 — `Run all` needs a manual restart after the install cell; recorded as PASSED / Release-grade

- **Cell/section:** Section 1 (cell 3; generated by `_INSTALL_GUARD`, `tools/build_notebook.py:47-70`, `pip install` at line 61); `docs/release-verification.md` row of 2026-09-19 (Kaggle T4); `tutorials/README.md` registry row; `STATUS.md`.
- **Observed issue:** the pinned install replaces packages the hosted kernel has already imported. The guard then raises `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` (archived `executed-pass1.ipynb`, cell 3). The run passed only on a second pass. The record nevertheless says "PASSED — 20/20 code cells ok (1 restart after install cell)", and the registry calls the notebook Release-grade. The opening promises that Run all needs "no configuration edit" and no intervention.
- **Consequence:** a first-time learner's `Run all` stops in cell 3. A restart-dependent run is presented as release evidence.
- **Evidence:** documented execution (archived pass-1 / pass-2 notebooks, same blob `7c5c3732`); source inspection (cell 3).
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern. A carrier cell bootstraps uv, runs `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`. Emit it from the install-cell builder in `tools/build_notebook.py`, not by hand. The `momentfm` source pin needs a hash-pinned archive in the lock (the prithvi-flood reference notebook builds a source-only pin from its hash-pinned archive). Until a one-pass hosted run exists, record the 2026-09-19 run as not Run-all conformant and set the registry status to Candidate.
- **Acceptance check:** on a fresh Colab (or Kaggle) runtime, a single `Run all` of the new blob completes every code cell with no error output and no restart, and `docs/release-verification.md` records that run (blob, runtime, outcome) with no restart note.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### MCL-M2 — Rerunning a section after the default run trains on an already-adapted encoder; a frozen-policy rerun exports an artifact that fails reload parity

- **Cell/section:** Section 7 (cell 37) and the carried `adapt` (`src/moment_pipeline/adaptation.py:294-498`, "The encoder inside `model.pipeline` is modified in place when the unfrozen policy wins"); the optional experiments (cell 42, `tools/notebook_template_classification.py:488`); the BYOD instruction "set `USE_BYOD = True` in Section 4 and re-run from that cell" (cell 0, template line 41); Section 9 (cell 41).
- **Observed issue:** on the default path the unfreeze wins (both recorded runs: best epoch 3), so `pipe` leaves Section 7 with two modified encoder blocks. Nothing reloads the base afterwards, and the notebook never says which cells to rerun for an experiment. A learner who changes `LEARNING_RATE`, `TRAINABLE_BLOCKS` or `EPOCHS` and reruns Sections 7–9, or switches to BYOD and reruns from Section 4, gets:
  1. a k-NN baseline and a "frozen policy" (Section 6 rerun, or the epoch-0 probe inside `adapt`) computed on the HAPT-adapted encoder, not the pretrained one;
  2. an unfreeze that starts from the previously trained blocks; and
  3. when the frozen policy is selected, which is the outcome the `3e-5` exercise tells the learner to expect, an artifact holding only `head.*`. That head was trained on the modified encoder's features, and `load_artifact` overlays it on a fresh base, so the Section 9 parity assertion fails.
  `save_artifact` records the pinned base digest in the manifest, although the in-memory encoder no longer matches it.
- **Consequence:** the documented active-learning experiments compare against the wrong baseline and can end in an `AssertionError` in Section 9. A BYOD user's "frozen policy" and k-NN numbers are not those of the pretrained encoder, and the artifact may not reproduce.
- **Evidence:** direct execution (P3, local CPU, stand-in subset of the default split): after an unfreeze win, a frozen-policy rerun exported 2 tensors (`head.bias`, `head.weight`). Reloaded on a fresh base, probabilities differed by up to 0.3135, 1 of 6 labels changed, and `probabilities_identical` was `False`. Source inspection (`adaptation.py`, cells 35–41). The full-size `3e-5` rerun was not executed. That its probe wins on the adapted encoder is inferred.
- **Recommended correction:** in `tools/notebook_template_classification.py`, start Sections 6 and 7 from a freshly loaded, digest-verified base (`pipe = load_moment(task="embedding", weights_dir=WEIGHTS_DIR)`), or make `adapt` work on a copy and return the trained blocks. Alternatively, make `save_artifact` refuse to export when any non-exported encoder tensor differs from the pinned base. Give explicit rerun instructions in the experiments and BYOD text ("edit X, then rerun from Section N"), with a bounded Predict → Change → Run → Observe → Explain prompt.
- **Acceptance check:** after a default `Run all`, (a) setting `LEARNING_RATE = 3e-5` and following the stated rerun instruction reproduces the default run's k-NN and frozen-policy test numbers exactly, and Section 9's parity assertion passes whichever policy wins; (b) switching to BYOD and following the stated instruction computes the k-NN and frozen policy on embeddings equal to a fresh base's.
- **Spec:** GDL10, RUN9, DAT13, VER3/VER5, SRC2 (hidden state dependence).

#### MCL-M3 — The interpretation fixes another run's numbers and a causal reading the evidence does not support

- **Cell/section:** Section 7 markdown (cell 36, template line 314), Section 8 markdown (cell 38, template line 344), Interpretation (cell 42, template line 462), optional experiments (template line 488).
- **Observed issue:** the prose presents "the build record" result as the claim and the finding: selected policy 77.8 % / macro-F1 0.776, "two windows better", "laying and sitting recall rising from 0.33 to 0.50", "partly recovering the static postures", and LR sweep outcomes (1e-4 → 75.0 %; 3e-5 → probe kept). Those numbers come from the local CPU pre-flight of blob `38bb74fc` (`docs/release-verification.md:135`), not from a run of this blob. The recorded hosted run of this blob printed 75.0 % / 0.741, Δ +0.0278 (one window), laying recall 0.33 → 0.33, sitting 0.33 → 0.50, and test-003 changed to `walking_downstairs` (still wrong), not to the gold `laying`. The notebook itself says one window is about 2.8 points, "a three-point delta is noise", and "a head cannot recover information the normalisation removed". Normalisation happens before the encoder, so the unfrozen blocks cannot recover it either. P4a confirms that the pooled embedding is invariant to a per-channel offset and scale (max difference 1.5e-8), while the postures' raw mean accelerations differ by gravity direction. The notebook does not mention CPU-versus-CUDA numeric variation, although the two records differ by one window on the same split.
- **Consequence:** the learner's printed comparison contradicts the text they are told is "the claim and the finding". The notebook teaches an effect (unfreezing partly recovers postures) that its own numbers call noise and its own representation argument rules out. This is the conclusion the learner leaves with.
- **Evidence:** documented execution (hosted `executed.ipynb` cells 39 and 41); direct execution (P4a); source inspection (template lines cited).
- **Recommended correction:** in the template, replace fixed results with **What to notice** notes that read the printed comparison: compute the delta in windows (`round(delta * n_test)`), compare it with the one-window ≈ 2.8-point resolution, and say what a 0–2 window difference does and does not show. Remove "partly recovering the static postures", or state it only as a hypothesis with the test that would check it (for example several split seeds). Name the build environment for any example figure, and add one line that CPU and CUDA runs of the same split can differ by a window. Add a conclusion template (task, principal result, baseline, failure mode, limits).
- **Acceptance check:** the regenerated notebook contains no build-record accuracy, recall or before/after example presented as the reader's result. The interpretation tells the learner to read the delta from their own output in windows. No sentence attributes a posture-recall change of one window or less per class to the unfreeze. A note names device-dependent variation.
- **Spec:** GDL8, GDL14, ENV8, EVAL15 (context), UX4.

### Minor

#### MCL-m1 — GUIDED layer is partial; 180 k characters of carried code are not marked as infrastructure

- **Cell/section:** opening; Section 2 (cells 4–27); final cell.
- **Observed issue:** no explicit intended learner or How-to-use block, roadmap, Input → Model → Output line, or glossary (terms such as macro-F1, log-loss, linear probe, instance normalisation, T5 block are used without one). No prediction before the comparison or the unfreeze, no interpretation checkpoints with sample answers, no troubleshooting section (install restart, Hub/UCI download, memory, BYOD upload), and no conclusion scaffold. The 12 carried module cells (180,023 characters) have no Infrastructure label and no `cellView: form`. Learning objectives open with "install", "read" and "push".
- **Consequence:** self-paced learners get a strong reference walkthrough rather than a guided lesson, and must scroll past 180 k characters of code before the first learner stage.
- **Evidence:** source inspection (P1).
- **Recommended correction:** add these elements in `tools/notebook_template_classification.py` and the shared opening and carrier builders in `tools/build_notebook.py`. Title carrier cells `# @title Infrastructure: …` with `cellView: form`, as in the 2.2 reference notebook (`prithvi-flood-segmentation-pipeline`). Add a prediction prompt before Section 8 (for example: "Will unfreezing help the walking classes, the postures, or neither? Why?") with a collapsible sample answer.
- **Acceptance check:** the regenerated notebook has an audience/How-to-use block, a roadmap, an Input → Model → Output line, a glossary, at least one prediction with a collapsible sample answer, a troubleshooting list, Infrastructure-labelled collapsed carrier cells, and a conclusion template. It declares spec 2.2.
- **Spec:** GDL1–GDL7, GDL9, GDL11–GDL14.

#### MCL-m2 — BYOD is Colab-only and has no location field

- **Cell/section:** Section 4 (cell 31; template line 174).
- **Observed issue:** `USE_BYOD = True` always imports `google.colab` and opens `files.upload()`. There is no `BYOD_PATH` form field, and `DIMER_BYOD_PATH` (used by the sibling notebooks) is not read. Prerequisites name "Google Colab or Jupyter", but outside Colab the branch fails with `ModuleNotFoundError: No module named 'google'` and no recovery hint. The loader itself accepts a directory or a zip, and works (P4b).
- **Consequence:** a Jupyter user cannot use BYOD without editing the cell, and an executor cannot drive it.
- **Evidence:** direct execution (P4b, P4c); source inspection.
- **Recommended correction:** add `BYOD_PATH = ''  # @param {type:"string"}`; read from it when set, and import `google.colab` only when it is empty; catch the import failure with a message naming `BYOD_PATH`.
- **Acceptance check:** with `USE_BYOD = True` and `BYOD_PATH` pointing at a valid zip, Section 4 loads it in plain Jupyter without importing `google.colab`. With `BYOD_PATH` empty outside Colab, the error names the field to set.
- **Spec:** EXE1, EXE2, UX10, DAT16.

#### MCL-m3 — Runtime estimates conflict and the measuring environment is not named; the declared default runtime has no hosted run

- **Cell/section:** opening (template line 38: "about five minutes of model time"), Prerequisites (template line 138: three-epoch unfreeze "took about 150 s"), Section 6 ("about half a minute on CPU"), Section 7 ("about three minutes on CPU"); `tutorials/README.md` ("~seven min of model time on CPU").
- **Observed issue:** the release record's CPU pre-flight measured the unfreeze at 307.5 s and the whole path at 430 s. This review's direct CPU run of cells 3–41 exceeded 590 s. "The build record" never names its machine. The declared default runtime is CPU, but the only clean-runtime record of this blob is a T4 GPU run; the CPU figures come from a pre-flight of a different blob.
- **Consequence:** a CPU learner may think the notebook has hung during Section 7; the CPU path is unverified at this revision.
- **Evidence:** documented execution; direct execution (P2 timeout); source inspection.
- **Recommended correction:** use one set of measured figures with the environment named (for example "Colab CPU, 2 vCPU: …") and label others as estimates. Either record a hosted CPU run or declare T4 as the verified runtime in the registry.
- **Acceptance check:** every runtime figure in the notebook and registry agrees and names its environment, and `docs/release-verification.md` holds a clean run on the runtime the notebook calls default.
- **Spec:** UX12, REL1, REL10.

#### MCL-m4 — Prerequisites show escaped braces: `{{id, x, label}}` and an id pattern `{{1,64}}`

- **Cell/section:** Prerequisites (cell 1; template line 140, which is not passed through `.format` but is written as if it were).
- **Observed issue:** the rendered text shows double braces. The id pattern a learner would copy, `[A-Za-z0-9_.:-]{{1,64}}`, is not the validator's pattern.
- **Consequence:** small confusion; a copied regex is wrong.
- **Evidence:** source inspection (P1).
- **Recommended correction:** use single braces on that line (or format it like the section cells).
- **Acceptance check:** the rendered Prerequisites show `{id, x, label}` and `[A-Za-z0-9_.:-]{1,64}`.
- **Spec:** SRC3 (knowingly stale text), UX2.

#### MCL-m5 — Section 3 prints `device: None` and a hard-coded `source`

- **Cell/section:** cell 29, last line (`tools/build_notebook.py:548`; shared with the sibling notebooks).
- **Observed issue:** `LoadedMoment` has no `device` or `source` attribute, so the line prints `{'device': None, 'source': 'local-snapshot'}` (hosted record). The markdown promises "The effective identity, device and weight source are printed before any inference". The real device (`pipe.identity.device`) first appears in Section 5.
- **Consequence:** the learner sees `device: None` at the point where the notebook says the device is confirmed.
- **Evidence:** documented execution; source inspection.
- **Recommended correction:** print `pipe.identity.device`, `pipe.identity.dtype` and `pipe.identity.weight_file_loaded` in the generator.
- **Acceptance check:** Section 3 prints the real device (`cpu` / `cuda`) and the weight file.
- **Spec:** MOD3, ENV3.

#### MCL-m6 — "No GitHub access … required" is false

- **Cell/section:** Prerequisites, last bullet (`tools/build_notebook.py:441`).
- **Observed issue:** `PINS` installs `momentfm @ git+https://github.com/moment-timeseries-foundation-model/moment@38f7310…`, which pip fetches from GitHub. The opening paragraph says so; the Prerequisites deny it.
- **Consequence:** a learner on a network that blocks GitHub gets an install failure the notebook says cannot happen.
- **Evidence:** source inspection.
- **Recommended correction:** state that the install fetches the pinned upstream `momentfm` source from GitHub (no DIMER repository source).
- **Acceptance check:** the external-access bullets list GitHub for the `momentfm` source pin.
- **Spec:** ST6 (explicit acquisition), SRC10.

#### MCL-m7 — The cosine "qualitative look" gives no guidance and the hosted output inverts the expected pair

- **Cell/section:** Section 5 (cell 32 markdown, template line 225; cell 33).
- **Observed issue:** the hosted run printed `cosine_same_activity 0.9844` and `cosine_other_activity 0.9973`: the cross-activity pair is closer. Both are near 1, from one pair each. The prose calls it a qualitative look at the representation but says nothing about how to read it, and it does not anticipate that pooled MOMENT vectors are all close in cosine.
- **Consequence:** the learner's first look at the representation suggests it does not separate activities, just before k-NN gets 72 %, and nothing reconciles the two.
- **Evidence:** documented execution; source inspection.
- **Recommended correction:** add a What-to-notice note (cosines of pooled vectors cluster near 1; one pair is anecdotal), or replace the pair with a small within- versus between-class mean-cosine table on the training split.
- **Acceptance check:** Section 5 tells the learner how to read the printed cosines, and the output gives a comparison that does not depend on a single pair.
- **Spec:** GDL8, UX4.

### Suggestions

- **MCL-S1 — Make the normalisation claim a learner activity.** P4a shows the embedding ignores per-channel offset and scale, while the postures' mean accelerations differ by gravity direction. An optional experiment that appends the per-channel window means as side features to the probe (outside the encoder) would let the learner test the notebook's central representation claim (GDL10).
- **MCL-S2 — Add a dispersion estimate.** A bootstrap interval over the 36 test windows, or two or three split seeds for the floor, k-NN and frozen policy, would turn "one window ≈ 2.8 points" into a number the learner can read beside the delta.

## 6. Readiness

**Needs revision.** Remaining gates:

1. MCL-M1: a one-pass hosted `Run all` without a restart, recorded for the new blob; the registry returns to Candidate until then.
2. MCL-M2: reruns start from a pristine base (or the export refuses a modified base), with explicit rerun instructions; verified by the acceptance check.
3. MCL-M3: the interpretation reads the learner's own output, and the unsupported causal claim is removed.
4. Execution evidence on the declared default runtime (MCL-m3), plus a hosted BYOD positive and negative run (REL12). This review's local BYOD probe reached evaluation only.

Minor findings may remain open, but MCL-m2 and MCL-m5 are cheap fixes in the shared generator.

## 7. Verified versus inferred

- **Verified by this review:** reviewed SHA and blob; static parse, compile, generator `--check` and validator PASS (P1); the stale-state mechanism and the parity failure at small scale (P3); embedding invariance to per-channel offset and scale (P4a); the BYOD loader, split, validation, probe and evaluation on a 72-window zip, plus 4 refusals (P4b/c); the hosted record's numbers and the pass-1 restart error, read from the archived notebooks.
- **Inferred:** that the full-size `LEARNING_RATE = 3e-5` rerun after the default run selects the frozen policy and fails Section 9 (the mechanism is verified; the full-size outcome is not); that a Colab CPU runtime also forces the restart (shown on Kaggle CPU for the sibling notebooks and on Kaggle T4 here).
- **Not verified:** the clean default path by direct execution (stopped at 590 s); any Colab run; BYOD unfreeze, export and reload.
- **Most likely to be wrong:** MCL-M2's severity for the full-size experiment. If, on the adapted encoder, the `3e-5` rerun still selects an unfrozen epoch, the export carries the blocks and parity holds, leaving only the contaminated baseline (still Major for validity, but no crash).

*Probe files (`run_probes.py`, `results.json`, `source_manifest.json`) are in `moment_classification_colab_Review_Probes.zip` beside this report.*
