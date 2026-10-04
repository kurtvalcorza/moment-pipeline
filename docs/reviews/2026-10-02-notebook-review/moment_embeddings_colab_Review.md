# MOMENT Embeddings Tutorial Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/moment-pipeline`  
**Notebook:** `tutorials/moment_embeddings_colab.ipynb`  
**Reviewed commit:** `44526760c75d43a4f314252fef297562c9651a32` (`main`, confirmed with `gh api repos/kurtvalcorza/moment-pipeline/commits/main`)  
**Notebook Git blob:** `b294013f0fa7563c9eaea18d34f120ff4c19f99e`. This is the blob executed in the recorded Kaggle CPU run of 2026-09-14 (commit `31ddb06`); the notebook has not changed since.  
**Finding prefix:** `MEM`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The engineering is sound. The notebook digest-verifies the pinned 3-file snapshot, regenerates the deterministic synthetic sample and asserts its checked-in digest, validates the input into a manifest (with a recorded `EMPTY_CHANNEL` rejection), extracts one mean-pooled 768-dimensional vector per window, reports `not-measurable` with a statement of what labelled data would be needed, and exports identifier-preserving vectors with provenance. This review's local CPU run reproduced the hosted embedding to a maximum absolute difference of 1.5e-7. BYOD through `DIMER_BYOD_PATH` reaches every stage, and six invalid inputs are rejected with coded messages before the model runs.

Two things need fixing before the notebook teaches what it says:

1. **No one-pass `Run all` (MEM-M1).** The recorded Kaggle run of this exact blob failed in the install cell (`numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.`) and passed only on a second pass after a restart. The release record calls this PASSED and the registry calls the notebook Release-grade.
2. **The missingness lesson is contradicted by the notebook's own labels and never exercised (MEM-M2).** The prose says pre-filled missing positions are visible to the encoder. The cell output says `missingness visible to model: False`, and the input manifest says `NaN marks a source-missing point, which never reaches the model`. The default sample has no missing values, so the learner never sees the effect. With 12.5 % of points missing, this review measured cosine 0.80 between the gappy and clean embeddings (L2 distance 5.06 on a vector of norm 7.22); the gappy embedding is bit-identical to writing `0.0` into the gaps, and linear interpolation of the same gaps gives cosine 0.9996.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `TASK-INFERENCE` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (`metadata.dimer.notebook_spec`, opening cell) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated as such. Prerequisites: "basic Python and pandas; what a long-format time-series table is" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default; CUDA used when present; float32 only |
| Promised outcomes | pinned install; carried package (10 modules); staged and digest-verified snapshot; deterministic synthetic sample (or BYOD CSV); validation into an input manifest; pooled embeddings through the production API; shape/pooling/channel/missingness semantics; `not-measurable` evaluation report; identifier-preserving CSV, provenance and result JSON |
| Learning objectives | install; read package guarantees; verify the model revision; generate the sample or bring a CSV; validate/canonicalize; extract pooled embeddings; "interpret embedding shape, pooling, channel, and missingness semantics"; produce a `not-measurable` report; export with provenance |
| Explicitly not demonstrated | classification, forecasting, anomaly decisions, fine-tuning, downstream suitability |

### Existing execution evidence

- `docs/release-verification.md`: Kaggle CPU (image torch 2.10.0+cpu), commit `31ddb06` / blob `b294013f` (= the reviewed blob), 2026-09-14, **PASSED — 17/17 code cells ok (1 restart after install cell)**, 221.1 s, 454 MB staged from the Hub. `tutorials/README.md` lists the notebook as **Release-grade**.
- The archived run (`.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-moment-embeddings/v1/evidence/`, workspace only) holds `executed-pass1.ipynb`, `executed.ipynb` and `run_summary.json` (`restarted_after_install_cell: true`). Pass 1 stops in cell 3 with `RuntimeError: Core dependencies changed while older modules were loaded: numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` (183 s). Pass 2 completes (38 s). The pinned install pulled `torch 2.14.0+cu130` and NVIDIA wheels (several downloads of 145–555 MB each) onto the CPU image.
- No hosted run covers BYOD, missing-value input, or a Colab runtime.

### Evidence obtained by this review

- **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`, 4 torch threads), venv `dimer-moment` (Python 3.12.10, torch 2.14.0+cpu, numpy 2.5.3, pandas 3.0.5, transformers 5.16.1, momentfm at the pinned commit). Install cell run with the notebook's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`; the 3 snapshot files pre-staged by copy (`fetched = []`) and then verified by the notebook; `HF_HUB_OFFLINE=1`; `google.colab.files.upload` was a shim. **This is not a Colab run.** Scale: the default sample is the full default (1 series, 2 channels, 256 steps); BYOD probes use small synthetic CSVs (≤ 1,424 rows).
- **P1 static:** JSON parses (nbformat 4.5); 17 code cells compile; no persisted outputs; blob equals the recorded-run blob. `tools/build_notebook.py --check` exit 0; `tools/validate_release_assets.py` exit 0 (PASS). Section 2 carries 124,812 characters of module code.
- **P2 default path (direct, CPU):** 17/17 cells ok in 27.3 s (model load 18.9 s, embedding 0.27 s). Shape `(1, 768)`, L2 norm 7.218, verdict `not-measurable`, 5 output files. Max abs difference against the hosted `moment_embeddings.csv`: 1.5e-7. Section 3 printed `{'device': None, 'source': 'local-snapshot'}` and the upstream warning "Only reconstruction head is pre-trained. Classification and forecasting heads must be fine-tuned."
- **P3 semantics (direct, CPU, same model instance):**
  - 64 of 256 `vibration` points set to NaN (every 4th step; `masked_point_fraction` 0.125): the flag still reads `missingness_visible_to_model = False`; cosine to the clean vector **0.803**, L2 distance **5.06**; the same positions written as literal `0.0` give a **bit-identical** vector (max abs diff 0.0); linearly interpolating the gaps gives cosine **0.9996** (L2 0.20).
  - Every value × 10 + 5: cosine 0.99999999998, max abs diff 5.7e-6 (RevIN removes per-channel level and scale).
  - `vibration` alone vs `temperature` alone: cosine 0.972; each vs the 2-channel vector: 0.993.
- **P4 BYOD and recovery:** a two-series, two-channel CSV (`flow`, `pressure`; A 600 h → truncated; B 120 h → padded, with an 8-step gap → listed in `irregular_series`) runs Sections 4–8 through `DIMER_BYOD_PATH`: shape `(2, 768)`, window ids `A::w0`, `B::w0`, 2 CSV rows, `sample.kind = byod` with digest. The two unrelated series have cosine 0.993. Only window 0's L2 norm is printed. Invalid inputs: duplicate header (`DUPLICATE_COLUMNS`), missing `value` (`MISSING_COLUMNS`), Latin-1 bytes (`INVALID_CSV_ENCODING`) in cell 27; unparseable timestamp (`UNPARSEABLE_TIMESTAMP`), non-numeric value (`NON_NUMERIC_VALUE`), duplicate key (`DUPLICATE_ROWS`) in cell 29. All raise before the model runs. A cancelled upload raises `ValueError: Upload exactly one CSV with columns series_id,timestamp,channel,value.` `USE_BYOD=True` outside Colab raises `ModuleNotFoundError: No module named 'google'`.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | The default path is correct and deterministic (local and hosted vectors agree to 1.5e-7). Supply-chain verification, validation and export are sound. The install step forces a restart on hosted images (MEM-M1). Section 3 prints placeholder device and source values (MEM-m3). |
| Promise fulfilment | Every promised stage runs and produces its output. The missingness objective is not exercised by the default sample, and the printed flag and manifest text say the opposite of the prose (MEM-M2). Scale/level invariance, part of what the vector represents, is not stated (MEM-m2). |
| Learner experience | Clear section prose with "Successful output means…" notes and an honest limits section. The GUIDED layer is thin: no roadmap, Input → Model → Output contract, prediction, comparison, checkpoints, troubleshooting or conclusion template; the learner sees one vector and has nothing to compare it with. About 125 k characters of carried code are not labelled as skippable infrastructure (MEM-m1). |
| Spec conformance | RUN1/RUN10/ENV6/REL2/REL11 fail on the recorded hosted run (MEM-M1). UX1/UX4 not met for the missingness objective (MEM-M2). §21.6 shape/pooling/unit/representation statements and identifier export are met. SHOULD gaps: GDL1–GDL14 (partial), UX5, EXE1/EXE2/EXE5, SRC10/UX12. Declared spec is 2.0, not 2.2. |

## 3. Promise and objective tracing

| Claim (cell) | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes with no intervention (0) | cell 3 pip install + stale-module guard | Hosted pass 1 raises restart error | Learner must restart and rerun | **Not met** (MEM-M1) |
| Pinned, digest-verified model (0, 24) | cell 25: manifest assert, `stage_missing_files`, `verify_snapshot`, `load_moment` | 3 files verified; revision printed | Clear, except `device: None` | Met (MEM-m3 display) |
| Deterministic sample, digest asserted (26) | cell 27 `build_samples` + SHA-256 check | digest `34fc4758…` printed | Clear | Met |
| Validate before the model; disclose padding/truncation (28) | cell 29 `validate_inputs`, `to_windows`, empty-channel probe | manifest; "padded windows: 1 / 1"; `EMPTY_CHANNEL` finding | Clear | Met |
| Pooled per-window embedding, channels averaged (30) | cell 31 `embed` | shape `(1, 768)`, reduction, channel policy printed | Clear | Met |
| Missingness semantics (0 objectives, 30) | `EmbeddingResult.missingness_visible_to_model=False`; manifest schema text | Sample has 0 missing; flag prints `False` | Learner likely concludes missing values are ignored | **Not delivered** (MEM-M2) |
| `not-measurable` report with stated needs (32) | cell 33 `evaluation_report` | verdict, empty metrics, `needs` | Clear (EVAL9 met) | Met |
| Identifier-preserving export + provenance (34) | cell 35 | 5 files; `series_id`, `window_id` columns | Clear (OUT4, OUT6 met) | Met |
| BYOD through the same path (0, 26) | cell 27 branches | P4: 2 windows end to end; 6 coded rejections | Clear in Colab; fails on Jupyter | Met in Colab (MEM-m5) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Interpret shape, pooling, channel semantics | Read printed shape / policy strings | Printed only; no comparison or prediction |
| Interpret missingness semantics | None (sample has no missing data) | Not exercised (MEM-M2) |
| Produce a `not-measurable` report | Run cell | Exercised (report printed and written) |
| Validate / canonicalize | Run cell; read manifest and rejection | Exercised |
| Bring your own CSV | Optional; set `USE_BYOD` | Exercised by this review via `DIMER_BYOD_PATH` |

## 4. Journeys

| Journey | Evidence basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | Prose is accurate and well signposted per section. Blocked at cell 3 on hosted images (restart). The learner gets one vector and contradictory missingness labels; no prediction, comparison or checkpoint; 10 carried-module cells not labelled as infrastructure. |
| Clean default | Documented execution (Kaggle CPU, same blob: pass 1 fails at install, pass 2 passes); direct execution (local CPU, install skipped, 17/17 ok, outputs match hosted to 1.5e-7) | **Not one-pass.** Passes after a restart. Final outputs belong to the run. |
| Active learning | Direct execution (local CPU) | The notebook has no in-notebook exercise. Its suggested next experiment (multi-series BYOD, inspect padding/truncation) works and discloses both. The suggested missingness comparison is not provided as a runnable activity; this review's probe shows it would be instructive (cosine 0.80 vs 0.9996). |
| Reuse and recovery | Direct execution (local CPU; upload shim) | BYOD via `DIMER_BYOD_PATH` reaches every stage; 6 invalid inputs rejected with coded, actionable messages before the model; cancelled upload gives an actionable `ValueError`; `USE_BYOD=True` on Jupyter fails with `ModuleNotFoundError`. Colab upload widget: **not verified**. |

## 5. Findings

### Major

#### MEM-M1 — `Run all` needs a manual restart after the install cell; recorded as PASSED / Release-grade

- **Cell/section:** Section 1 (cell 3; `tools/build_notebook.py` install-cell builder); `docs/release-verification.md` row of 2026-09-14; `tutorials/README.md` registry row.
- **Observed issue:** the pinned install replaces packages the hosted kernel has already imported. The guard then raises `RuntimeError: … numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` (archived `executed-pass1.ipynb`, same blob). The run passed only on a second pass. The record says "PASSED — 17/17 code cells ok (1 restart after install cell)" and the registry calls the notebook Release-grade, although the opening promises that Run all needs no intervention. On the CPU image the pins also pull the CUDA build of torch 2.14.0 and its NVIDIA wheels.
- **Consequence:** a first-time learner's `Run all` stops in cell 3. A restart-dependent run is presented as release evidence.
- **Evidence:** documented execution (archived pass 1 / pass 2 notebooks and `run_summary.json`, `restarted_after_install_cell: true`); source inspection (cell 3).
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern. A carrier cell bootstraps uv, runs `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:` (CPU torch wheels for the CPU path), and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`. Generate it from `tools/build_notebook.py`, not by hand, so the four MOMENT notebooks change together. Until a one-pass hosted run exists, record the 2026-09-14 run as not Run-all conformant and set the registry status to Candidate.
- **Acceptance check:** on a fresh Colab (or Kaggle) CPU runtime, a single `Run all` of the new blob completes every code cell with no error output and no restart, and `docs/release-verification.md` records that run (blob, runtime, outcome) with no restart note.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### MEM-M2 — The missingness objective is never exercised, and the printed labels contradict the prose

- **Cell/section:** opening objectives (cell 0); Section 5 manifest (cell 29; `src/moment_pipeline/roles.py:42`); Section 6 (cells 30–31; `tools/notebook_template.py:198`, `src/moment_pipeline/embedding.py:83`).
- **Observed issue:** the objectives promise that the learner will "interpret … missingness semantics". Section 6's prose says pre-filled missing positions are visible to the encoder. But the cell prints `missingness visible to model: False`, and the manifest printed one section earlier says `"values": "numeric; NaN marks a source-missing point, which never reaches the model"`. Both are literally true (the *mask* and the *NaN* do not reach the model) but read as "missing values are ignored", which is the opposite of the contract stated in `embedding.py` and `MODEL_CARD.md` ("Embeddings are NOT missingness-aware"). The default sample has `masked_point_fraction 0.0`, so the learner never sees what missing data does to a vector. The limits section's "next experiment" (compare clean vs missingness-bearing windows) has no runnable cell.
- **Consequence:** the learner can leave believing gappy series are embedded safely. In fact, with 12.5 % of points missing, the vector moved to cosine 0.80 from the clean one, and it is identical to the vector obtained by writing zeros into the gaps.
- **Evidence:** direct execution (P3: NaN vs literal 0.0 max abs diff 0.0; clean vs gappy cosine 0.803, L2 5.06; clean vs interpolated cosine 0.9996, L2 0.20; flag `False` throughout); source inspection (cells 29–31, `roles.py:42`, `embedding.py:44-52`).
- **Recommended correction:** in `tools/notebook_template.py`, rename the printed label to what it means (for example `per-point missingness mask passed to model: False (pre-filled values ARE seen as data)`) and print `MISSINGNESS_POLICY` beside it. In `roles.py`, say for the embedding task that NaN is replaced by `prefill_value` and that the filled value is seen by the encoder. Add a short, bounded, non-blocking experiment after Section 6: embed a copy of the sample with a stated fraction of points set to NaN, and a copy with those gaps interpolated, and print cosine/L2 against the clean vector, with a Predict prompt before it and a "What to notice" note after it.
- **Acceptance check:** the regenerated notebook (a) contains no printed text stating or implying that missing values do not reach the model in the embedding path; (b) runs a cell on the default path that embeds a NaN-bearing copy of the sample and prints its similarity to the clean vector, with a markdown note explaining the difference; (c) still completes `Run all` unchanged.
- **Spec:** UX1, UX2, UX4, GDL7, GDL8.

### Minor

#### MEM-m1 — GUIDED layer is partial; nothing to compare

- **Cell/section:** opening, Sections 2, 6–7, final cell.
- **Observed issue:** no explicit intended learner or How-to-use section, roadmap, or Input → Model → Output contract. No question or prediction before extraction, no interpretation checkpoints with worked answers, no troubleshooting section (install restart, Hub download, upload, validation codes), no conclusion template. The default run produces a single vector, so the learner has nothing to compare and the "interpret" objectives reduce to reading printed strings. Section 2's ten carried-module cells (~125 k characters) are not labelled as infrastructure the learner may run without studying. If a comparison is added, note that raw cosine similarities are high even between unrelated inputs (P3/P4: 0.972 for `vibration` vs `temperature`, 0.993 for two unrelated BYOD series), so a comparison needs a contrast (e.g. clean vs gappy, or a centred/relative reading), not an absolute threshold.
- **Consequence:** self-paced learners get a reference walkthrough rather than a guided lesson.
- **Evidence:** source inspection; direct execution (similarity values).
- **Recommended correction:** add these elements in `tools/notebook_template.py` and the shared opening in `tools/build_notebook.py`. One Predict → Change one thing → Run → Explain activity can be the missingness experiment in MEM-M2.
- **Acceptance check:** the regenerated notebook has an audience/How-to-use block, a roadmap, an Input → Model → Output line, at least one Predict → Run → Explain prompt with a collapsible sample answer, a troubleshooting list covering install, download, upload and validation failures, an Infrastructure label on Section 2, and a conclusion template.
- **Spec:** GDL1–GDL5, GDL7, GDL9–GDL14, UX5.

#### MEM-m2 — The vector's invariance to level and scale is not stated

- **Cell/section:** Section 6 markdown (cell 30; `tools/notebook_template.py`).
- **Observed issue:** MOMENT normalises each channel with RevIN before encoding, so the embedding discards each channel's level and scale. Multiplying every value by 10 and adding 5 left the vector unchanged (cosine 0.99999999998, max abs diff 5.7e-6). The notebook describes shape, pooling and channel averaging but not this. The model card mentions RevIN only in passing.
- **Consequence:** a learner who uses the vectors for retrieval or clustering may expect them to separate, say, hot and cool days, or high and low loads, when the vector cannot see the difference.
- **Evidence:** direct execution (P3); source inspection (cell 30).
- **Recommended correction:** add one sentence to Section 6 stating that per-channel level and scale are normalised away (RevIN), so the vector represents shape, and that level/scale must be carried as separate features if they matter.
- **Acceptance check:** Section 6 states the level/scale invariance in learner-facing prose.
- **Spec:** UX2; §21.6 (unit represented — clarity).

#### MEM-m3 — Section 3 prints `device: None` and a hard-coded `source`; an upstream warning is unexplained

- **Cell/section:** cell 25, last line (`tools/build_notebook.py:548`).
- **Observed issue:** `LoadedMoment` has no `device` or `source` attribute, so `getattr(pipe, 'device', None)` prints `None` and `getattr(pipe, 'source', 'local-snapshot')` prints the fallback literal. The markdown promises that the effective identity, device and weight source are printed before inference; the real device (`pipe.identity.device` = `cpu`) appears only in Section 5. The same cell prints upstream's "Only reconstruction head is pre-trained. Classification and forecasting heads must be fine-tuned." The learner-facing prose never explains it; the explanation (the embedding path uses `nn.Identity`, so nothing untrained is involved) is only in the module docstring in Section 2.
- **Consequence:** the learner sees `device: None` where the device is said to be confirmed, and a warning that suggests the embeddings come from an untrained model.
- **Evidence:** direct execution (P2) and documented execution (hosted output identical).
- **Recommended correction:** print `pipe.identity.device`, `pipe.identity.dtype`, `pipe.identity.weight_file_loaded` and the snapshot directory from the generator's model-load cell; add one markdown sentence in Section 3 saying the warning is expected and refers to task heads, not the encoder.
- **Acceptance check:** Section 3 output shows a concrete device (`cpu`/`cuda`) and the verified weight file, contains no `None` or fallback literal, and the Section 3 markdown explains the head warning.
- **Spec:** MOD3, ENV3 (display), UX4.

#### MEM-m4 — Prerequisites misdescribe external access and download size

- **Cell/section:** Prerequisites (cell 1; `tools/build_notebook.py:441`, `tools/notebook_template.py:67`).
- **Observed issue:** "No GitHub access … required", yet `PINS` installs `momentfm @ git+https://github.com/…@38f7310…`, which pip clones from GitHub. The size note ("`torch==2.14.0` … largest download") omits that on the Kaggle CPU image the install pulled the CUDA build and NVIDIA wheels, several GB in total.
- **Consequence:** learners on restricted networks, or estimating time and disk, are misinformed.
- **Evidence:** source inspection; documented execution (pass 1 download log).
- **Recommended correction:** state that the install fetches `momentfm` from GitHub at a pinned commit, and give a measured download size for the supported runtime (or pin CPU wheels for the CPU path; see MEM-M1).
- **Acceptance check:** the Prerequisites name GitHub (upstream `momentfm`, pinned commit) as an install-time source and give a measured download figure with its environment.
- **Spec:** UX12, SRC3 (stale instruction).

#### MEM-m5 — Conformance and BYOD-interface hygiene

- **Cell/section:** metadata and opening; cells 3, 27, 31.
- **Observed issue:** the notebook declares spec 2.0 (current 2.2). `DIMER_NOTEBOOK_CI_PREINSTALLED` is read but never documented. The non-interactive BYOD location is only an environment variable (`DIMER_BYOD_PATH`), not a form field. On Jupyter, `USE_BYOD=True` fails with `ModuleNotFoundError: No module named 'google'`, although Jupyter is a stated runtime. The "sanity" L2 norm covers only window 0, so a multi-window BYOD run checks one vector of N.
- **Consequence:** executors and Jupyter users have no documented, field-based way to supply data; the sanity line under-reports on BYOD.
- **Evidence:** direct execution (P4); source inspection.
- **Recommended correction:** add `BYOD_PATH = ''  # @param {type:"string"}` (falling back to the environment variable), document both environment variables, say that the upload widget is Colab-only, print per-window norms (or min/max) instead of window 0, and regenerate against spec 2.2.
- **Acceptance check:** cell 27 reads a form-field path without importing `google.colab`; the opening documents both environment variables; the sanity line covers every window; metadata declares the current spec.
- **Spec:** EXE1, EXE2, EXE5.

### Suggestions

- **MEM-S1:** a three-shape toy retrieval (e.g. sine, trend, step) with nearest-neighbour ranks would give learners a concrete, falsifiable use of the vectors without claiming downstream quality.
- **MEM-S2:** archive the pass-1 error text in `docs/release-verification.md` beside the outcome, so a restart-dependent pass cannot be read as one-pass.

## 6. Readiness

**Needs revision.** Open Majors: MEM-M1 (no one-pass Run all; the applicable MUSTs RUN1, RUN10, ENV6, REL2 and REL11 fail on the recorded run) and MEM-M2 (missingness objective not delivered and mislabelled). Remaining gates after fixes: a one-pass hosted `Run all` of the regenerated blob recorded in `docs/release-verification.md`; a Colab upload-widget check of BYOD (not verified here).

## 7. What was verified vs inferred

- **Verified by direct execution (local Windows CPU, not Colab):** default path 17/17 with install skipped and snapshot pre-staged; values match the hosted run; missingness, rescaling and channel probes; BYOD end to end via `DIMER_BYOD_PATH`; six coded rejections; cancelled-upload and Jupyter `USE_BYOD` failures; `build_notebook.py --check` and `validate_release_assets.py` exit 0.
- **Verified from documented execution:** the hosted pass-1 restart failure and pass-2 success for this blob.
- **Inferred / not verified:** behaviour on a current Colab image (the Kaggle image is the only hosted evidence); the Colab upload widget; learner understanding (no learner observation).
- **Most likely to be wrong:** MEM-M2's severity. The limits section does say the run "does not prove that missing values were ignored by the encoder", so a careful reader may not be misled; if Kurt judges the prose sufficient, M2 drops to Minor (the label fix and the runnable comparison remain worthwhile).
