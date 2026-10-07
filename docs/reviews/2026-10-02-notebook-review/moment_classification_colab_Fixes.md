# moment_classification_colab — review fixes (2026-10-02 review, fixed 2026-10-07)

Fixes for the Notebook Review Framework v1 findings (prefix `MCL`, review PR #18) on top of the fleet-sweep fixes of
2026-10-05 (`fe26480`). All changes are in the generator (`tools/build_notebook.py`,
`tools/notebook_template_classification.py`) and `tutorials/README.md`; the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending.**

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| MCL-M1 (Run all needs a restart) | Fixed (sweep `fe26480`) — hosted confirmation pending | Isolated runtime (sweep tests); the hosted one-pass run remains a gate. | Section 1 | sweep tests |
| MCL-M2 (re-runs train on an adapted encoder; frozen-policy artifact fails parity) | Fixed (sweep `fe26480` + this pass) — hosted confirmation pending | The sweep snapshots the pinned encoder once (Section 6) and restores it, digest-checked, at the start of Sections 6 and 7, so k-NN, the probe and every unfreeze start from the pinned model and a frozen-policy artifact is a head trained on the pinned encoder (parity on a fresh base holds by construction). This pass adds the explicit re-run instructions the review asked for (Section 7: change a field, re-run Section 6 then 7–9; BYOD: re-run from Section 4) and turns the optional experiments into Predict → change one field → re-run → observe → explain prompts. The acceptance check's exact reproduction of the k-NN/probe numbers after a `3e-5` re-run needs the real weights (hosted). | Sections 6, 7; closing | `test_mcl_m2_rerun_instructions_and_frozen_restore_before_both_policies`; sweep `test_swp_f_*` |
| MCL-M3 (interpretation fixes another run's numbers and an unsupported causal reading) | Fixed | Every build-record figure (77.8 % / 0.776, "two windows better", "partly recovering the static postures", the 1e-4 / 3e-5 sweep outcomes, the 150 s) is removed from the prose. Section 8 code converts the delta into windows (`comparison['delta_in_windows']`: `delta_windows`, `n_test`, `one_window_in_points`, a reading) and the markdown tells the learner to read it from their own output; the only quoted numbers are the recorded Kaggle T4 run of 2026-09-19, named as such in each checkpoint, with the delta given as one window; the posture-recovery claim is restated as a hypothesis with the test that would check it (several `SPLIT_SEED`s); the opening, Section 8 and the Interpretation name device-dependent variation (CPU vs T4 differed by one window on the same split). Conclusion template from the sweep. | Opening; Sections 6, 7, 8; Interpretation and limits; Glossary | `test_mcl_m3_no_build_record_result_in_the_prose`, `test_mcl_m3_delta_is_converted_to_windows_in_code` (the reading executed for the three cases), `test_mcl_m3_interpretation_reads_the_delta_in_windows_and_names_device_variation` |
| MCL-m1 (guided layer partial; carried code unmarked) | Fixed (sweep `fe26480`) | Audience, how-to-use, roadmap, glossary, predictions, checkpoints, troubleshooting, Infrastructure labels. Spec declaration stays 2.0 (maintainer decision; see MCL-m3). | — | sweep `test_swp_g_*` |
| MCL-m2 (BYOD Colab-only, no path field) | Fixed (sweep `fe26480`) | `BYOD_PATH` field read before any Colab import; the error off Colab names the field. | Section 4 | sweep `test_swp_b_*` |
| MCL-m3 (runtime estimates conflict; environment unnamed; default runtime has no hosted run) | Partly fixed | One set of measured figures with their environment everywhere (opening, Prerequisites, Sections 6–7, registry row): local CPU pre-flight of 2026-09-19 (k-NN 18.7 s, probe 76.8 s, unfreeze 307.5 s, path 430 s), the review's > 590 s on another CPU, and 294 s for everything on the recorded Kaggle T4 run; the "about five minutes", "150 s", "half a minute", "three minutes" and "~seven min" figures are gone. **Not fixed:** a clean hosted run on the runtime the notebook calls default (CPU) does not exist and cannot be produced here. | Opening; Prerequisites; Sections 6, 7; `tutorials/README.md` | `test_mcl_m3_timing_figures_name_their_environment` |
| MCL-m4 (escaped braces in Prerequisites) | Fixed (sweep `fe26480`) — does not reproduce on the sweep head | The regenerated notebook renders `{id, x, label}` and `[A-Za-z0-9_.:-]{1,64}`. | Prerequisites | `test_mcl_m4_prerequisites_render_single_braces` |
| MCL-m5 (`device: None`) | Fixed | Section 3 prints the real device, dtype, weight file and snapshot directory, and the prose explains the upstream head warning. | Section 3 (`tools/build_notebook.py`; template `model_section_note`) | `test_mcl_m5_section_3_prints_identity_and_explains_the_head_warning` |
| MCL-m6 ("No GitHub access" false) | Fixed | External access lists GitHub for the commit-pinned `momentfm` source and the measured size of the locked wheels. | Prerequisites | `test_mcl_m6_external_access_names_github_and_a_measured_download` |
| MCL-m7 (cosine "qualitative look" gives no guidance) | Fixed | Section 5 keeps the pair (labelled an anecdote) and adds a within- versus between-class mean-cosine table over the training split (`class_cosine_table`, with the gap per class and a note on how to read it); the prose says pooled vectors cluster near 1 and quotes the recorded run's inverted pair (0.984 vs 0.997). Costs one `features` pass over the training windows (about 20 s on CPU). | Section 5; Glossary | `test_mcl_m7_class_cosine_table_reads_the_gap` |
| MCL-S1 (normalisation claim as an activity), MCL-S2 (dispersion estimate) | Not done | `SPLIT_SEED` is now named in the optional experiments as the way to see dispersion. | — | — |

## User-visible changes

- Section 5 computes the training-split features once more (about 20 s on CPU) and prints a within/between-class cosine table.
- Section 8 prints `delta_in_windows` and writes it into the evaluation report's `comparison`.
- Section 3 prints the real device, dtype and weight file.
- Prose: no build-record numbers; all timings carry their environment; explicit re-run instructions.

## Verification (offline; not clean-runtime evidence)

- **Real input:** no model stage could run here (no torch, Hub and UCI unreachable from the test environment); the review's `run_probes.py` was not re-run. The frozen-restore mechanism of the sweep is covered by the sweep's own tests; the exact-reproduction half of the MCL-M2 acceptance check and the `3e-5` parity case need the hosted run.
- **Stand-in:** `tests/test_review_fixes_classification.py` executes `class_cosine_table` on synthetic unit vectors and the delta-in-windows reading for the three cases (probe kept, one window, four windows), and checks the prose statically.
- Commands: all four `--check` invocations OK; `validate_release_assets.py` PASS (markers updated); `ruff check .` clean; `pytest -m "not integration"` without the torch-importing modules: 197 passed / 21 failed → 242 passed / 21 failed (same environment-only failures).

## Remaining gates

- A hosted one-pass Run all of the regenerated blob (T4 or CPU) recorded in `docs/release-verification.md`; a CPU run is what MCL-m3 asks for.
- The MCL-M2 acceptance re-run: after Run all, `LEARNING_RATE = 3e-5`, re-run Section 6 then 7–9; the k-NN and frozen-policy numbers must repeat exactly and parity must pass.
- The REL12 BYOD run.
- Maintainer decisions: spec declaration 2.0 → 2.2; MCL-S1/S2.
