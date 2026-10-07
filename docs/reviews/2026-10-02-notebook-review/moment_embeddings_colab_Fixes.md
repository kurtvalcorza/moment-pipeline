# moment_embeddings_colab — review fixes (2026-10-02 review, fixed 2026-10-07)

Fixes for the Notebook Review Framework v1 findings (prefix `MEM`, review PR #19) on top of the fleet-sweep fixes of
2026-10-05 (`fe26480`). All changes are in the generator (`tools/build_notebook.py`, `tools/notebook_template.py`) and
`src/moment_pipeline/roles.py`; the notebook is regenerated. STATUS and the release labels are unchanged.
**Readiness: Verification pending.**

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| MEM-M1 (Run all needs a restart) | Fixed (sweep `fe26480`) — hosted confirmation pending | Isolated runtime; no `pip install` into the kernel (sweep tests `test_swp_r_*`). Hosted one-pass run and the record row remain a gate. | Section 1 | sweep tests |
| MEM-M2 (missingness objective never exercised; labels contradict the prose) | Fixed — hosted confirmation pending | (a) The printed line is now `per-point missingness mask passed to model: False (pre-filled values ARE seen as data)`, followed by the package's `MISSINGNESS_POLICY`; `INPUT_SCHEMA["values"]` in `roles.py` says the NaN never reaches the model but that the embedding path replaces it by `prefill_value`, which the encoder sees as data; Section 6 prose and checkpoint rewritten. (b) New Section 6b on the default path: a `MISSING_STRIDE` field (12.5 % missing by default), a NaN-bearing copy and an interpolated copy embedded and compared with the clean vector (cosine and L2), plus the zero-written copy, which must equal the missing one; Predict prompt before, "What to notice" after (quoting the review's measured 0.80 / 5.1 vs 0.9996 / 0.20). (c) Run all unchanged otherwise. | Sections 6, 6b; `src/moment_pipeline/roles.py`; `tools/validate_release_assets.py` markers | `test_mem_m2_no_printed_text_implies_missing_values_are_ignored`, `test_mem_m2_experiment_embeds_a_gappy_copy_and_prints_its_similarity` (Section 6b executed with a stand-in `embed`), `test_mem_m2_experiment_refuses_a_degenerate_stride`, `test_mem_m2_experiment_has_a_prediction_and_a_what_to_notice_note` |
| MEM-m1 (guided layer partial; nothing to compare) | Fixed | Guided layer from the sweep; Section 6b is the Predict → Change one thing → Run → Explain activity and gives the learner a contrast to read. Spec declaration stays 2.0 (see MEM-m5). | — | sweep `test_swp_g_*`; MEM-M2 tests |
| MEM-m2 (level/scale invariance not stated) | Fixed | Section 6 states the RevIN invariance in learner-facing prose (×10 + 5 leaves the vector unchanged) and that level/scale must be carried as a separate feature; Glossary entry; Next experiments. | Section 6; closing | `test_mem_m2_section_6_states_the_revin_invariance` |
| MEM-m3 (`device: None`; head warning unexplained) | Fixed | Section 3 prints `pipe.identity.device`, `dtype`, the verified weight file and the snapshot directory; the Section 3 prose says the upstream head warning is expected and refers to task heads (the embedding instance uses an identity head). | Section 3 (`tools/build_notebook.py`; template `model_section_note`) | `test_mem_m3_section_3_prints_identity_and_explains_the_head_warning` |
| MEM-m4 (prerequisites misdescribe access and size) | Fixed | External access names the Hub, PyPI with the measured ~3.1 GB of locked wheels (PyPI release metadata, 2026-10-07; CUDA torch build) and GitHub for `momentfm`; Compute bullet corrected. | Prerequisites | `test_mem_m4_external_access_names_github_and_a_measured_download` |
| MEM-m5 (conformance and BYOD hygiene) | Partly fixed | `BYOD_PATH` field (sweep) read before any Colab import; the opening documents both executor variables; the L2 norm line now covers every window (min/max, all finite). **Not changed:** spec declaration stays 2.0 (maintainer decision). | Opening; Section 4; Section 6 | `test_mem_m5_norm_line_covers_every_window_and_opening_documents_executor_variables` |
| MEM-S1 (toy retrieval) | Not done | — | — | — |
| MEM-S2 (archive pass-1 error text) | Not done | Maintainer's record. | — | — |

## User-visible changes

- New Section 6b (`MISSING_STRIDE` form field, default 8) with three extra `embed` calls on the default path.
- Section 6 prints the mask flag under its new label, the missingness policy, the source missing fraction and per-window norm bounds (the old single-window norm line is gone).
- Section 3 prints the real device, dtype and weight file.
- `INPUT_SCHEMA["values"]` text in the package's input manifest changed (same meaning, states the per-path behaviour).

## Verification (offline; not clean-runtime evidence)

- **Real input:** no model stage could run here (no torch, Hub unreachable); the review's `run_probes.py` was not re-run. The experiment's numbers quoted in "What to notice" are the review's measured values on the real weights (local CPU), named as such.
- **Stand-in:** `tests/test_review_fixes_embeddings.py` executes Section 6b with a stand-in `embed` (a fixed random projection of the pre-filled window values) and checks the gap construction (32 of 256 points per channel), interpolation, the zero-written identity, the ordering of the two distances and the printed reading.
- Commands: all four `--check` invocations OK; `validate_release_assets.py` PASS; `ruff check .` clean; `pytest -m "not integration"` without the torch-importing modules: 197 passed / 21 failed → 242 passed / 21 failed (same environment-only failures).

## Remaining gates

- A hosted one-pass Run all of the regenerated blob recorded in `docs/release-verification.md`; the printed Section 6b figures compared with the "What to notice" numbers.
- The REL12 BYOD run.
- Maintainer decisions: spec declaration 2.0 → 2.2; MEM-S1/S2.
