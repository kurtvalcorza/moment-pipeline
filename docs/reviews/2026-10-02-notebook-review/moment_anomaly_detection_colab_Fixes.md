# moment_anomaly_detection_colab — review fixes (2026-10-02 review, fixed 2026-10-07)

Fixes for the Notebook Review Framework v1 findings (prefix `MAD`, review PR #17) on top of the fleet-sweep fixes of
2026-10-05 (`fe26480`). All changes are in the generator (`tools/build_notebook.py`,
`tools/notebook_template_anomaly_detection.py`) and `src/moment_pipeline/roles.py`; the notebook is regenerated.
STATUS and the release labels are unchanged. **Readiness: Verification pending.**

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| MAD-M1 (Run all needs a restart) | Fixed (sweep `fe26480`) — hosted confirmation pending | Isolated runtime (sweep tests `test_swp_r_*`); the hosted one-pass run remains a gate. | Section 1 | sweep tests |
| MAD-M2 (residual scale taught as universal; hard-coded channel; BYOD channels dropped) | Fixed — hosted confirmation pending | Section 6 prints a per-channel table (scored points, median, max) and says raw residuals are comparable only within one channel of one series; the checkpoint explains temperature's larger scale with the recorded numbers (mean 1.62 / max 3.77 vs 0.30; 42 clean points above the lowest spike). Section 7: `SCORE_CHANNEL` form field (empty = the labelled channel, `vibration`, else the first channel; an unknown name is refused with the list), the choice and its reason printed **before** the recall, the channels not ranked for the metric named, the **pooled all-channel ranking** of the injected points printed (ranks and pooled top-k recall, and which channels fill the pooled top k), and the three highest residuals of **every** channel listed (BYOD included). `report["ranked_channel"]` and `report["per_channel_residuals"]` record it. | Sections 6, 7; opening | `test_mad_m2_per_channel_residual_statistics_are_printed_and_explained`, `test_mad_m2_channel_choice_is_stated_before_the_recall_and_pooled_ranking_is_shown`, `test_mad_m2_score_channel_field_overrides_and_rejects_unknown`, `test_mad_m2_byod_without_labels_lists_every_channel_and_names_the_omitted_ones` (Sections 6–8 executed on a stand-in score table of the sample, one and two series) |
| MAD-m1 (guided layer partial) | Fixed | Guided layer from the sweep; the Section 6 Predict now asks which channel will have the larger typical residual; a Predict → Change one thing → Run → Explain activity (the aggregation) after Section 7. Spec declaration stays 2.0 (see MAD-m7). | Sections 6, 7 | sweep `test_swp_g_*`; `test_mad_m2_experiment_is_the_aggregation_with_rerun_instructions` |
| MAD-m2 (`loss="mse"` cannot change the ranking) | Fixed | The suggested experiment is now `channel_aggregation="mean"` then `"max"` (recall 0.667 / 0.333 in the review), with the cells to re-run (Section 6, then 7–8) and a prediction prompt; the prose states that MSE only rescales the per-channel scores while channels are not aggregated. | Section 7 checkpoint; closing | `test_mad_m2_experiment_is_the_aggregation_with_rerun_instructions` |
| MAD-m3 (no naive baseline) | Fixed | A \|z-score\| of the raw values of the ranked channel is ranked with the same k through `top_k_recall`, printed with its injected ranks, carried in the evaluation report under `baselines` (id `abs_zscore`, via the new `baseline`/`baseline_id` support for anomaly results in `evaluation_report`), and read in a `VERDICT` line (the model does / does not separate from a one-line statistic on this sample). | Section 7; `src/moment_pipeline/roles.py` | `test_mad_m3_abs_zscore_baseline_is_computed_on_the_same_channel_and_k` |
| MAD-m4 (`device: None`) | Fixed | Section 3 prints the real device, dtype, weight file and snapshot directory. | Section 3 (`tools/build_notebook.py`) | `test_mad_m4_section_3_prints_the_real_device_and_weight_file` |
| MAD-m5 (prerequisites misdescribe access and size) | Fixed | External access names the Hub, PyPI with the measured ~3.1 GB of locked wheels (PyPI release metadata, 2026-10-07; CUDA torch build) and GitHub for `momentfm`; Compute bullet corrected. | Prerequisites | `test_mad_m5_external_access_names_github_and_a_measured_download` |
| MAD-m6 (plot hard to read; merges series) | Fixed | One SVG per series (up to four; the rest named) of the ranked channel: raw score over time with a labelled axis (three ticks), the raw value beneath, the first/last timestamps, and the labelled spikes circled on the sample. | Section 8 | `test_mad_m6_plot_is_per_series_with_axis_labels_and_spike_markers` |
| MAD-m7 (conformance hygiene) | Partly fixed | `BYOD_PATH` field (sweep) read before any Colab import; the opening documents both executor variables. **Not changed:** spec declaration stays 2.0 (maintainer decision). | Opening; Section 4 | `test_mad_m7_opening_documents_executor_variables_and_byod_field` |
| MAD-S1 (difficult normal event in the sample) | Not done | Needs a new pinned sample and hosted re-qualification. | — | — |
| MAD-S2 (archive pass-1 error text) | Not done | Maintainer's record. | — | — |
| MAD-S3 (why temperature reconstructs worse) | Partly | One-line reason in the Section 6 checkpoint (amplitude and period), not backed by a check. | Section 6 | — |

## User-visible changes

- New `SCORE_CHANNEL` form field in Section 7 (empty = previous behaviour).
- Section 6 prints a per-channel residual table; Section 7 prints the channel choice, the pooled ranking, the |z-score| baseline, a `VERDICT` line and every channel's top residuals; the evaluation report gains a baseline entry, `ranked_channel` and `per_channel_residuals`; `outputs/moment_anomaly_detection_result.json` gains `score_channel_reason`, `per_channel_residuals` and `zscore_baseline`.
- The plot is one SVG per series (`outputs/moment_anomaly_scores_<series>.svg`); `outputs/moment_anomaly_scores.svg` is no longer written.
- Section 3 prints the real device, dtype and weight file.
- `evaluation_report(AnomalyResult, labels, baseline=..., baseline_id=...)` now carries a naive baseline (default unchanged).

## Verification (offline; not clean-runtime evidence)

- **Real input:** no model stage could run here; the review's `run_probes.py` was not re-run. Numbers quoted in the prose (per-channel means, pooled ranks, aggregation recalls) are the recorded Kaggle CPU run of 2026-09-14 and the review's measured values, named as such.
- **Stand-in:** `tests/test_review_fixes_anomaly_detection.py` executes Sections 6–8 on a stand-in score table built from the sample's formulas (large smooth residual on the clean channel, spikes on the labelled one), with the package's `top_k_recall` semantics restated in the test; the channel choice, the pooled ranking, the baseline, the report and the per-series plot are exercised for one and two series, with and without labels.
- Commands: all four `--check` invocations OK; `validate_release_assets.py` PASS (markers updated); `ruff check .` clean; `pytest -m "not integration"` without the torch-importing modules: 197 passed / 21 failed → 242 passed / 21 failed (same environment-only failures).

## Remaining gates

- A hosted one-pass Run all of the regenerated blob recorded in `docs/release-verification.md`, with the printed per-channel table, pooled ranks and baseline recall checked against the checkpoint text.
- The REL12 BYOD run (two-channel CSV: every channel's top residuals listed; per-series plots).
- Maintainer decisions: spec declaration 2.0 → 2.2; MAD-S1/S2/S3.
