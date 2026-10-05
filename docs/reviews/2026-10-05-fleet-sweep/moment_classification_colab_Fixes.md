# moment_classification_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`4452676`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template*.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed (each recorded Kaggle run notes "1 restart after install cell"): Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. This repository's one commit-pinned source dependency (`momentfm`, no usable PyPI release) cannot be hash-locked as a wheel: the new lock is `requirements.lock.txt` minus that line, and the kernel cell builds `momentfm` afterwards from its pinned commit with `--no-deps` (its build tools come from PyPI unhashed; it needs `git`). The generator's `check_lock` accepts such a pin only if it is absent from the lock. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab.lock.txt`; `tools/validate_release_assets.py` (install markers, bootstrap check, kernel cell excluded from the library-use scan); `docs/release-verification.md` (the line describing that check); one lock shared by the four notebooks | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED mode with 1 of 9 guided markers. Added audience and Input → Model → Output table, How to use this notebook, roadmap, a Learner prerequisite, four Predict / Check your reasoning pairs (volunteer-level split; floor, 5-NN and probe; the unfreeze and selection; held-out comparison) quoting the recorded Kaggle T4 run of 2026-09-19 (floor 16.7 %, 5-NN 72.2 %, probe 72.2 % / macro-F1 0.715, selected 75.0 % / 0.741, validation log-loss 0.746 → 0.703 → 0.751 → 0.605), Troubleshooting, Glossary and a Conclusion template. Sections 1–3 labelled Infrastructure and collapsed. The literal `{{id, x, label}}` and `{{1,64}}` in the data-contract prerequisite (prerequisites are not formatted) now render with single braces. | Template opening, prerequisites, Sections 4–8, closing | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders` |
| SWP-A (quality asserts) | Fixed | Confirmed: Section 6 `assert frozen_test['accuracy'] > floor['accuracy'] and probe_adapter.policy == POLICY_FROZEN` and Section 8 `assert adapted_test['accuracy'] > floor['accuracy']` aborted a run (e.g. BYOD) before export and reload. The accuracy comparisons are now recorded verdicts (`frozen_policy_vs_floor`, `selected_policy_vs_floor`, plus the selected-vs-frozen delta) in the evaluation report; the policy identity check stays a hard `RuntimeError` (contract), as does reload parity. | Sections 6 and 8; validator markers | `test_swp_a_quality_comparisons_are_verdicts_not_asserts`, `test_swp_a_verdict_lines_report_a_failure_without_raising` |
| SWP-F (frozen re-run) | Fixed | Confirmed: `adapt` modifies the encoder inside `pipe` in place when the unfreeze wins, so re-running Section 6 (k-NN and probe labelled frozen) or Section 7 afterwards used the adapted encoder. Section 6 now snapshots the encoder state once, before any training, and Sections 6 and 7 restore it (with a digest check) at the start of every run. Section 9 already used a fresh `load_moment` instance for the frozen column. | Sections 6 and 7; validator marker | `test_swp_f_sections_6_and_7_restart_from_the_frozen_encoder` (executes the two cells' restore code with a stand-in encoder that an unfreeze moved in place) |
| SWP-B (BYOD) | Fixed | BYOD worked only through `files.upload()`, and an empty upload raised a bare `StopIteration`. Added a `BYOD_PATH` form field with the Colab upload as a guarded fallback; refusals name the path and the rule; `load_byod_dataset` / `validate_dataset` keep naming the record and rule. | Section 4 | `test_swp_b_byod_path_reads_file_and_refuses_with_names`, `test_swp_b_cancelled_colab_upload_gives_a_clear_message`, `test_swp_b_byod_path_fields_default_off` |

## User-visible changes

- Section 1 no longer installs into the notebook's Python and never asks for a restart; it builds (first run) or reuses `dimer_isolated_env_<lock digest>/` and every later code cell runs there. Linux x86_64 runtimes only; `momentfm` is built from its pinned commit (needs `git`, present on Colab and Kaggle).
- Guided material added; Sections 1–3 collapsed.
- Sections 6 and 8 print and record verdicts against the majority floor instead of stopping.
- Sections 6 and 7 restart from the frozen encoder on every run.
- New `BYOD_PATH` form field in Section 4.

## Verification (offline; not clean-runtime evidence)

- Real input: none of the model stages could run here (the Hugging Face Hub is unreachable and torch is not installed).
- Stand-in: the kernel-cell test runs the generated bootstrap against a pre-built environment folder whose `python` is the test interpreter (routing, reuse and idempotence are real; the managed CPython and locked packages are stand-ins). The preinstalled-executor path that CI's integration job uses (`DIMER_NOTEBOOK_CI_PREINSTALLED=1`, `scripts/run_notebook.py`) is checked to install nothing and route nothing.
- The lock was dry-run resolved with `uv pip install --dry-run --require-hashes --only-binary :all:` against a Python 3.12 environment (every entry has a wheel).
- All four `tools/build_notebook.py --check` invocations from CI: OK. `python tools/validate_release_assets.py`: PASS. `ruff check .`: clean.
- `pytest -m "not integration"` with lightweight dependencies (CI installs the full lock, including torch; here the six modules that import torch at collection were excluded both before and after): 145 passed / 22 failed before → 197 passed / 21 failed after. The 21 failures are the same environment-only failures before and after (tests that need torch or `uv export` from the CI toolchain); the one extra failure before was a worktree path artefact.
- Every code cell of the four regenerated notebooks parses.

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment; a second Run all reports `'reused': True`).
- The REL12 BYOD run with the BYOD gates and path fields set.
- A full Notebook Review Framework v1 review has not been done.
