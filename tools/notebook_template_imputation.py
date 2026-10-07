"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

This is the imputation tutorial of moment-pipeline; the repository ships three TASK-INFERENCE notebooks
(embeddings, imputation, anomaly scoring) generated from three templates that share the same carried
package (ten modules under src/moment_pipeline/, in dependency order) and the same model cell.

Generator /2 keys in use: ``modules`` lists every module except ``__init__.py``; ``entry_module`` is
``model.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/``MODEL_LICENSE``/``MODEL_KEY``); ``model_load`` is
the package's documented loader ``load_moment(task="reconstruction", weights_dir=WEIGHTS_DIR)`` — MOMENT
replaces its head per task (RFC M-8), so the task is part of the loaded model's identity.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    'package': 'moment_pipeline',
    'repo_name': 'moment-pipeline',
    'stem': 'moment_imputation',
    'notebook_name': 'moment_imputation_colab.ipynb',
    'profile': 'TASK-INFERENCE',
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    # MIM-m4: the measured size of the locked wheels (sum of the locked manylinux x86_64 wheel sizes from PyPI
    # release metadata, read on 2026-10-07; the lock's torch 2.14.0 is the CUDA build and carries the NVIDIA libraries).
    "lock_download": {"bytes": 3078108227, "measured": "sum of the locked manylinux x86_64 wheel sizes from PyPI release metadata, 2026-10-07; the lock's `torch==2.14.0` is the CUDA build, so the CPU path downloads its NVIDIA libraries too"},
    'pipeline_class': 'LoadedMoment',
    'weights_key': 'moment-1-base',
    'modules': ['anomaly.py', 'canonical.py', 'config.py', 'csvio.py', 'embedding.py', 'imputation.py', 'model.py', 'provenance.py', 'roles.py', 'validation.py'],
    'entry_module': 'model.py',
    'model_load': 'load_moment(task="reconstruction", weights_dir=WEIGHTS_DIR)',
    'runtime_imports': ['torch', 'transformers', 'pandas', 'numpy'],
    'title': 'MOMENT imputation — DIMER task-inference tutorial (standalone)',
    'badges': [
        (
            'GitHub',
            'https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/kurtvalcorza/moment-pipeline',
        ),
        (
            'Open In Colab',
            'https://colab.research.google.com/assets/colab-badge.svg',
            'https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_imputation_colab.ipynb',
        ),
        (
            'Hugging Face',
            'https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat',
            'https://huggingface.co/AutonLab/MOMENT-1-base',
        ),
        (
            'Upstream',
            'https://img.shields.io/badge/Upstream-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/moment-timeseries-foundation-model/moment',
        ),
        (
            'arXiv',
            'https://img.shields.io/badge/arXiv-2402.03885-b31b1b.svg',
            'https://arxiv.org/abs/2402.03885',
        ),
    ],
    'capability': (
        'pretrained reconstruction-backed time-series imputation (patch-granular artificial masking) with the pinned `AutonLab/MOMENT-1-base` checkpoint'
    ),
    'intro': (
        "This notebook uses MOMENT's pretrained reconstruction path for patch-granular artificial masking through the repository's public `moment_pipeline` API, carried in this notebook. It withholds known source values, evaluates only deliberately hidden truth, preserves observed values in the exported imputed series, and writes machine-readable metrics and provenance. **No gradient training, fine-tuning, in-context conditioning, or fitted preprocessing occurs** — **no adaptation occurs.** **Upstream vs. this repository.** Upstream MOMENT supplies the pretrained reconstruction model. This repository supplies immutable pinning/integrity verification, long-format validation and canonicalization, patch-quantized masking semantics, masked-point evaluation, an observed-value-preserving imputed product, and provenance/export contracts. The displayed MAE/RMSE values are sample/tutorial evidence only."
    ),
    "guided": {"opening": [(
        '**Who this notebook is for.** A learner who knows basic Python and pandas, has used Colab or Jupyter and has met time-series data, and wants to see how a pretrained time-series foundation model is used without any training to fill hidden values — what goes in, what comes out, and what the output does and does not prove. The audience is students and practitioners preparing their own sensor or monitoring series; no prior experience with MOMENT is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | a long-format CSV: the synthetic two-channel series `A` (256 steps) or your own, plus an explicit visibility mask that hides one stretch of known values |\n| Model | the MOMENT-1-base reconstruction head (`task="reconstruction"`), with masking applied per 8-step patch |\n| Output | an imputed series in which only missing or deliberately hidden cells are replaced; MAE and RMSE on the hidden cells, per channel and pooled, beside a same-support baseline (linear interpolation across an interior gap, a last-value hold for a trailing one); a printed **model-vs-baseline verdict** and a visible-point fidelity check; and a `sample-sanity` report |\n\n**How to use this notebook.** Choose a runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook\'s own Python, so no restart is needed (the recorded hosted runs of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the default path. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer. The worked answers quote only numbers from recorded runs (the Kaggle CPU run of 14 September 2026 and the review of 2 October 2026, both named where used) and otherwise state what the code and the sample guarantee; compare them with what your run prints. One warning up front, so that the printed metrics are read correctly: **on every recorded run of this sample the pinned MOMENT-1-base checkpoint reconstructed the hidden patch worse than the naive baseline** — Section 7 prints that comparison as a verdict, and the Interpretation section explains what it does and does not mean. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the synthetic sample (or your CSV) → 5 validation, canonicalization and the evaluation mask, with a `HOLDOUT_START` field *(core concept: windows, padding, masks)* → 6 impute *(core concept: patch-quantized masking)* → 7 masked-point metrics, a same-support baseline and the verdict *(evaluation practice)* → 8 the plot around the hidden patch → 9 → export and provenance *(engineering)* → conclude.'
    )]},
    'learning_objectives': (
        'install the pinned runtime; read what the carried package guarantees; resolve and digest-verify the immutable model revision; generate the deterministic synthetic sample or bring your own long-format CSV; validate and canonicalize the input into an input manifest; hide one complete 8-step patch with known truth, choosing where it goes; impute and evaluate only withheld truth through an evaluation report; compare a same-support naive baseline and read the printed verdict, including when the model loses; and export the imputed series, metrics, and provenance. By the end of this notebook you will be able to do each of these without the repository being reachable.'
    ),
    'exclusions': (
        'forecasting, classification, stochastic uncertainty intervals, or production fitness. This path returns point reconstructions only; no uncertainty interval is provided.'
    ),
    'prerequisites': [
        '- **Learner:** basic Python and pandas familiarity; no prior experience with MOMENT or time-series foundation models. Windows, patches, padding, masks and the evaluation verdicts are explained where they are first used and again in the Glossary.',
        '- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (plus `momentfm`, built from its pinned commit), so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported.',
        '- **Compute:** CPU is the default path and no GPU is required; CUDA is used automatically when available. The public v1 API accepts `float32` only. The locked `torch==2.14.0` wheel (the CUDA build, ~555 MB) and its NVIDIA libraries are the largest downloads of the run — see the measured total under External access — followed by the ~454 MB `model.safetensors`.',
        '- **Knowledge:** basic Python and pandas; what a long-format time-series table is.',
        "- **Data:** the default sample is the repository's deterministic clean two-channel synthetic series regenerated in code, so nothing is downloaded and no private data is needed. The artificial evaluation mask is deterministic, so no random seed is required. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV with columns `series_id`, `timestamp`, `channel`, `value`; `value` numeric or missing; identifiers and timestamps valid. Duplicate or ambiguous column names are rejected from the raw CSV header before dataframe parsing, and duplicate `(series_id, channel, timestamp)` rows are rejected by production validation. Operational ceilings: at most 5,000,000 rows, 1,024 series, 32 channels, and 1,024 canonical windows; MOMENT uses 512-step windows and 8-step non-overlapping patches; short series are left-padded, long series keep the final 512 timestamps, irregular spacing is surfaced rather than silently interpolated. BYOD is read locally in the notebook runtime and is not sent to an external inference service. Do not upload confidential or restricted data (personal or otherwise sensitive data included) to a hosted notebook environment unless you are authorized to do so.",
    ],
    "cells": [
        {
            "md": (
                '## 4. Generate the synthetic sample or optional BYOD\n'
                '\n'
                "The default sample is **synthetic**: the repository's `examples/sample-data/generate_samples.py` formulas (one series `A`, channels `vibration` and `temperature`, 256 steps at 15 minutes — a trend plus two sinusoids) regenerated in code and rendered to the same canonical CSV bytes, so the `moment_clean.csv` SHA-256 is asserted against the digest the repository checks in. It is deterministic teaching data, not benchmark evidence. The optional upload path first validates the **raw CSV header** with `read_long_csv_bytes()` (duplicate or ambiguous names cannot be silently renamed by pandas); the resulting frame still goes through the same production validation/canonicalization path in the next stage. Set `USE_BYOD=True` and `BYOD_PATH` to a CSV already in the runtime (Colab, Kaggle or Jupyter); on Colab an empty `BYOD_PATH` opens an upload dialog, and a cancelled upload stops with a message. `DIMER_BYOD_PATH` remains the automation hook. Successful completion means one identified input frame is available and its source/digest are recorded; look for the sample identity (kind, name, digest) and the first rows."
            ),
            "code": (
                'import hashlib\n'
                'import io\n'
                'import json\n'
                'from pathlib import Path\n'
                '\n'
                'import numpy as np\n'
                'import pandas as pd\n'
                '\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = ""  # @param {{type:"string"}}\n'
                '# The form path is used when USE_BYOD is on; DIMER_BYOD_PATH remains the automation hook.\n'
                'BYOD_PATH = (BYOD_PATH.strip() if USE_BYOD else "") or os.environ.get("DIMER_BYOD_PATH", "")\n'
                'SAMPLE_SHA256 = "34fc475828d2f62108b340efd255501a898577be11286c034da2e0f766ee963c"  # examples/sample-data/SHA256SUMS\n'
                '\n'
                '\n'
                'def build_samples() -> dict:\n'
                "    # The repository's examples/sample-data/generate_samples.py formulas; no random state.\n"
                '    n = 256\n'
                '    timestamps = pd.date_range("2026-01-01", periods=n, freq="15min")\n'
                '    step = np.arange(n, dtype=float)\n'
                '    rows = []\n'
                '    for channel, base, amplitude, period, phase, trend in (("vibration", 1.5, 0.35, 32.0, 0.0, 0.0008), ("temperature", 28.0, 2.5, 96.0, 9.0, 0.0015)):\n'
                '        values = base + trend * step + amplitude * np.sin(2.0 * np.pi * (step + phase) / period) + 0.08 * np.cos(2.0 * np.pi * step / 16.0)\n'
                '        rows.extend(("A", stamp, channel, float(value)) for stamp, value in zip(timestamps, values, strict=True))\n'
                '    clean = pd.DataFrame(rows, columns=["series_id", "timestamp", "channel", "value"])\n'
                '    anomaly = clean.copy()\n'
                '    labels = pd.DataFrame({{"series_id": "A", "timestamp": timestamps, "is_injected_anomaly": False}})\n'
                '    for index, delta in {{184: 2.8, 201: -3.2, 233: 3.6}}.items():\n'
                '        labels.loc[index, "is_injected_anomaly"] = True\n'
                '        selector = (anomaly["channel"] == "vibration") & (anomaly["timestamp"] == timestamps[index])\n'
                '        anomaly.loc[selector, "value"] = anomaly.loc[selector, "value"] + delta\n'
                '    return {{"moment_clean.csv": clean, "moment_anomaly.csv": anomaly, "moment_anomaly_labels.csv": labels}}\n'
                '\n'
                '\n'
                'def sample_csv_bytes(frame: pd.DataFrame) -> bytes:\n'
                '    return frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.6f", lineterminator="\\n").encode("utf-8")\n'
                '\n'
                '\n'
                'if BYOD_PATH:\n'
                '    if not Path(BYOD_PATH).expanduser().is_file():\n'
                '        raise FileNotFoundError(f"BYOD_PATH {{BYOD_PATH!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one long-format CSV.")\n'
                '    payload = Path(BYOD_PATH).expanduser().read_bytes()\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    labels = None\n'
                '    sample_identity = {{"kind": "byod", "name": Path(BYOD_PATH).name, "sha256": hashlib.sha256(payload).hexdigest()}}\n'
                '    sample_kind = "BYOD"\n'
                'elif USE_BYOD:\n'
                '    try:\n'
                '        from google.colab import files\n'
                '    except ImportError:\n'
                '        raise RuntimeError("USE_BYOD is on but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: copy the CSV into this runtime (or attach it as a Kaggle dataset) and set BYOD_PATH.") from None\n'
                '    uploaded = files.upload()\n'
                '    if len(uploaded) != 1:\n'
                '        raise ValueError(f"Upload exactly one CSV with columns series_id,timestamp,channel,value (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.")\n'
                '    name, payload = next(iter(uploaded.items()))\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    labels = None\n'
                '    sample_identity = {{"kind": "byod", "name": name, "sha256": hashlib.sha256(payload).hexdigest()}}\n'
                '    sample_kind = "BYOD"\n'
                'else:\n'
                '    samples = build_samples()\n'
                '    payload = sample_csv_bytes(samples["moment_clean.csv"])\n'
                '    observed = hashlib.sha256(payload).hexdigest()\n'
                '    if observed != SAMPLE_SHA256:\n'
                '        raise ValueError(f"Synthetic sample digest mismatch: {{observed}} != {{SAMPLE_SHA256}}")\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    sample_identity = {{"kind": "synthetic", "name": "moment_clean.csv", "sha256": observed}}\n'
                '    sample_kind = "synthetic"\n'
                '\n'
                'print({{"sample_kind": sample_kind, **sample_identity, "rows": len(frame), "columns": list(frame.columns)}})\n'
                'print(frame.head())'
            ),
        },
        {
            "md": (
                '## 5. Validate and canonicalize → input manifest\n'
                '\n'
                "Before the model runs, the cell prints the effective runtime and the operational ceilings — the `ResourceLimits` (rows, series, channels, windows), the fixed 512-step window and 8-step patch — and the device/precision policy. `validate_inputs` is the package's public validation stage: it runs exactly the two calls every task path makes (`validate_long_frame`, then `to_windows`), so it raises exactly what canonicalization would raise, and returns an **input manifest** naming the schema and ceilings, each window's series, valid positions, padding and truncation, the source-missingness fractions, and the verdict; it is written to `outputs/moment_imputation_input_manifest.json`. To show what rejection looks like, the cell also validates a deliberately broken copy (a channel with no observed value) and records the pipeline's own error code as a finding. The canonical `WindowSet` used by the model is built by the same calls; any padding, truncation, source missingness, or irregular frequency is disclosed here before model execution. Successful output means the input passed validation and every canonicalization effect is disclosed before the model runs. The cell then chooses one complete 8-step patch of the **first canonical window** for which every channel has source-observed truth and hides it — the **masking unit is the 8-step patch**, and this separates **source-missing data from artificial evaluation masking**: the artificial holdout consists only of genuine known truth. Only that first window is held out and scored; on a multi-series CSV the other windows are imputed but carry no artificial mask. `HOLDOUT_START = -1` picks the **middle interior patch** — one with source-observed points on both sides, so the task is gap filling and the baseline in Section 7 can interpolate across it; set it to any printed candidate start to move the gap (a trailing patch, with no observed point after it, turns the task into an extrapolation, and the cell says so)."
                '\n\n**Predict before running:** the cell hides one stretch of known values for evaluation. How many points will be held out per channel, and why that number? And why should the hidden patch have observed neighbours on both sides?'
            ),
            "code": (
                'import os\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                'config = MomentConfig(task="reconstruction")\n'
                'limits = config.limits\n'
                'print({{"python": platform.python_version(), "torch": torch.__version__, "device": pipe.identity.device, "dtype": pipe.identity.dtype}})\n'
                'print({{"ceilings": {{"max_rows": limits.max_rows, "max_series": limits.max_series, "max_channels": limits.max_channels, "max_windows": limits.max_windows, "sequence_length": config.sequence_length, "patch_length": config.patch_length}}}})\n'
                '\n'
                'report, normalized = validate_long_frame(frame, config)\n'
                'windows = to_windows(normalized, config, report=report, frame=normalized)\n'
                'input_manifest = validate_inputs(frame, config, names=[str(w) for w in windows.window_ids])\n'
                '# Demonstrate rejection on an input that breaks a rule; the finding is recorded, not swallowed.\n'
                'broken = frame.copy()\n'
                'broken.loc[broken["channel"] == broken["channel"].iloc[0], "value"] = np.nan\n'
                'try:\n'
                '    validate_inputs(broken, config)\n'
                'except ValidationError as exc:\n'
                '    input_manifest["findings"].append({{"input": "empty-channel-probe", "verdict": "rejected", "code": exc.code, "message": str(exc)}})\n'
                'with open("outputs/moment_imputation_input_manifest.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(input_manifest, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(input_manifest, indent=2, default=str))\n'
                'print("window tensor:", windows.x_enc.shape)\n'
                'print("padded windows:", int(sum(windows.padded)), "/", windows.n_windows)\n'
                'print("truncated windows:", int(sum(windows.truncated)), "/", windows.n_windows)\n'
                'print("source missing fraction:", windows.masked_point_fraction)\n'
                'if any(windows.truncated):\n'
                '    print("WARNING: long input series were truncated to their final 512 timestamps.")\n'
                'if any(windows.padded):\n'
                '    print("NOTE: short input series were left-padded; padding is excluded by the input mask.")\n'
                '\n'
                'HOLDOUT_START = -1  # @param {{type:"integer"}}\n'
                '# -1 = the middle interior patch (observed neighbours on both sides); otherwise one of the candidate starts printed below.\n'
                '\n'
                '\n'
                'def choose_holdout(observed, patch_length, requested=-1):\n'
                '    """Return (start, complete_starts, interior_starts) for a 1-D boolean array of fully observed positions."""\n'
                '    n = len(observed)\n'
                '    complete = [c for c in range(0, n, patch_length) if bool(np.all(observed[c:c + patch_length]))]\n'
                '    if not complete:\n'
                '        raise ValueError(f"The first canonical window has no complete {{patch_length}}-step source-observed patch to hold out. Provide a series with at least {{patch_length}} consecutive observed timestamps.")\n'
                '    interior = [c for c in complete if c > 0 and c + patch_length < n and bool(observed[c - 1]) and bool(observed[c + patch_length])]\n'
                '    if requested < 0:\n'
                '        chosen = interior[len(interior) // 2] if interior else complete[-1]\n'
                '    elif requested in complete:\n'
                '        chosen = int(requested)\n'
                '    else:\n'
                '        raise ValueError(f"HOLDOUT_START={{requested}} is not a complete source-observed {{patch_length}}-step patch start of this window; choose -1 (automatic) or one of {{complete}}.")\n'
                '    return chosen, complete, interior\n'
                '\n'
                '\n'
                'visible = np.ones_like(windows.input_mask, dtype=np.float32)\n'
                'observed = (windows.input_mask[0] == 1) & np.all(windows.point_mask[0] == 1, axis=0)\n'
                'start, complete_patch_starts, interior_patch_starts = choose_holdout(observed, windows.patch_length, HOLDOUT_START)\n'
                'stop = start + windows.patch_length\n'
                'visible[0, start:stop] = 0.0\n'
                'holdout_interior = start in interior_patch_starts\n'
                'holdout_placement = "interior (source-observed points on both sides: a gap)" if holdout_interior else "trailing or leading (no source-observed point on one side: an extrapolation, not a gap)"\n'
                'print("artificial holdout window:", windows.window_ids[0], "(only the first canonical window is held out and scored)")\n'
                'print("candidate patch starts (complete, source-observed):", complete_patch_starts)\n'
                'print("interior candidates:", interior_patch_starts)\n'
                'print("deliberately hidden positions:", start, "through", stop - 1, "->", holdout_placement)\n'
                'print("source-observed points after the holdout:", int(observed[stop:].sum()), "| before it:", int(observed[:start].sum()))\n'
                'print("held-out point count across channels:", windows.n_channels * windows.patch_length)'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>One 8-step patch per channel: MOMENT masks whole patches, so the evaluation holdout is one patch length (8 steps) per channel — the cell prints the held-out point count as channels × patch length, 2 × 8 = 16 on the two-channel sample. Hiding less than a patch is impossible for this model; hiding a single point would hide its whole patch. Observed neighbours on both sides make the task a **gap** — the setting imputation is for — and give the baseline something to interpolate between; the previous version of this notebook hid the series\' final patch (positions 504–511 on the sample), which is a two-hour extrapolation, and its "interpolation" baseline could only hold the last observed value.</details>'
            ),
        },
        {
            "md": (
                '## 6. Impute\n'
                '\n'
                '`impute()` receives the explicit visibility mask and runs the pinned reconstruction path (loaded in Section 3 with `task="reconstruction"`); the requested mask is combined with source missingness and **patch-quantized** for MOMENT, and the result records how the requested masking mapped to the effective model masking (`masked_point_fraction` for the source, `model_masked_point_fraction` and `masked_patch_fraction` for what the model actually hid). In the exported imputed product, `imputed_value` replaces only source-missing or deliberately hidden cells — **observed values are preserved**, and observed neighbours hidden from the model only because they share a patch are not overwritten. Successful output means the verified reconstruction path executed on this validated input.'
                '\n\n**Predict before running:** you asked to hide some timestamps. Will the model hide exactly those, or more? Will any value you supplied change in the exported product?'
            ),
            "code": (
                'result = impute(windows, pipe, mask=visible, warmup=False)\n'
                'provenance = build_provenance(pipe, windows, result)\n'
                'print("effective model:", pipe.identity.name)\n'
                'print("effective revision:", pipe.identity.revision)\n'
                'print("verified weight file:", pipe.identity.weight_file_loaded)\n'
                'print("source masked_point_fraction:", result.masked_point_fraction)\n'
                'print("effective model_masked_point_fraction:", result.model_masked_point_fraction)\n'
                'print("effective masked_patch_fraction:", result.masked_patch_fraction)'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>More, never fewer: the requested mask is combined with source missingness and rounded out to whole patches, which is why `model_masked_point_fraction` can exceed `masked_point_fraction`. And no supplied value changes: `imputed_value` replaces only source-missing or deliberately hidden cells; observed neighbours hidden only because they share a patch keep their original values.</details>'
            ),
        },
        {
            "md": (
                '## 7. Evaluate → evaluation report (masked-point metrics, a same-support baseline and the verdict)\n'
                '\n'
                "`masked_point_metrics()` computes **MAE** and **RMSE** only on deliberately hidden cells that had genuine source truth. The pooled figures average over every channel, so on a multi-channel input they mix units (temperature in degrees and vibration in its own unit on the sample) and are **not** in any single original unit; the cell therefore also prints MAE and RMSE **per channel** for the model and the baseline, each beside the channel's standard deviation, and the verdict is read from the per-channel comparison as well as the pooled ratio. The baseline is evaluated on the **same artificially withheld positions** from the source series with the held-out patch removed, and the cell names what it computed: a **linear interpolation baseline** between the observed neighbours when the holdout is interior, or a **last-value hold** when it is trailing (there is nothing to interpolate towards). It is a same-support descriptive comparison, not used to tune or select MOMENT, and one sample comparison does not establish superiority either way. The cell then prints a one-line **verdict** (model better than, about equal to, or worse than the baseline, with the ratio) and a **visible-point fidelity** check: the Pearson correlation between the model's reconstruction and the truth on the points it *could see*, beside the error of simply predicting the channel mean — a reconstruction that tracks its input has a correlation near 1. `evaluation_report` is the package's public evaluation stage and always produces a report: with a deliberate artificial mask it carries the `masked_point_metrics` values and the named baseline with the verdict `sample-sanity` (one deterministic holdout, no dispersion estimate); without a deliberate mask the verdict is `not-measurable`. Successful output means the verified reconstruction was scored on the declared holdout only and the verdict was printed, whichever way it went; the reported sample metrics are not stable estimates of domain performance. The report is written to `outputs/{stem}_evaluation_report.json`."
                '\n\n**Predict before running:** on a smooth synthetic series, will the pretrained model beat a straight line between the neighbours on the hidden patch? Write down the ratio you expect (model MAE ÷ baseline MAE) and which channel you expect to be harder.'
            ),
            "code": (
                'def error_summary(truth, prediction):\n'
                '    truth = np.asarray(truth, dtype=float)\n'
                '    prediction = np.asarray(prediction, dtype=float)\n'
                '    errors = prediction - truth\n'
                '    return {{"n": int(errors.size), "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2)))}}\n'
                '\n'
                '\n'
                'def compare_to_baseline(model_mae, baseline_mae):\n'
                '    """One-line verdict from the pooled MAE ratio; the thresholds only name the direction, they are not a test."""\n'
                '    if baseline_mae <= 0:\n'
                '        return float("inf"), "baseline error is zero; ratio undefined"\n'
                '    ratio = model_mae / baseline_mae\n'
                '    if ratio > 1.05:\n'
                '        return ratio, f"model WORSE than the baseline by {{ratio:.1f}}x on this sample"\n'
                '    if ratio < 0.95:\n'
                '        return ratio, f"model BETTER than the baseline by {{1.0 / ratio:.1f}}x on this sample"\n'
                '    return ratio, "model and baseline about equal on this sample (within 5 %)"\n'
                '\n'
                '\n'
                'def visible_point_fidelity(truth, reconstruction):\n'
                '    """Pearson correlation and MAE of the reconstruction on points the model could see, beside the mean predictor."""\n'
                '    truth = np.asarray(truth, dtype=float)\n'
                '    reconstruction = np.asarray(reconstruction, dtype=float)\n'
                '    pearson = float(np.corrcoef(truth, reconstruction)[0, 1]) if truth.size > 1 and truth.std() > 0 and reconstruction.std() > 0 else float("nan")\n'
                '    return {{"n": int(truth.size), "pearson": pearson, "mae_reconstruction": float(np.mean(np.abs(reconstruction - truth))), "mae_mean_predictor": float(np.mean(np.abs(truth.mean() - truth)))}}\n'
                '\n'
                '\n'
                'metrics = masked_point_metrics(result)\n'
                'print("tutorial masked-point MAE (pooled over channels):", metrics.mae)\n'
                'print("tutorial masked-point RMSE (pooled over channels):", metrics.rmse)\n'
                'print("scored held-out cells:", metrics.n)\n'
                '\n'
                'imputed_frame = result.to_frame()\n'
                'scored_window = imputed_frame[imputed_frame["window_id"] == windows.window_ids[0]]\n'
                'held = scored_window[scored_window["requested_hidden"] & scored_window["source_observed"]]\n'
                'baseline_name = "linear interpolation between the observed neighbours" if holdout_interior else "last-value hold (no observed neighbour after the holdout, so nothing to interpolate towards)"\n'
                'baseline_rows = []\n'
                'per_channel = {{}}\n'
                'for channel in windows.channels:\n'
                '    source = normalized[(normalized["series_id"] == windows.series_ids[0]) & (normalized["channel"] == channel)].sort_values("timestamp").copy()\n'
                '    held_timestamps = set(pd.Series(windows.timestamps[0][start:stop]).dropna().tolist())\n'
                '    if not held_timestamps:\n'
                '        continue\n'
                '    source["is_holdout"] = source["timestamp"].isin(held_timestamps)\n'
                '    truth = source.loc[source["is_holdout"], ["timestamp", "value"]].copy()\n'
                '    baseline_pred = source["value"].mask(source["is_holdout"]).interpolate(method="linear", limit_direction="both")\n'
                '    pred = baseline_pred[source["is_holdout"]].to_numpy(dtype=float)\n'
                '    for (_, truth_row), prediction in zip(truth.iterrows(), pred, strict=True):\n'
                '        baseline_rows.append({{"channel": channel, "timestamp": truth_row["timestamp"], "truth": float(truth_row["value"]), "prediction": float(prediction)}})\n'
                '    channel_held = held[held["channel"] == channel]\n'
                '    channel_std = float(source["value"].std(ddof=0))\n'
                '    per_channel[channel] = {{"channel_std": channel_std, "model": error_summary(channel_held["original_value"], channel_held["imputed_value"])}}\n'
                '    if np.isfinite(pred).all() and pred.size:\n'
                '        per_channel[channel]["baseline"] = error_summary(truth["value"], pred)\n'
                '        per_channel[channel]["baseline_predictions_all_equal"] = bool(np.allclose(pred, pred[0]))\n'
                'baseline_frame = pd.DataFrame(baseline_rows)\n'
                'if baseline_frame.empty or not np.isfinite(baseline_frame["prediction"]).all():\n'
                '    print("baseline unavailable: held-out region could not be interpolated from neighboring data")\n'
                '    baseline_metrics = None\n'
                '    verdict_line = "no baseline available on this input; no verdict"\n'
                '    mae_ratio = None\n'
                'else:\n'
                '    baseline_metrics = {{k: v for k, v in error_summary(baseline_frame["truth"], baseline_frame["prediction"]).items() if k != "n"}}\n'
                '    print(f"tutorial baseline ({{baseline_name}}):", baseline_metrics)\n'
                '    mae_ratio, verdict_line = compare_to_baseline(metrics.mae, baseline_metrics["mae"])\n'
                'print("per-channel masked-point errors (model vs baseline; channel_std = spread of the channel):")\n'
                'for channel, row in per_channel.items():\n'
                '    print("  ", channel, json.dumps(row))\n'
                'print("VERDICT:", verdict_line, f"(pooled MAE {{metrics.mae:.4f}} vs baseline {{baseline_metrics[\'mae\']:.4f}})" if baseline_metrics else "")\n'
                '\n'
                'visible_rows = scored_window[scored_window["source_observed"] & ~scored_window["model_hidden"]]\n'
                'fidelity = {{channel: visible_point_fidelity(rows["original_value"], rows["reconstruction"]) for channel, rows in visible_rows.groupby("channel", sort=False)}}\n'
                'print("visible-point fidelity (reconstruction of points the model could see; Pearson near 1 = tracks the input):")\n'
                'for channel, row in fidelity.items():\n'
                '    print("  ", channel, json.dumps(row))\n'
                '\n'
                'report = evaluation_report(result, model=pipe, sample_kind=sample_kind, baseline=baseline_metrics, baseline_id="linear_interpolation" if holdout_interior else "last_value_hold")\n'
                'report["verdict_line"] = verdict_line\n'
                'report["per_channel"] = per_channel\n'
                'report["visible_point_fidelity"] = fidelity\n'
                'with open("outputs/{stem}_evaluation_report.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(report, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(report, indent=2, default=str))\n'
                'if report["verdict"] == "not-measurable":\n'
                '    print("No deliberately hidden truth exists, so masked_point_metrics is not computed.")'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>On the recorded runs it did not. The Kaggle CPU run of 14 September 2026 (previous version, trailing holdout) printed pooled MAE 1.121 for the model against 0.171 for the baseline (a last-value hold) — 6.6× worse, temperature 2.047 vs 0.172 and vibration 0.196 vs 0.170 per channel — and the review of 2 October 2026 moved the 8-step holdout to every one of the 32 patch positions of this sample: the model lost to the baseline at all 32 (interior median MAE 0.99 vs 0.062). Its visible-point correlation was 0.087 (temperature) and 0.012 (vibration), so the reconstruction does not track the series either. Over 8 steps a smooth trend plus sinusoids is close to linear, so interpolation is a strong baseline here, but a correlation near zero on visible points is not explained by that: it means the pinned base checkpoint, called through this path, returns values near the series mean. Compare your `VERDICT` line and `fidelity` with these numbers. The verdict `sample-sanity` still applies: one holdout cannot rank the methods in general, and nothing here says what MOMENT does on other data.</details>\n\n**Change one thing (optional):** pick another interior candidate printed in Section 5, set `HOLDOUT_START` to it, and re-run Sections 5–9. Predict first: will the verdict flip, and will the baseline predictions still vary within a channel? Explain what you see from the per-channel table and the plot in Section 8.'
            ),
        },
        {
            "md": (
                '## 8. Visualize the held-out patch: truth, model and baseline\n'
                '\n'
                'One diagnostic plot per channel of the **scored window** (the series Section 5 held out, whichever series the CSV lists last), centred on the hidden patch: the source truth, the model reconstruction across the whole span (on the shaded held-out positions it is the imputed value; elsewhere it shows how closely the reconstruction tracks points the model could see), and the baseline on the held-out positions only. `imputed_value` replaces only source-missing or deliberately hidden cells; observed neighbours hidden from the model only because they share the patch are not overwritten. Successful output makes the verdict of Section 7 visually inspectable; it does not replace the machine-readable export.'
            ),
            "code": (
                'def write_line_svg(path, layers, *, title, width=760, height=280, shade=None, x_labels=("", "")):\n'
                '    """Polylines over a shared index; a value of None breaks the line. shade=(i0, i1) shades index positions i0..i1."""\n'
                '    all_values = [float(value) for _, values in layers for value in values if value is not None and np.isfinite(value)]\n'
                '    low, high = min(all_values), max(all_values)\n'
                '    span = high - low or 1.0\n'
                '    max_points = max(len(values) for _, values in layers)\n'
                '    left, right, top, bottom = 56, width - 20, 30, height - 38\n'
                '\n'
                '    def x_of(index):\n'
                '        return left + (right - left) * index / max(max_points - 1, 1)\n'
                '\n'
                '    def point(index, value):\n'
                '        return f"{{x_of(index):.1f}},{{bottom - (bottom - top) * (float(value) - low) / span:.1f}}"\n'
                '\n'
                '    svg = [f\'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}">\', f\'<text x="{{left}}" y="18" font-family="sans-serif" font-size="14">{{title}}</text>\']\n'
                '    if shade is not None:\n'
                '        x0, x1 = x_of(shade[0]), x_of(shade[1])\n'
                '        svg.append(f\'<rect x="{{x0:.1f}}" y="{{top}}" width="{{max(x1 - x0, 2.0):.1f}}" height="{{bottom - top}}" fill="#fde68a" fill-opacity="0.6"/>\')\n'
                '        svg.append(f\'<text x="{{x0:.1f}}" y="{{top - 4}}" font-family="sans-serif" font-size="11" fill="#92400e">held out</text>\')\n'
                '    svg.append(f\'<text x="4" y="{{top + 4}}" font-family="sans-serif" font-size="11">{{high:.3g}}</text>\')\n'
                '    svg.append(f\'<text x="4" y="{{bottom}}" font-family="sans-serif" font-size="11">{{low:.3g}}</text>\')\n'
                '    svg.append(f\'<text x="{{left}}" y="{{bottom + 14}}" font-family="sans-serif" font-size="11">{{x_labels[0]}}</text>\')\n'
                '    svg.append(f\'<text x="{{right - 150}}" y="{{bottom + 14}}" font-family="sans-serif" font-size="11">{{x_labels[1]}}</text>\')\n'
                '    for idx, (label, values) in enumerate(layers):\n'
                '        stroke = ["#111827", "#2563eb", "#dc2626"][idx % 3]\n'
                '        segment = []\n'
                '        for i, value in enumerate(values):\n'
                '            if value is None or not np.isfinite(value):\n'
                '                if len(segment) > 1:\n'
                '                    svg.append(f\'<polyline fill="none" stroke="{{stroke}}" stroke-width="2" points="{{" ".join(segment)}}"/>\')\n'
                '                segment = []\n'
                '                continue\n'
                '            segment.append(point(i, value))\n'
                '        if len(segment) > 1:\n'
                '            svg.append(f\'<polyline fill="none" stroke="{{stroke}}" stroke-width="2" points="{{" ".join(segment)}}"/>\')\n'
                '        svg.append(f\'<text x="{{left + 200 * idx}}" y="{{height - 10}}" font-family="sans-serif" font-size="12" fill="{{stroke}}">{{label}}</text>\')\n'
                '    svg.append("</svg>")\n'
                '    Path(path).parent.mkdir(parents=True, exist_ok=True)\n'
                '    Path(path).write_text("\\n".join(svg), encoding="utf-8")\n'
                '    return Path(path)\n'
                '\n'
                '\n'
                'held_stamps = set(pd.Series(windows.timestamps[0][start:stop]).dropna().tolist())\n'
                'plot_paths = []\n'
                'for channel in windows.channels:\n'
                '    view = scored_window[scored_window["channel"] == channel].sort_values("timestamp").reset_index(drop=True)\n'
                '    held_index = view.index[view["timestamp"].isin(held_stamps)].tolist()\n'
                '    lo, hi = max(held_index[0] - 24, 0), min(held_index[-1] + 25, len(view))\n'
                '    view = view.iloc[lo:hi].reset_index(drop=True)\n'
                '    held_index = view.index[view["timestamp"].isin(held_stamps)].tolist()\n'
                '    baseline_by_stamp = {{row["timestamp"]: row["prediction"] for row in baseline_rows if row["channel"] == channel}}\n'
                '    layers = [\n'
                '        ("truth (source)", view["original_value"].tolist()),\n'
                '        ("model reconstruction / imputed", view["reconstruction"].where(~view["model_hidden"], view["imputed_value"]).tolist()),\n'
                '        (f"baseline: {{baseline_name.split(\' (\')[0]}}", [baseline_by_stamp.get(stamp) for stamp in view["timestamp"]]),\n'
                '    ]\n'
                '    stamps = view["timestamp"]\n'
                '    plot_paths.append(write_line_svg(f"outputs/moment_imputation_{{channel}}.svg", layers, title=f"MOMENT imputation - series {{windows.series_ids[0]}}, channel {{channel}} (positions {{start}}-{{stop - 1}} held out)", shade=(held_index[0], held_index[-1]), x_labels=(str(stamps.iloc[0]), str(stamps.iloc[-1]))))\n'
                'plot_path = plot_paths[0]\n'
                'try:\n'
                '    from IPython.display import SVG, display\n'
                '\n'
                '    for path in plot_paths:\n'
                '        display(SVG(filename=str(path)))\n'
                'except ImportError:\n'
                '    print("SVGs written:", [str(path) for path in plot_paths])\n'
                'print(imputed_frame.loc[imputed_frame["requested_hidden"], ["series_id", "timestamp", "channel", "original_value", "imputed_value", "reconstruction", "model_hidden"]].head(16))'
            ),
        },
        {
            "md": (
                '## 9. Export outputs and provenance\n'
                '\n'
                "This stage writes the stable machine-readable handoff files: `outputs/moment_imputed_series.csv` (observed values preserved, imputed cells flagged), `outputs/moment_imputation_metrics.json`, `outputs/moment_imputation_provenance.json` (plus the per-channel diagnostic SVGs from the previous stage), and `outputs/{stem}_result.json` with the input manifest, the evaluation report, the sample identity and digest, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, and the runtime identity. Successful completion establishes that the imputed product, the sample-evaluation procedure/baseline, and model/runtime/data provenance were serialized under documented filenames; file creation does **not** establish generalized model quality. No credentials are recorded."
            ),
            "code": (
                'imputed_frame.to_csv("outputs/moment_imputed_series.csv", index=False)\n'
                'metric_payload = {{"estimation_procedure": "single deterministic artificial 8-step patch holdout on the first canonical window", "sample_evidence_only": True, "n": metrics.n, "mae": metrics.mae, "rmse": metrics.rmse, "pooled_over_channels": True, "per_channel": per_channel, "baseline_method": baseline_name, "linear_interpolation_baseline": baseline_metrics if holdout_interior else None, "last_value_hold_baseline": None if holdout_interior else baseline_metrics, "mae_ratio_model_over_baseline": mae_ratio, "verdict": verdict_line, "visible_point_fidelity": fidelity, "holdout": {{"window_id": windows.window_ids[0], "series_id": windows.series_ids[0], "start": start, "stop_inclusive": stop - 1, "placement": holdout_placement}}}}\n'
                'with open("outputs/moment_imputation_metrics.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(metric_payload, handle, indent=2)\n'
                'provenance["data"] = sample_identity\n'
                'provenance["evaluation"] = metric_payload\n'
                'provenance["notebook_source"] = NOTEBOOK_SOURCE\n'
                'with open("outputs/moment_imputation_provenance.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(provenance, handle, indent=2, default=str)\n'
                'payload_out = {{\n'
                '    "result": {{"n_windows": len(result.window_ids), "held_out_positions": [start, stop - 1], "holdout_placement": holdout_placement, "masked_point_fraction": result.masked_point_fraction, "model_masked_point_fraction": result.model_masked_point_fraction, "masked_patch_fraction": result.masked_patch_fraction, "metrics": metric_payload}},\n'
                '    "evaluation_report": report,\n'
                '    "input_manifest": input_manifest,\n'
                '    "sample": {{"kind": sample_kind, **sample_identity}},\n'
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                '    "model_id": MODEL_ID,\n'
                '    "model_revision": MODEL_REVISION,\n'
                '    "model_license": MODEL_LICENSE,\n'
                '    "runtime": {{"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "pandas": pandas.__version__, "numpy": numpy.__version__, "device": pipe.identity.device, "dtype": pipe.identity.dtype}},\n'
                '}}\n'
                'with open("outputs/moment_imputation_result.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(payload_out, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(sorted(os.listdir("outputs")))'
            ),
        },
    ],
    "closing": (
        '## Interpretation and limits\n'
        '\n'
        "A successful run proves that the carried package can validate this input, reconstruct with the pinned MOMENT checkpoint, keep source missingness separate from artificial evaluation masking, score only known withheld truth, preserve observed values in the imputed product, and export results and provenance. Successful execution proves that the recorded repository revision's package, carried in this notebook, can do exactly that — without the repository being reachable — and no more.\n"
        '\n'
        '**The recorded result is a negative one, and the notebook says so.** On the synthetic sample the pinned MOMENT-1-base checkpoint, called through this path, **did not beat the naive baseline on any recorded run**: the Kaggle CPU run of 14 September 2026 (previous version, trailing holdout) scored pooled MAE 1.121 against 0.171 for a last-value hold; the review of 2 October 2026 (local CPU, same weights, transformers 5.16.1 and upstream\'s own 4.33.3 giving the same numbers to seven significant figures) rotated the holdout over all 32 patch positions and the model lost at every one (interior median 0.99 vs 0.062), and its reconstruction of points it could see correlated with the truth at 0.087 and 0.012 — no better than predicting the series mean. The head tensors are proven live against the pinned file, so this is what the pinned base checkpoint returns here, not a loading error; the cause (checkpoint behaviour, or the upstream call convention for the base model) was not determined and is listed as an open repository question. Read your own `VERDICT` line and `fidelity` figures the same way: if the ratio is above 1, the imputed values on this sample are **not** better than the interpolation between neighbours, and the exported imputed product should be read as a demonstration of the contract (masking, scoring, preservation, provenance), not as a usable imputation of this series.\n'
        '\n'
        'It **does not prove** that the displayed MAE/RMSE generalize to other series or domains, that MOMENT loses (or wins) against interpolation in general, or that the model supplies calibrated per-prediction uncertainty; the evaluation report says `sample-sanity` for that reason. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain. This path returns point reconstructions only; no uncertainty interval is provided.\n'
        '\n'
        '**Next experiments:** move `HOLDOUT_START` over several interior candidates and tabulate the per-channel MAE of model and baseline (the review\'s 32-position sweep takes under 10 s on CPU); repeat domain-appropriate masking over an independent dataset, report dispersion across windows/series, and compare multiple baselines without using the evaluation set for model selection; enable `USE_BYOD` with a series that has genuine source gaps and read how `masked_point_fraction` and `model_masked_point_fraction` diverge — note the **cross-channel patch coupling**: when one channel is missing at a position, MOMENT hides that whole 8-step patch for *every* channel, so observed points of the other channels are hidden from the model (and reported as `model_hidden`, never overwritten), which is why `model_masked_point_fraction` exceeds the source fraction.\n'
        '\n'
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **Section 1 fails while building `momentfm`** — the one source dependency is built from its pinned upstream commit and needs `git` and access to its Git host; run Section 1 again, or use Colab or Kaggle (both ship `git`).\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        '- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n'
        '- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, copy the CSV into the runtime and set `BYOD_PATH`.\n'
        '- **BYOD: "Upload exactly one CSV"** — the dialog was cancelled or several files were chosen; run Section 4 again.\n'
        '- **BYOD: a `ValidationError`** — it carries a code and names the column, series or rule (duplicate header, duplicate row, a ceiling, an all-missing channel); fix the CSV.\n'
        '- **Section 5: "no complete 8-step source-observed patch to hold out"** — the first series needs at least 8 consecutive timestamps observed in every channel; fill or drop the gaps, or reorder the CSV so a complete series comes first.\n'
        '- **Section 5: "HOLDOUT_START=… is not a complete … patch start"** — use -1 or one of the candidate starts the message lists.\n'
        '- **Section 7 prints "baseline unavailable"** — the holdout had no observed neighbour to interpolate from; choose an interior candidate.\n\n'
        '## Glossary\n\n'
        '- **Long-format table** — one row per `(series_id, timestamp, channel, value)`; the package turns it into windows.\n'
        '- **Window / patch** — the 512-step input MOMENT reads per series, cut into 64 non-overlapping 8-step patches.\n'
        '- **Left padding / truncation** — a series shorter than 512 steps is padded at the start and the padding masked out; a longer one keeps its final 512 timestamps.\n'
        '- **Instance normalisation** — each window is rescaled by its own mean and spread before the encoder, so the absolute level is removed.\n'
        '- **Input manifest** — the record of what validation saw and changed (padding, truncation, missingness) before the model ran.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **Visibility mask / patch quantization** — which positions the model may see; MOMENT hides whole 8-step patches, so a requested hidden point hides its patch neighbours too.\n'
        '- **MAE / RMSE** — mean absolute error and root-mean-square error, computed only on deliberately hidden points with known truth; per channel they are in that channel\'s units, pooled over channels they are not.\n'
        '- **Interior / trailing holdout** — a hidden patch with observed points on both sides (a gap) versus one at the end of the series (an extrapolation).\n'
        '- **Interpolation baseline / last-value hold** — a straight line between the nearest visible neighbours, or (for a trailing holdout) the last observed value repeated; scored on the same hidden points.\n'
        '- **Verdict / MAE ratio** — model MAE ÷ baseline MAE on the hidden points; above 1 the model did worse than the naive baseline.\n'
        '- **Visible-point fidelity** — the correlation between the reconstruction and the truth on points the model could see; near 1 means the output tracks the input, near 0 means it does not.\n'
        '- **`sample-sanity`** — the evaluation verdict for one deterministic holdout: a check, not an estimate.\n'
        '- **BYOD** — bring your own data: your long-format CSV through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- The model hid ___ points for the ___ I requested (`model_masked_point_fraction` vs `masked_point_fraction`).\n'
        '- The holdout was at positions ___–___ (interior / trailing); MOMENT scored MAE ___ / RMSE ___ against the ___ baseline ___ / ___; the verdict line said ___ (ratio ___).\n'
        '- Visible-point correlation was ___ (temperature) and ___ (vibration), so the reconstruction does / does not track the series.\n'
        '- One reason one holdout cannot rank the two methods in general: ___.\n'
        '\n'
        '## References\n'
        '\n'
        '- Repository README: https://github.com/kurtvalcorza/moment-pipeline/blob/main/README.md\n'
        '- Repository model card: https://github.com/kurtvalcorza/moment-pipeline/blob/main/MODEL_CARD.md\n'
        '- Sample dataset card: https://github.com/kurtvalcorza/moment-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n'
        '- Upstream model: https://huggingface.co/{MODEL_ID}\n'
        '- Upstream library: https://github.com/moment-timeseries-foundation-model/moment\n'
        '- MOMENT: A Family of Open Time-series Foundation Models: https://arxiv.org/abs/2402.03885'
    ),
}
