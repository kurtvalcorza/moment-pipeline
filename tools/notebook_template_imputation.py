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
        '**Who this notebook is for.** A learner who knows basic Python and pandas, has used Colab or Jupyter and has met time-series data, and wants to see how a pretrained time-series foundation model is used without any training to fill hidden values — what goes in, what comes out, and what the output does and does not prove. The audience is students and practitioners preparing their own sensor or monitoring series; no prior experience with MOMENT is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | a long-format CSV: the synthetic two-channel series `A` (256 steps) or your own, plus an explicit visibility mask that hides one stretch of known values |\n| Model | the MOMENT-1-base reconstruction head (`task="reconstruction"`), with masking applied per 8-step patch |\n| Output | an imputed series in which only missing or deliberately hidden cells are replaced, MAE and RMSE on the hidden cells beside a linear-interpolation baseline, and a `sample-sanity` report |\n\n**How to use this notebook.** Choose a runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook\'s own Python, so no restart is needed (the recorded hosted runs of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the default path. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer. No per-cell outputs of the hosted runs are recorded in this repository (only their pass/fail), so the worked answers state what the code and the sample guarantee rather than quoting numbers; compare them with what your run prints. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the synthetic sample (or your CSV) → 5 validation, canonicalization and the evaluation mask *(core concept: windows, padding, masks)* → 6 impute *(core concept: patch-quantized masking)* → 7 masked-point metrics against an interpolation baseline *(evaluation practice)* → 8 the plot → 9 → export and provenance *(engineering)* → conclude.'
    )]},
    'learning_objectives': (
        'install the pinned runtime; read what the carried package guarantees; resolve and digest-verify the immutable model revision; generate the deterministic synthetic sample or bring your own long-format CSV; validate and canonicalize the input into an input manifest; hide one complete 8-step patch with known truth; impute and evaluate only withheld truth through an evaluation report; compare a simple interpolation baseline; and export the imputed series, metrics, and provenance. By the end of this notebook you will be able to do each of these without the repository being reachable.'
    ),
    'exclusions': (
        'forecasting, classification, stochastic uncertainty intervals, or production fitness. This path returns point reconstructions only; no uncertainty interval is provided.'
    ),
    'prerequisites': [
        '- **Learner:** basic Python and pandas familiarity; no prior experience with MOMENT or time-series foundation models. Windows, patches, padding, masks and the evaluation verdicts are explained where they are first used and again in the Glossary.',
        '- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (plus `momentfm`, built from its pinned commit), so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported.',
        '- **Compute:** CPU is the default path and no GPU is required; CUDA is used automatically when available. The public v1 API accepts `float32` only. The pinned `torch==2.14.0` install is the largest download of the run, followed by the ~454 MB `model.safetensors`.',
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
                "Before the model runs, the cell prints the effective runtime and the operational ceilings — the `ResourceLimits` (rows, series, channels, windows), the fixed 512-step window and 8-step patch — and the device/precision policy. `validate_inputs` is the package's public validation stage: it runs exactly the two calls every task path makes (`validate_long_frame`, then `to_windows`), so it raises exactly what canonicalization would raise, and returns an **input manifest** naming the schema and ceilings, each window's series, valid positions, padding and truncation, the source-missingness fractions, and the verdict; it is written to `outputs/moment_imputation_input_manifest.json`. To show what rejection looks like, the cell also validates a deliberately broken copy (a channel with no observed value) and records the pipeline's own error code as a finding. The canonical `WindowSet` used by the model is built by the same calls; any padding, truncation, source missingness, or irregular frequency is disclosed here before model execution. Successful output means the input passed validation and every canonicalization effect is disclosed before the model runs. The cell then selects the latest complete 8-step patch in the first window for which every channel has source-observed truth and hides it — the **masking unit is the 8-step patch**, and this separates **source-missing data from artificial evaluation masking**: the artificial holdout consists only of genuine known truth."
                '\n\n**Predict before running:** the cell hides one stretch of known values for evaluation. How many points will be held out per channel, and why that number?'
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
                'visible = np.ones_like(windows.input_mask, dtype=np.float32)\n'
                'complete_patch_starts = []\n'
                'for candidate in range(0, windows.sequence_length, windows.patch_length):\n'
                '    stop = candidate + windows.patch_length\n'
                '    if np.all(windows.input_mask[0, candidate:stop] == 1) and np.all(windows.point_mask[0, :, candidate:stop] == 1):\n'
                '        complete_patch_starts.append(candidate)\n'
                'if not complete_patch_starts:\n'
                '    raise ValueError("The first canonical window has no complete 8-step source-observed patch to hold out. Provide a series with at least 8 consecutive observed timestamps.")\n'
                'start = complete_patch_starts[-1]\n'
                'stop = start + windows.patch_length\n'
                'visible[0, start:stop] = 0.0\n'
                'print("artificial holdout window:", windows.window_ids[0])\n'
                'print("deliberately hidden positions:", start, "through", stop - 1)\n'
                'print("held-out point count across channels:", windows.n_channels * windows.patch_length)'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>One 8-step patch per channel: MOMENT masks whole patches, so the evaluation holdout is one patch length (8 steps) per channel — the cell prints the held-out point count as channels × patch length, 2 × 8 = 16 on the two-channel sample. Hiding less than a patch is impossible for this model; hiding a single point would hide its whole patch.</details>'
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
                '## 7. Evaluate → evaluation report (masked-point metrics and an interpolation baseline)\n'
                '\n'
                "`masked_point_metrics()` computes **MAE** and **RMSE** only on deliberately hidden cells that had genuine source truth — **MAE** is average absolute error in the original units; **RMSE** is also in the original units but weights larger errors more strongly. A linear interpolation baseline is evaluated on the **same artificially withheld positions**, using the source series with the held-out patch removed; it is a same-support descriptive comparison, not used to tune or select MOMENT, and one sample comparison does not establish superiority. `evaluation_report` is the package's public evaluation stage and always produces a report: with a deliberate artificial mask it carries the `masked_point_metrics` values and the baseline with the verdict `sample-sanity` (one deterministic holdout, no dispersion estimate); without a deliberate mask the verdict is `not-measurable` and the report says what would make the task measurable. Successful output means the verified reconstruction was scored on the declared holdout only; the reported sample metrics are not stable estimates of domain performance. The report is written to `outputs/{stem}_evaluation_report.json`."
                '\n\n**Predict before running:** on a smooth synthetic series, will the pretrained model beat a straight line between the neighbours on the hidden patch?'
            ),
            "code": (
                'metrics = masked_point_metrics(result)\n'
                'print("tutorial masked-point MAE:", metrics.mae)\n'
                'print("tutorial masked-point RMSE:", metrics.rmse)\n'
                'print("scored held-out cells:", metrics.n)\n'
                '\n'
                'baseline_rows = []\n'
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
                'baseline_frame = pd.DataFrame(baseline_rows)\n'
                'if baseline_frame.empty or not np.isfinite(baseline_frame["prediction"]).all():\n'
                '    print("baseline unavailable: held-out region could not be interpolated from neighboring data")\n'
                '    baseline_metrics = None\n'
                'else:\n'
                '    errors = baseline_frame["prediction"].to_numpy(dtype=float) - baseline_frame["truth"].to_numpy(dtype=float)\n'
                '    baseline_metrics = {{"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2)))}}\n'
                '    print("linear-interpolation tutorial baseline:", baseline_metrics)\n'
                '\n'
                'report = evaluation_report(result, model=pipe, sample_kind=sample_kind, baseline=baseline_metrics)\n'
                'with open("outputs/{stem}_evaluation_report.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(report, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(report, indent=2, default=str))\n'
                'if report["verdict"] == "not-measurable":\n'
                '    print("No deliberately hidden truth exists, so masked_point_metrics is not computed.")'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Not necessarily, and the notebook does not claim it: over 8 steps a smooth trend plus sinusoids is close to linear, so interpolation can be hard to beat. The report compares both on the same hidden points with the verdict `sample-sanity` — one deterministic holdout with no dispersion estimate cannot rank the methods, whichever wins.</details>'
            ),
        },
        {
            "md": (
                '## 8. Visualize original versus imputed values\n'
                '\n'
                'This diagnostic plot compares the source series with the returned imputed product. `imputed_value` replaces only source-missing or deliberately hidden cells; observed neighbours hidden from the model only because they share the patch are not overwritten. Successful output makes that product behaviour visually inspectable but does not replace the machine-readable export.'
            ),
            "code": (
                'def write_line_svg(path, layers, *, title, width=760, height=280):\n'
                '    all_values = [float(value) for _, values in layers for value in values]\n'
                '    low, high = min(all_values), max(all_values)\n'
                '    span = high - low or 1.0\n'
                '    max_points = max(len(values) for _, values in layers)\n'
                '    left, right, top, bottom = 48, width - 20, 30, height - 38\n'
                '\n'
                '    def point(index, value):\n'
                '        x = left + (right - left) * index / max(max_points - 1, 1)\n'
                '        y = bottom - (bottom - top) * (float(value) - low) / span\n'
                '        return f"{{x:.1f}},{{y:.1f}}"\n'
                '\n'
                '    svg = [f\'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}">\', f\'<text x="{{left}}" y="18" font-family="sans-serif" font-size="14">{{title}}</text>\']\n'
                '    for idx, (label, values) in enumerate(layers):\n'
                '        points = " ".join(point(i, value) for i, value in enumerate(values))\n'
                '        stroke = ["#111827", "#2563eb", "#dc2626"][idx % 3]\n'
                '        svg.append(f\'<polyline fill="none" stroke="{{stroke}}" stroke-width="2" points="{{points}}"/>\')\n'
                '        svg.append(f\'<text x="{{left + 180 * idx}}" y="{{height - 10}}" font-family="sans-serif" font-size="12" fill="{{stroke}}">{{label}}</text>\')\n'
                '    svg.append("</svg>")\n'
                '    Path(path).parent.mkdir(parents=True, exist_ok=True)\n'
                '    Path(path).write_text("\\n".join(svg), encoding="utf-8")\n'
                '    return Path(path)\n'
                '\n'
                '\n'
                'imputed_frame = result.to_frame()\n'
                'channel = windows.channels[0]\n'
                'view = imputed_frame[imputed_frame["channel"] == channel].tail(96)\n'
                'plot_path = write_line_svg("outputs/moment_imputation.svg", [("original", view["original_value"].fillna(view["imputed_value"]).tolist()), ("imputed", view["imputed_value"].tolist())], title=f"MOMENT imputation - {{channel}}")\n'
                'try:\n'
                '    from IPython.display import SVG, display\n'
                '\n'
                '    display(SVG(filename=str(plot_path)))\n'
                'except ImportError:\n'
                '    print(f"SVG written to {{plot_path}}")\n'
                'print(imputed_frame.loc[imputed_frame["requested_hidden"], ["series_id", "timestamp", "channel", "original_value", "imputed_value", "model_hidden"]].head(16))'
            ),
        },
        {
            "md": (
                '## 9. Export outputs and provenance\n'
                '\n'
                "This stage writes the stable machine-readable handoff files: `outputs/moment_imputed_series.csv` (observed values preserved, imputed cells flagged), `outputs/moment_imputation_metrics.json`, `outputs/moment_imputation_provenance.json` (plus the diagnostic SVG from the previous stage), and `outputs/{stem}_result.json` with the input manifest, the evaluation report, the sample identity and digest, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, and the runtime identity. Successful completion establishes that the imputed product, the sample-evaluation procedure/baseline, and model/runtime/data provenance were serialized under documented filenames; file creation does **not** establish generalized model quality. No credentials are recorded."
            ),
            "code": (
                'imputed_frame.to_csv("outputs/moment_imputed_series.csv", index=False)\n'
                'metric_payload = {{"estimation_procedure": "single deterministic artificial 8-step patch holdout", "sample_evidence_only": True, "n": metrics.n, "mae": metrics.mae, "rmse": metrics.rmse, "linear_interpolation_baseline": baseline_metrics}}\n'
                'with open("outputs/moment_imputation_metrics.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(metric_payload, handle, indent=2)\n'
                'provenance["data"] = sample_identity\n'
                'provenance["evaluation"] = metric_payload\n'
                'provenance["notebook_source"] = NOTEBOOK_SOURCE\n'
                'with open("outputs/moment_imputation_provenance.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(provenance, handle, indent=2, default=str)\n'
                'payload_out = {{\n'
                '    "result": {{"n_windows": len(result.window_ids), "held_out_positions": [start, stop - 1], "masked_point_fraction": result.masked_point_fraction, "model_masked_point_fraction": result.model_masked_point_fraction, "masked_patch_fraction": result.masked_patch_fraction, "metrics": metric_payload}},\n'
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
        'It **does not prove** that the displayed MAE/RMSE generalize to other series or domains, that MOMENT beats the interpolation baseline reliably, or that the model supplies calibrated per-prediction uncertainty; the evaluation report says `sample-sanity` for that reason. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain. This path returns point reconstructions only; no uncertainty interval is provided.\n'
        '\n'
        '**Next experiments:** repeat domain-appropriate masking over an independent dataset, report dispersion across windows/series, and compare multiple baselines without using the evaluation set for model selection; enable `USE_BYOD` with a series that has genuine source gaps and read how `masked_point_fraction` and `model_masked_point_fraction` diverge.\n'
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
        '- **BYOD: a `ValidationError`** — it carries a code and names the column, series or rule (duplicate header, duplicate row, a ceiling, an all-missing channel); fix the CSV.\n\n'
        '## Glossary\n\n'
        '- **Long-format table** — one row per `(series_id, timestamp, channel, value)`; the package turns it into windows.\n'
        '- **Window / patch** — the 512-step input MOMENT reads per series, cut into 64 non-overlapping 8-step patches.\n'
        '- **Left padding / truncation** — a series shorter than 512 steps is padded at the start and the padding masked out; a longer one keeps its final 512 timestamps.\n'
        '- **Instance normalisation** — each window is rescaled by its own mean and spread before the encoder, so the absolute level is removed.\n'
        '- **Input manifest** — the record of what validation saw and changed (padding, truncation, missingness) before the model ran.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **Visibility mask / patch quantization** — which positions the model may see; MOMENT hides whole 8-step patches, so a requested hidden point hides its patch neighbours too.\n'
        '- **MAE / RMSE** — mean absolute error and root-mean-square error in the original units, computed only on deliberately hidden points with known truth.\n'
        '- **Interpolation baseline** — a straight line between the nearest visible neighbours, scored on the same hidden points.\n'
        '- **`sample-sanity`** — the evaluation verdict for one deterministic holdout: a check, not an estimate.\n'
        '- **BYOD** — bring your own data: your long-format CSV through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- The model hid ___ points for the ___ I requested (`model_masked_point_fraction` vs `masked_point_fraction`).\n'
        '- MOMENT scored MAE ___ / RMSE ___ against the interpolation baseline ___ / ___.\n'
        '- One reason one holdout cannot rank the two methods: ___.\n'
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
